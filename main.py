import argparse
import os
import time
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src.dataset import get_loader, load_json
from src.models import Chimera
from src.domain_adaptation import DomainAdaptiveChimera
from src.multitask_expansion import MultiTaskChimera
from src.advanced_interaction import AdvancedInteractionChimera
from src.metrics_eval import compute_ad_rca_metrics
from src.callbacks import ValidationMetricsCallback
from pytorch_lightning.callbacks import ModelCheckpoint
from src.unified_benchmark import evaluate_unified_benchmark
from src.utils import get_abs_path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def select_model_class(mode: str):
    if mode.startswith("novelty1"):
        return DomainAdaptiveChimera
    if mode.startswith("novelty2"):
        return MultiTaskChimera
    if mode.startswith("novelty3"):
        return AdvancedInteractionChimera
    return Chimera


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default="cuda", type=str)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--mode", default="train", type=str)
    parser.add_argument("--dataset", default="BGL", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--warmup_epochs", default=8, type=int)
    parser.add_argument("--model_save_path", default="checkpoint", type=str)
    parser.add_argument("--load_checkpoint", nargs="?", const=True, default=False, type=str2bool)
    parser.add_argument("--threshold", default=0.5, type=float)
    parser.add_argument("--auto_threshold", default=True, type=str2bool)
    return parser.parse_args()


def build_loaders(args):
    root = f"data/{args.dataset}"
    train_loader = get_loader(f"{root}/n_train.txt", f"{root}/an_train.txt", batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = get_loader(f"{root}/n_dev.txt", f"{root}/an_dev.txt", batch_size=args.batch_size, shuffle=False, num_workers=4)
    n_eval_loader = get_loader(f"{root}/n_test.txt", f"{root}/n_test.txt", batch_size=args.batch_size, shuffle=False, num_workers=4)
    an_eval_loader = get_loader(f"{root}/an_test.txt", f"{root}/an_test.txt", batch_size=args.batch_size, shuffle=False, num_workers=4)
    return train_loader, valid_loader, n_eval_loader, an_eval_loader


def calibrate_threshold(model, normal_loader, anomaly_loader):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for batch in normal_loader:
            _, pair = model(batch)
            probs = torch.softmax(pair[0], dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
            labels.extend([0] * len(probs))
        for batch in anomaly_loader:
            _, pair = model(batch)
            probs = torch.softmax(pair[0], dim=-1)[:, 1]
            scores.extend(probs.cpu().tolist())
            labels.extend([1] * len(probs))

    if not scores:
        return 0.5

    scores = np.asarray(scores)
    labels = np.asarray(labels)
    candidates = np.linspace(0.05, 0.95, 91)
    best_acc, best_t = -1.0, 0.5
    for c in candidates:
        pred = scores >= c
        tp = ((pred == 1) & (labels == 1)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
        if acc > best_acc:
            best_acc = acc
            best_t = float(c)
    return best_t


def main():
    args = parse_args()
    if args.hard_device != "cpu" and not torch.cuda.is_available():
        args.hard_device = "cpu"

    device = torch.device("cpu" if args.hard_device == "cpu" else f"cuda:{args.gpu_index}")

    model_cls = select_model_class(args.mode)
    embedding = load_json(f"data/{args.dataset}/emd_dict.json")
    model = model_cls(args, embedding)
    model.to(device)

    os.makedirs(args.model_save_path, exist_ok=True)

    train_loader, valid_loader, n_eval_loader, an_eval_loader = build_loaders(args)

    if args.auto_threshold and args.mode.endswith("eval"):
        try:
            t = calibrate_threshold(model, valid_loader, an_eval_loader)
            model.decision_threshold = t
            print(f"Calibrated threshold: {t:.4f}")
        except Exception as e:
            print(f"Calibration failed: {e}")
            model.decision_threshold = args.threshold
    else:
        model.decision_threshold = args.threshold

    accelerator = "cpu" if args.hard_device == "cpu" else "gpu"
    devices = 1 if args.hard_device == "cpu" else [args.gpu_index]

    # Validation AD F1 callback + checkpointing by val_ad_f1
    val_callback = ValidationMetricsCallback(
        n_val_fp=f"data/{args.dataset}/n_dev.txt",
        an_val_fp=f"data/{args.dataset}/an_dev.txt",
        batch_size=args.batch_size,
        device=device,
    )
    checkpoint_cb = ModelCheckpoint(dirpath=args.model_save_path, filename=f"{model.__class__.__name__}_best", monitor="val_ad_f1", mode="max", save_top_k=1)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[EarlyStopping(monitor="val_ad_f1", patience=150, mode="max"), val_callback, checkpoint_cb],
    )

    checkpoint_path = get_abs_path("checkpoint", f"{model.__class__.__name__}_model.bin")
    if args.load_checkpoint and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.mode in {"eval", "novelty1_eval", "novelty2_eval", "novelty3_eval"}:
        metrics = compute_ad_rca_metrics(model, n_eval_loader, an_eval_loader)
        print(metrics)
        return

    if args.mode in {"train", "novelty1_train", "novelty2_train", "novelty3_train"}:
        start = time.time()
        trainer.fit(model, train_loader, valid_loader)
        print("training time:", time.time() - start)
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        return

    if args.mode == "unified_benchmark":
        results = evaluate_unified_benchmark(model, args.dataset, batch_size=args.batch_size)
        print(results)
        return

    trainer.test(model, dataloaders=[n_eval_loader, an_eval_loader])


if __name__ == "__main__":
    main()
