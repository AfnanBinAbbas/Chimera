#!/usr/bin/env python3
"""Evaluate all saved checkpoints in `checkpoint/` and pick the best by AD F1.

Usage:
  python scripts/select_best_checkpoint.py --dataset BGL --device cpu --checkpoint-dir checkpoint

This script loads every `*_model.bin` file under the checkpoint directory, infers
which model class it represents (by filename), evaluates AD/RCA metrics on the
BGL test loaders using `src.metrics_eval.compute_ad_rca_metrics`, and writes a
copy of the best checkpoint to `checkpoint/best_model.bin`.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path
import importlib
import torch

# repo root
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from scripts.generate_project_report import build_args


def find_model_class_by_name(name: str):
    # Map simple class names to their modules
    mapping = {
        "Chimera": ("src.models", "Chimera"),
        "MultiTaskChimera": ("src.multitask_expansion", "MultiTaskChimera"),
        "AdvancedInteractionChimera": ("src.advanced_interaction", "AdvancedInteractionChimera"),
    }
    if name in mapping:
        mod_name, cls_name = mapping[name]
        module = importlib.import_module(mod_name)
        return getattr(module, cls_name)
    return None


def evaluate_checkpoint_file(ckpt_path: Path, args_ns):
    from src.dataset import get_loader, load_json
    from src.metrics_eval import compute_ad_rca_metrics, compute_aux_multitask_metrics

    # prepare loaders
    data = lambda rel: ROOT / "data" / args_ns.dataset / rel
    n_loader = get_loader(str(data("n_test.txt")), str(data("n_test.txt")), batch_size=args_ns.batch_size, shuffle=False, num_workers=0)
    an_loader = get_loader(str(data("an_test.txt")), str(data("an_test.txt")), batch_size=args_ns.batch_size, shuffle=False, num_workers=0)

    emb = load_json(str(data("emd_dict.json")))

    name = ckpt_path.name.replace("_model.bin", "")
    ModelClass = find_model_class_by_name(name)
    if ModelClass is None:
        print(f"Skipping unknown checkpoint {ckpt_path.name}")
        return None

    model = ModelClass(args_ns, emb).to(args_ns.hard_device)
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=args_ns.hard_device))
    except Exception as e:
        print(f"Failed to load {ckpt_path}: {e}")
        return None

    metrics = compute_ad_rca_metrics(model, n_loader, an_loader)
    # attach aux if available
    try:
        aux_m = compute_aux_multitask_metrics(model, an_loader)
        metrics["aux_weak_labels"] = aux_m
    except Exception:
        pass
    metrics["checkpoint_file"] = str(ckpt_path)
    metrics["checkpoint_label"] = name
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BGL")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--checkpoint-dir", default="checkpoint")
    args = parser.parse_args()

    args_ns = build_args(args.dataset, args.device, args.batch_size)
    args_ns.dataset = args.dataset

    ckpt_dir = Path(args.checkpoint_dir)
    if not ckpt_dir.exists():
        print("Checkpoint dir not found:", ckpt_dir)
        return

    cand = list(ckpt_dir.glob("*_model.bin"))
    if not cand:
        print("No *_model.bin files found in", ckpt_dir)
        return

    best = None
    best_score = -1.0
    results = []
    for c in cand:
        print("Evaluating", c.name)
        metrics = evaluate_checkpoint_file(c, args_ns)
        if metrics is None:
            continue
        results.append(metrics)
        score = metrics.get("ad_f1", 0.0)
        print(f" - ad_f1={score:.4f}")
        if score > best_score:
            best_score = score
            best = c

    if best is None:
        print("No valid checkpoints evaluated.")
        return

    # copy best
    dst = ckpt_dir / "best_model.bin"
    shutil.copy2(best, dst)
    print(f"Selected best checkpoint: {best.name} (AD F1={best_score:.4f}) -> {dst}")

    # also write a small JSON summary
    out = ROOT / "report_output" / "selected_checkpoint_summary.json"
    import json
    with open(out, "w") as f:
        json.dump({"best_checkpoint": str(best), "best_ad_f1": best_score, "all": results}, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
