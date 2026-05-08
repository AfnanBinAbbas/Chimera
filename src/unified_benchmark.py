import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score

from src.metrics_eval import compute_ad_rca_metrics
from src.multitask_dataset import (
    has_explicit_multitask_labels,
    get_multitask_eval_loader,
    get_weak_multitask_eval_loader,
)


def _safe_mean(x):
    return float(np.mean(x)) if len(x) > 0 else 0.0


def evaluate_unified_benchmark(model, dataset_name, batch_size=256, num_workers=4):
    """
    Novelty 4: Unified evaluation benchmark for one dataset.
    AD + RCA match ``main.run_eval`` / ``metrics_eval.compute_ad_rca_metrics``.
    Auxiliary metrics use weak or explicit multi-task labels as in other eval paths.
    """
    root = f"data/{dataset_name}"
    use_explicit = has_explicit_multitask_labels(root)
    if use_explicit:
        n_loader = get_multitask_eval_loader(f"{root}/n_test_mt.txt", batch_size=batch_size, shuffle=False, num_workers=num_workers)
        an_loader = get_multitask_eval_loader(f"{root}/an_test_mt.txt", batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        n_loader = get_weak_multitask_eval_loader(f"{root}/n_test.txt", batch_size=batch_size, shuffle=False, num_workers=num_workers)
        an_loader = get_weak_multitask_eval_loader(f"{root}/an_test.txt", batch_size=batch_size, shuffle=False, num_workers=num_workers)

    core = compute_ad_rca_metrics(model, n_loader, an_loader)
    rca = core.get("rca", {})

    failure_preds = []
    failure_targets = []
    impact_mse = []
    rem_preds = []
    rem_targets = []

    model.eval()
    with torch.no_grad():
        for batch in an_loader:
            _, pair = model(batch)
            aux = pair[4] if len(pair) > 4 else None
            if aux is not None:
                failure_preds.extend(torch.argmax(aux["failure_logits"], dim=-1).cpu().tolist())
                failure_targets.extend(aux["failure_target"].cpu().tolist())
                rem_preds.extend(torch.argmax(aux["remediation_logits"], dim=-1).cpu().tolist())
                rem_targets.extend(aux["remediation_target"].cpu().tolist())
                impact_mse.append(
                    F.mse_loss(aux["impact_pred"], aux["impact_target"], reduction="mean").item()
                )

    result = {
        "dataset": dataset_name,
        "explicit_multitask_labels": use_explicit,
        "ad_precision": core.get("ad_precision", 0.0),
        "ad_recall": core.get("ad_recall", 0.0),
        "ad_f1": core.get("ad_f1", 0.0),
        "ad_accuracy": core.get("ad_accuracy", 0.0),
        "tp": core.get("tp", 0),
        "tn": core.get("tn", 0),
        "fp": core.get("fp", 0),
        "fn": core.get("fn", 0),
        "rca_hr@1": rca.get("hr_at_1", 0.0),
        "rca_mrr": rca.get("mrr_at_20", 0.0),
        "failure_acc": accuracy_score(failure_targets, failure_preds) if len(failure_targets) else 0.0,
        "failure_macro_f1": f1_score(failure_targets, failure_preds, average="macro") if len(failure_targets) else 0.0,
        "impact_mse": _safe_mean(impact_mse),
        "remediation_acc": accuracy_score(rem_targets, rem_preds) if len(rem_targets) else 0.0,
        "inference_time_sec": core.get("inference_time_sec", 0.0),
    }
    return result


def run_cross_dataset_benchmark(model, datasets, output_path, batch_size=256, num_workers=4):
    """
    Evaluate a model on multiple datasets with a standardized protocol
    and save results as JSON.
    """
    all_results = []
    for ds in datasets:
        ds_root = f"data/{ds}"
        if not os.path.isdir(ds_root):
            continue
        if not os.path.exists(f"{ds_root}/n_test.txt") or not os.path.exists(f"{ds_root}/an_test.txt"):
            continue
        all_results.append(
            evaluate_unified_benchmark(
                model=model,
                dataset_name=ds,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf8") as f:
        json.dump({"results": all_results}, f, indent=2)
    return all_results
