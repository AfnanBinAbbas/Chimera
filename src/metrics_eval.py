"""
Programmatic AD + RCA metrics (same logic as main.run_eval prints).
Used by scripts/generate_project_report.py for reproducible JSON + plots.
"""

import math
import time
import torch
from sklearn.metrics import ndcg_score


def compute_ad_rca_metrics(model, n_eval_loader, an_eval_loader):
    model.eval()
    metrics = {}
    t0 = time.time()

    prev_log = getattr(model, "log", None)
    setattr(model, "log", lambda *args, **kwargs: None)

    try:
        with torch.no_grad():
            tp = tn = fp = fn = 0

            for batch in n_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair[:4]
                for i in range(len(out)):
                    if out[i][1].item() >= out[i][0].item():
                        fp += 1
                    else:
                        tn += 1

            for batch in an_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair[:4]
                for i in range(len(out)):
                    if out[i][1].item() >= out[i][0].item():
                        tp += 1
                    else:
                        fn += 1

            metrics["tp"] = int(tp)
            metrics["tn"] = int(tn)
            metrics["fp"] = int(fp)
            metrics["fn"] = int(fn)

            if tp > 0:
                p = tp / (tp + fp)
                r = tp / (tp + fn)
                f1 = 2 * p * r / (p + r)
                acc = (tp + tn) / (tp + tn + fp + fn)
            else:
                p = r = f1 = acc = 0.0

            metrics["ad_precision"] = float(p)
            metrics["ad_recall"] = float(r)
            metrics["ad_f1"] = float(f1)
            metrics["ad_accuracy"] = float(acc)

            rca = {}
            maps = 0
            for topk in range(1, 6):
                pos = nums = ndcg = counts = pr_sum = faults = 0

                for batch in an_eval_loader:
                    _, pair = model(batch)
                    out, score, ad_label, rca_label = pair[:4]
                    for i in range(len(out)):
                        if torch.sum(rca_label[i], dim=-1).item() > 0:
                            faults += 1
                        if out[i][1].item() >= out[i][0].item():
                            candidates = torch.topk(score[i], topk, dim=-1).indices
                            target = rca_label[i].unsqueeze(0)
                            res = target[
                                torch.arange(target.size(0)).unsqueeze(1),
                                candidates.unsqueeze(0),
                            ]
                            if torch.sum(res[0], dim=-1).item() > 0:
                                pos += 1
                                pr_sum += torch.sum(res[0], dim=-1).item() / min(
                                    topk, torch.sum(target[0], dim=-1).item()
                                )
                        nums += 1
                    ndcg += ndcg_score(rca_label.cpu(), score.cpu(), k=topk)
                    counts += 1

                pr_at_k = pr_sum / faults if faults else 0.0
                maps += pr_at_k
                rca[f"hr_at_{topk}"] = float(pos / nums) if nums else 0.0
                rca[f"ndcg_at_{topk}"] = float(ndcg / counts) if counts else 0.0
                rca[f"pr_at_{topk}"] = float(pr_at_k)
                rca[f"map_at_{topk}"] = float(maps / topk)

            mrr_total = faults_mrr = 0
            for batch in an_eval_loader:
                _, pair = model(batch)
                out, score, ad_label, rca_label = pair[:4]
                for i in range(len(out)):
                    if torch.sum(rca_label[i], dim=-1).item() > 0:
                        faults_mrr += 1
                    if out[i][1].item() >= out[i][0].item():
                        candidates = torch.topk(score[i], 20, dim=-1).indices
                        target = rca_label[i].unsqueeze(0)
                        res = target[
                            torch.arange(target.size(0)).unsqueeze(1),
                            candidates.unsqueeze(0),
                        ]
                        if torch.sum(res[0], dim=-1).item() > 0:
                            mrr_total += 1 / (
                                (res[0] > 0).nonzero(as_tuple=True)[0][0].item() + 1
                            )

            rca["mrr_at_20"] = float(mrr_total / faults_mrr) if faults_mrr else 0.0
            metrics["rca"] = rca
            metrics["inference_time_sec"] = float(time.time() - t0)
    finally:
        if prev_log is not None:
            setattr(model, "log", prev_log)

    return metrics


def compute_aux_multitask_metrics(model, an_eval_loader):
    import torch.nn.functional as F

    model.eval()
    prev_log = getattr(model, "log", None)
    setattr(model, "log", lambda *args, **kwargs: None)
    out = {
        "failure_acc": 0.0,
        "remediation_acc": 0.0,
        "impact_rmse": 0.0,
    }
    failure_correct = failure_total = 0
    remediation_correct = remediation_total = 0
    impact_sse = impact_n = 0

    try:
        with torch.no_grad():
            for batch in an_eval_loader:
                _, pair = model(batch)
                if len(pair) < 5:
                    continue
                aux = pair[4]

                fp = torch.argmax(aux["failure_logits"], dim=-1)
                ft = aux["failure_target"]
                failure_correct += (fp == ft).sum().item()
                failure_total += ft.numel()

                rp = torch.argmax(aux["remediation_logits"], dim=-1)
                rt = aux["remediation_target"]
                remediation_correct += (rp == rt).sum().item()
                remediation_total += rt.numel()

                ip = aux["impact_pred"]
                it = aux["impact_target"]
                impact_sse += F.mse_loss(ip, it, reduction="sum").item()
                impact_n += it.numel()
    finally:
        if prev_log is not None:
            setattr(model, "log", prev_log)

    if failure_total:
        out["failure_acc"] = failure_correct / failure_total
    if remediation_total:
        out["remediation_acc"] = remediation_correct / remediation_total
    if impact_n:
        out["impact_rmse"] = math.sqrt(impact_sse / impact_n)
    return out
