#!/usr/bin/env python3
"""
Generate PROJECT_REPORT.md, metrics JSON, CSV tables, and Matplotlib figures
from Lightning logs and checkpoint evaluations.

Run from repo root OR chimera folder:
  python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import numpy as np

# chimera root = parent of scripts/
CHIMERA_ROOT = Path(__file__).resolve().parent.parent
os.chdir(CHIMERA_ROOT)
sys.path.insert(0, str(CHIMERA_ROOT))


def build_args(dataset: str, device: str, batch_size: int) -> Namespace:
    return Namespace(
        hard_device=device,
        gpu_index=0,
        load_checkpoint=False,
        model_save_path="checkpoint",
        epochs=10,
        batch_size=batch_size,
        warmup_epochs=8,
        lr=1e-3,
        accumulate_grad_batches=1,
        mode="eval",
        threshold=0.9,
        topk=1,
        dataset=dataset,
        source_datasets=None,
        target_dataset=None,
        num_domains=2,
        domain_lambda=0.1,
        domain_lambda_max=1.0,
        shots=10,
        adaptation_steps=100,
        adaptation_lr=1e-4,
        num_failure_types=10,
        num_remediation_actions=8,
        mt_logparse_lambda=0.2,
        mt_failure_lambda=0.3,
        mt_impact_lambda=0.2,
        mt_remediation_lambda=0.3,
        benchmark_datasets=None,
        benchmark_output="benchmark/results.json",
        device=__import__("torch").device("cuda:0") if device == "cuda" else __import__("torch").device("cpu"),
    )


def load_lightning_metrics_df():
    import pandas as pd
    import glob

    paths = sorted(glob.glob(str(CHIMERA_ROOT / "lightning_logs" / "version_*" / "metrics.csv")))
    frames = []
    for p in paths:
        ver = Path(p).parent.name.replace("version_", "v")
        df = pd.read_csv(p)
        df["run_version"] = ver
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def plot_training_curves(df, out_dir: Path):
    import matplotlib.pyplot as plt

    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        pass

    fig, ax = plt.subplots(figsize=(10, 5))
    epoch_loss = df.dropna(subset=["epoch_loss"])
    if len(epoch_loss) > 0:
        # best effort: group by run
        for run in epoch_loss["run_version"].unique():
            sub = epoch_loss[epoch_loss["run_version"] == run]
            ax.plot(sub["epoch"], sub["epoch_loss"], marker="o", ms=3, label=f"Val loss ({run})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss (epoch_loss)")
        ax.legend()
        ax.set_title("FaultGuard-AI training: validation loss per epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_validation_loss.png", dpi=150)
    plt.close(fig)

    if "adv_total_loss_step" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        step_loss = df["adv_total_loss_step"].dropna()
        if len(step_loss) > 1:
            ax2.plot(step_loss.index[:2000], step_loss.values[:2000], alpha=0.7)
            ax2.set_xlabel("Step index (truncated)")
            ax2.set_ylabel("Advanced interaction total loss (step)")
            ax2.set_title("Novelty 3: training loss (first 2000 logged steps)")
        fig2.tight_layout()
        fig2.savefig(out_dir / "fig_adv_step_loss.png", dpi=150)
        plt.close(fig2)

    task_cols = sorted([c for c in df.columns if c.startswith("task_w_") and c.endswith("_epoch")])
    if task_cols:
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        runs = sorted(df["run_version"].unique())
        width = 0.8 / max(len(runs), 1)
        xi = np.arange(len(task_cols))
        for j, run in enumerate(runs):
            sub = df[df["run_version"] == run].dropna(subset=["epoch"])
            if sub.empty:
                continue
            mx = int(sub["epoch"].max())
            row = sub[sub["epoch"] == mx]
            if row.empty:
                continue
            row = row.iloc[-1]
            vals = [float(row[c]) if pd.notna(row.get(c)) else 0.0 for c in task_cols]
            ax3.bar(xi + j * width, vals, width=width * 0.9, label=run, alpha=0.85)

        ax3.set_xticks(xi + width * (len(runs) - 1) / 2)
        ax3.set_xticklabels(
            [c.replace("task_w_", "").replace("_epoch", "") for c in task_cols],
            rotation=25,
            ha="right",
        )
        ax3.set_ylabel("Task weight (logged at epoch end)")
        ax3.legend()
        ax3.set_title("Novelty 3: dynamic task prioritization (final epoch)")
        fig3.tight_layout()
        fig3.savefig(out_dir / "fig_task_weights_final.png", dpi=150)
        plt.close(fig3)


def eval_checkpoint(name: str, model, n_loader, an_loader, aux=False):
    from src.metrics_eval import compute_ad_rca_metrics, compute_aux_multitask_metrics

    m = compute_ad_rca_metrics(model, n_loader, an_loader)
    m["checkpoint_label"] = name
    if aux:
        aux_m = compute_aux_multitask_metrics(model, an_loader)
        m["aux_weak_labels"] = aux_m
    return m


def run_all_evals(args_ns, ckpt_root: Path, dataset: str):
    import torch
    from src.dataset import get_loader, load_json
    from src.models import Chimera
    from src.multitask_expansion import MultiTaskChimera
    from src.advanced_interaction import AdvancedInteractionChimera

    data = lambda rel: CHIMERA_ROOT / "data" / dataset / rel
    n_loader = get_loader(
        str(data("n_test.txt")), str(data("n_test.txt")),
        batch_size=args_ns.batch_size, shuffle=False, num_workers=0,
    )
    an_loader = get_loader(
        str(data("an_test.txt")), str(data("an_test.txt")),
        batch_size=args_ns.batch_size, shuffle=False, num_workers=0,
    )

    # Use Chimera-balanced loaders for ALL models (same sampling as baseline).
    emb = load_json(str(data("emd_dict.json")))
    results = []

    ck_chimera = ckpt_root / "Chimera_model.bin"
    if ck_chimera.exists():
        m = Chimera(args_ns, emb).to(args_ns.hard_device)
        m.load_state_dict(torch.load(ck_chimera, map_location=args_ns.hard_device))
        results.append(eval_checkpoint("baseline_Chimera", m, n_loader, an_loader, aux=False))

    ck_mt = ckpt_root / "MultiTaskChimera_model.bin"
    if ck_mt.exists():
        m = MultiTaskChimera(args_ns, emb).to(args_ns.hard_device)
        m.load_state_dict(torch.load(ck_mt, map_location=args_ns.hard_device))
        results.append(eval_checkpoint("Novelty2_MultiTaskChimera", m, n_loader, an_loader, aux=True))

    ck_ai = ckpt_root / "AdvancedInteractionChimera_model.bin"
    if ck_ai.exists():
        m = AdvancedInteractionChimera(args_ns, emb).to(args_ns.hard_device)
        m.load_state_dict(torch.load(ck_ai, map_location=args_ns.hard_device))
        results.append(
            eval_checkpoint("Novelty3_AdvancedInteractionChimera", m, n_loader, an_loader, aux=True)
        )

    return results


def plot_model_comparison_bar(rows: list, out_dir: Path):
    import matplotlib.pyplot as plt

    if not rows:
        return
    import pandas as pd

    flat = []
    for r in rows:
        row = {
            "model": r["checkpoint_label"],
            "ad_f1": r.get("ad_f1", 0),
            "hr_at_1": r.get("rca", {}).get("hr_at_1", 0),
            "mrr_at_20": r.get("rca", {}).get("mrr_at_20", 0),
        }
        if "aux_weak_labels" in r:
            row["failure_acc"] = r["aux_weak_labels"].get("failure_acc", 0)
            row["impact_rmse"] = r["aux_weak_labels"].get("impact_rmse", 0)
        flat.append(row)
    dfp = pd.DataFrame(flat)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    x = np.arange(len(dfp))
    w = 0.25
    axes[0].bar(x, dfp["ad_f1"], color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(dfp["model"], rotation=15, ha="right")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Anomaly detection F1")

    axes[1].bar(x, dfp["hr_at_1"], color="darkorange")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(dfp["model"], rotation=15, ha="right")
    axes[1].set_ylabel("HR@1")
    axes[1].set_title("Root-cause HR@1")

    axes[2].bar(x, dfp["mrr_at_20"], color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(dfp["model"], rotation=15, ha="right")
    axes[2].set_ylabel("MRR@20")
    axes[2].set_title("Root-cause MRR (top-20)")

    fig.suptitle("FaultGuard-AI — test-set comparison (same eval as original Chimera)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_model_comparison_ad_rca.png", dpi=150)
    plt.close(fig)

    if "failure_acc" in dfp.columns:
        fig2, ax = plt.subplots(figsize=(7, 4))
        ax.bar(dfp["model"], dfp["failure_acc"], color="mediumpurple")
        ax.set_ylabel("Accuracy (weak auxiliary labels)")
        ax.set_title("Extra tasks — failure-class head (weak labels if no *_mt.txt)")
        plt.xticks(rotation=15, ha="right")
        fig2.tight_layout()
        fig2.savefig(out_dir / "fig_aux_failure_acc.png", dpi=150)
        plt.close(fig2)


def _friendly_model_heading(checkpoint_label: str) -> str:
    return {
        "baseline_Chimera": "Baseline (original Chimera — anomaly + root-cause only)",
        "Novelty2_MultiTaskChimera": "Novelty 2 (multi-task: failure/impact/remediation/etc.)",
        "Novelty3_AdvancedInteractionChimera": "Novelty 3 (graph + dynamic loss weights)",
    }.get(checkpoint_label, checkpoint_label)


def _row_interpretation(r: dict) -> str:
    """One short sentence from confusion counts (user-facing)."""
    tp = int(r.get("tp", 0))
    tn = int(r.get("tn", 0))
    fp = int(r.get("fp", 0))
    fn = int(r.get("fn", 0))
    parts = []
    if tp == 0 and fn > 0:
        parts.append("Treats anomaly test windows mostly as normal, so alerting is suppressed.")
    if tn == 0 and fp > 0:
        parts.append("Treats essentially all normal windows as anomalies (very high alert noise).")
    if tn > 0 and fp > 0 and tp > 0:
        parts.append("Mix of hits and misses on both splits—see ratios in the JSON file.")
    if not parts:
        parts.append(f"Counts: anomalies detected {tp} / missed {fn}; normal rejected {tn} / false-alert {fp}.")
    return " ".join(parts)


def write_report_md(
    path: Path,
    eval_rows: list,
    df_lightning,
    generated_at: str,
    dataset: str = "BGL",
):
    lines = [
        "# FaultGuard-AI — results summary (auto-generated)",
        "",
        "This note is meant for humans: metrics, plots, and what the project accomplished.",
        "",
        f"*Generated:* **{generated_at}** • *Benchmark dataset:* **{dataset}** (held-out `n_test` + `an_test`)",
        "",
        "## TL;DR (read this first)",
        "",
        "**FaultGuard-AI** extends the Chimera-style log anomaly + root-cause stack with optional cross-domain training, multi-task heads, richer task interactions, and a single JSON benchmark format.",
        "",
        "- **Infrastructure achieved:** reproducible trains/evals (`main.py`), one command to regenerate this report (`scripts/generate_project_report.py`), and metrics that match `main.run_eval` / `metrics_eval`.",
        "- **Scientific caveat:** richer models need careful training and checkpoints; stronger AD/RCA than the slim baseline is a *hypothesis*, not guaranteed out of the box.",
        "",
        "## What has been achieved (deliverables)",
        "",
        "| Area | Concrete outcome |",
        "|------|------------------|",
        "| **Detection & diagnosis** | Same evaluation protocol as the original code (balanced loaders, anomaly F1 + root-cause rank metrics). |",
        "| **Research extensions** | Four modes wired into `main.py` (cross-domain, multi-task, advanced interaction, unified benchmark). See `NOVELTIES.md` for formulas. |",
        "| **Reporting** | Figures + CSV + JSON under `report_output/`; this Markdown file summarizes them in plain language. |",
        "| **Reproducibility** | Paths are relative to `chimera/data/<dataset>/…`; rerun with `--dataset` when another corpus is prepared (e.g. Thunderbird splits). |",
        "",
        "## 1. What we implemented",
        "",
        "| Novelty | Description | Key files |",
        "|--------|-------------|-----------|",
        "| **1 – Cross-domain** | Domain-adversarial training (DANN-style), few-shot head adaptation | `src/domain_adaptation.py`, `src/cross_domain_dataset.py`, `main.py` modes `cd_train`, `cd_eval`, `few_shot` |",
        "| **2 – Multi-task** | Four extra heads: parsing, failure class, impact, remediation | `src/multitask_expansion.py`, `src/multitask_dataset.py`, `main.py` `mt_train` / `mt_eval` |",
        "| **3 – Advanced interaction** | Graph task interaction, hierarchical mixing, dynamic loss weights | `src/advanced_interaction.py`, `main.py` `ai_train` / `ai_eval` |",
        "| **4 – Unified benchmark** | Standard metrics JSON across datasets | `src/unified_benchmark.py`, `main.py` `unified_benchmark` |",
        "",
        "Full formulas and CLI: see `NOVELTIES.md`.",
        "",
        "## 2. How novelties are achieved (summary)",
        "",
        "- **N1:** Shared encoder features are pushed to be domain-invariant via a gradient-reversal domain discriminator.",
        "- **N2:** Shared sequence representations feed auxiliary heads with multi-task losses (explicit `*_mt.txt` labels when present, else weak labels from AD/RCA).",
        "- **N3:** Task embeddings pass through a learned graph, hierarchical grouping, then a prioritizer yields softmax weights multiplied into per-task losses.",
        "- **N4:** Same AD/RCA protocol plus auxiliary metrics; results written to JSON for cross-dataset tables.",
        "",
        "## 3. Training telemetry (Lightning CSV)",
        "",
        "Figures:",
        "",
        "- `report_output/fig_validation_loss.png` – validation loss by epoch.",
        "- `report_output/fig_adv_step_loss.png` – novelty-3 step loss sample (if logged).",
        "- `report_output/fig_task_weights_final.png` – dynamic task weights at last epoch (if logged).",
        "",
        "## 4. Test results — anomaly detection & root cause",
        "",
        "### Metric cheat sheet",
        "",
        "| Symbol | Meaning in one line | Higher / lower |",
        "|--------|---------------------|----------------|",
        "| **Anomaly F1** | Blend of precision and recall for “is this window faulty?” vs normal | Higher is usually better |",
        "| **HR@1** | Fraction of anomaly test windows where the **top** ranked component is truly failing | Higher is better |",
        "| **MRR@20** | Average inverse rank when the truth appears in the **top 20** ranked components | Higher is better |",
        "| **Infer. time** | Seconds to score the test loaders on your machine | Lower is faster (hardware-dependent) |",
        "",
        "Raw confusion counts (**tp / fp / tn / fn**) live in `report_output/metrics_summary.json` — use them whenever F1 alone looks confusing.",
        "",
    ]

    baseline_f1 = None
    extended_f1s = []
    for r in eval_rows:
        lbl = r.get("checkpoint_label", "")
        f1 = r.get("ad_f1")
        if f1 is None:
            continue
        if lbl == "baseline_Chimera":
            baseline_f1 = f1
        else:
            extended_f1s.append((lbl, f1))
    best_ext = max((x[1] for x in extended_f1s), default=None)

    if eval_rows:
        lines.append("Machine-readable copy: **`report_output/metrics_summary.json`** (and `metrics_summary_flat.csv`).")
        lines.append("")
        lines.append("### Snapshot table")
        lines.append("")
        lines.append("| What you ran | Anomaly F1 | Root HR@1 | MRR top-20 | Inference (s) |")
        lines.append("|--------------|-----------|-----------|-----------|---------------|")
        for r in eval_rows:
            rca = r.get("rca", {})
            nice = _friendly_model_heading(r.get("checkpoint_label", ""))
            lines.append(
                f"| {nice} "
                f"| {r.get('ad_f1',0):.4f} "
                f"| {rca.get('hr_at_1',0):.4f} "
                f"| {rca.get('mrr_at_20',0):.4f} "
                f"| {r.get('inference_time_sec',0):.2f} |"
            )
        lines.append("")
        lines.append("### Plain-language readout for this run")
        lines.append("")
        for r in eval_rows:
            nice = _friendly_model_heading(r.get("checkpoint_label", ""))
            lines.append(f"- **{nice}** — {_row_interpretation(r)}")
        lines.append("")
        lines.append("### Compared to the original two-task model")
        lines.append("")
        if baseline_f1 is not None and best_ext is not None:
            delta = best_ext - baseline_f1
            pct = (100.0 * delta / baseline_f1) if baseline_f1 else 0.0
            lines.append(
                f"The saved **baseline** checkpoint reached anomaly F1 **{baseline_f1:.4f}**. "
                f"The strongest extension in this folder is **{best_ext:.4f}** (about **{delta:+.4f}** absolute, ~**{pct:+.1f}%** relative)."
            )
        else:
            lines.append(
                "No side-by-side baseline was generated because **`checkpoint/Chimera_model.bin`** was not found."
            )
            lines.append("")
            lines.append(
                'To unlock a fair "original vs FaultGuard extensions" sentence: '
                "`python main.py --mode train --dataset " + dataset + "` "
                "then rerun `python scripts/generate_project_report.py --dataset " + dataset + "`."
            )
        lines.append("")
        lines.append("### Extra task heads (Novelties 2 and 3 only)")
        lines.append("")
        lines.append(
            "If `aux_weak_labels` appears in the JSON, those scores use **weak labels** when `*_mt.txt` sidecars are missing—handy to see whether "
            "auxiliary heads move at all, not a substitute for real parse/failure/impact/remediation annotations."
        )
        lines.append("")
        lines.append("### Technical notes (for debugging odd F1 values)")
        lines.append("")
        lines.append(
            "- **F1 = 0** usually means the model never raises an anomaly flag on the anomaly split, or never accepts the normal split—always open `tp/fp/tn/fn`."
        )
        lines.append(
            "- **High HR@1 with low F1** can happen if the model fires on anomalies but floods normal traffic with alerts; the table above does not replace operations dashboards."
        )
        lines.append(
            "- **Original Chimera** is the slim two-head network; extended models add capacity and losses, so retrain/tune rather than expecting a free win."
        )
    else:
        lines.append("_No checkpoints evaluated (missing models or loaders)._")

    lines.extend(
        [
            "",
            "## 5. Figures (visual summary)",
            "",
            "- `report_output/fig_model_comparison_ad_rca.png` — bar chart of the same F1 / HR@1 / MRR values as the snapshot table.",
            "- `report_output/fig_aux_failure_acc.png` — optional: weak-label accuracy for the failure head (only when multi-task style models were evaluated).",
            "- `report_output/fig_validation_loss.png` — did training loss trend down (from Lightning logs)?",
            "",
            "## 6. How to regenerate",
            "",
            "```bash",
            "cd chimera",
            "python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128",
            "```",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BGL")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available; using CPU.")
            args.device = "cpu"

    out_dir = CHIMERA_ROOT / "report_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    args_ns = build_args(args.dataset, args.device, args.batch_size)

    df_lightning = load_lightning_metrics_df()
    if df_lightning is not None:
        df_lightning.to_csv(out_dir / "lightning_metrics_concat.csv", index=False)
        plot_training_curves(df_lightning, out_dir)

    eval_rows = run_all_evals(args_ns, CHIMERA_ROOT / "checkpoint", args.dataset)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf8") as f:
        json.dump({"evaluations": eval_rows, "dataset": args.dataset}, f, indent=2)

    if eval_rows:
        pd.json_normalize(eval_rows, sep="_").to_csv(out_dir / "metrics_summary_flat.csv", index=False)
        plot_model_comparison_bar(eval_rows, out_dir)

    report_ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    report_primary = out_dir / "PROJECT_REPORT.md"
    write_report_md(report_primary, eval_rows, df_lightning, report_ts, dataset=args.dataset)
    shutil.copy2(report_primary, CHIMERA_ROOT / "PROJECT_REPORT.md")

    print("Wrote:")
    print(f"  {report_primary}")
    print(f"  {CHIMERA_ROOT / 'PROJECT_REPORT.md'} (copy)")
    print(f"  {out_dir / 'metrics_summary.json'}")
    if df_lightning is not None:
        print(f"  {out_dir / 'lightning_metrics_concat.csv'}")


if __name__ == "__main__":
    main()
