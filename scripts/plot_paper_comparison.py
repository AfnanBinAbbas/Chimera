#!/usr/bin/env python3
"""Plot BGL-only comparison against the reference paper.

The repository already stores the BGL paper-vs-ours comparison in
`report_output/paper_comparison_metrics.json`. This script turns those saved
metrics into two BGL-only figures:

- paper vs. baseline + novelty variants for AD/RCA metrics
- auxiliary feature-head metrics for novelties 2 and 3
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CHIMERA_ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = CHIMERA_ROOT / "report_output"
OUT_MAIN = REPORT_DIR / "fig_bgl_paper_vs_novelties.png"
OUT_AUX = REPORT_DIR / "fig_bgl_novelty_auxiliary_heads.png"


def _load_metrics() -> dict:
    with open(REPORT_DIR / "paper_comparison_metrics.json", "r", encoding="utf-8") as handle:
        return json.load(handle)


def _build_series(payload: dict) -> dict:
    paper = payload["paper"]
    ours = payload["ours"]
    return {
        "paper": {
            "ad_f1": paper["ad_bgl"]["f1"],
            "ad_precision": paper["ad_bgl"]["precision"],
            "ad_recall": paper["ad_bgl"]["recall"],
            "rca_hr_at_1": paper["rca_bgl"]["hr_at_1"],
            "rca_mrr": paper["rca_bgl"]["mrr"],
        },
        "baseline": {
            "ad_f1": ours["baseline_chimera_eval"]["ad_f1"],
            "ad_precision": ours["baseline_chimera_eval"]["ad_precision"],
            "ad_recall": ours["baseline_chimera_eval"]["ad_recall"],
            "rca_hr_at_1": ours["baseline_chimera_eval"]["rca_hr_at_1"],
            "rca_mrr": ours["baseline_chimera_eval"]["rca_mrr_at_20"],
        },
        "novelty1": {
            "ad_f1": ours["novelty1_eval"]["ad_f1"],
            "ad_precision": ours["novelty1_eval"]["ad_precision"],
            "ad_recall": ours["novelty1_eval"]["ad_recall"],
            "rca_hr_at_1": ours["novelty1_eval"]["rca_hr_at_1"],
            "rca_mrr": ours["novelty1_eval"]["rca_mrr_at_20"],
        },
        "novelty2": {
            "ad_f1": ours["novelty2_eval"]["ad_f1"],
            "ad_precision": ours["novelty2_eval"]["ad_precision"],
            "ad_recall": ours["novelty2_eval"]["ad_recall"],
            "rca_hr_at_1": ours["novelty2_eval"]["rca_hr_at_1"],
            "rca_mrr": ours["novelty2_eval"]["rca_mrr_at_20"],
        },
        "novelty3": {
            "ad_f1": ours["novelty3_eval"]["ad_f1"],
            "ad_precision": ours["novelty3_eval"]["ad_precision"],
            "ad_recall": ours["novelty3_eval"]["ad_recall"],
            "rca_hr_at_1": ours["novelty3_eval"]["rca_hr_at_1"],
            "rca_mrr": ours["novelty3_eval"]["rca_mrr_at_20"],
        },
    }


def _plot_metric(ax, labels: list[str], values_by_series: dict[str, list[float]], title: str, ylabel: str) -> None:
    series_names = list(values_by_series.keys())

    x = np.arange(len(labels))
    width = 0.8 / max(len(series_names), 1)

    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values_by_series[series_name], width=width * 0.95, label=series_name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.25)


def _plot_aux_metric(ax, labels: list[str], values_by_series: dict[str, list[float]], title: str, ylabel: str, lower_is_better=False) -> None:
    series_names = list(values_by_series.keys())
    x = np.arange(len(labels))
    width = 0.8 / max(len(series_names), 1)

    for idx, series_name in enumerate(series_names):
        offset = (idx - (len(series_names) - 1) / 2) * width
        bars = ax.bar(x + offset, values_by_series[series_name], width=width * 0.95, label=series_name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if lower_is_better:
        ax.invert_yaxis()
    ax.grid(axis="y", alpha=0.25)


def main() -> None:
    payload = _load_metrics()
    series = _build_series(payload)

    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        import matplotlib.pyplot as plt

    labels = ["Paper", "Baseline", "Novelty 1", "Novelty 2", "Novelty 3"]
    ad_series = {
        "AD F1": [series["paper"]["ad_f1"], series["baseline"]["ad_f1"], series["novelty1"]["ad_f1"], series["novelty2"]["ad_f1"], series["novelty3"]["ad_f1"]],
        "AD Precision": [series["paper"]["ad_precision"], series["baseline"]["ad_precision"], series["novelty1"]["ad_precision"], series["novelty2"]["ad_precision"], series["novelty3"]["ad_precision"]],
        "AD Recall": [series["paper"]["ad_recall"], series["baseline"]["ad_recall"], series["novelty1"]["ad_recall"], series["novelty2"]["ad_recall"], series["novelty3"]["ad_recall"]],
    }
    rca_series = {
        "Paper": [series["paper"]["rca_hr_at_1"], series["paper"]["rca_mrr"]],
        "Baseline": [series["baseline"]["rca_hr_at_1"], series["baseline"]["rca_mrr"]],
        "Novelty 1": [series["novelty1"]["rca_hr_at_1"], series["novelty1"]["rca_mrr"]],
        "Novelty 2": [series["novelty2"]["rca_hr_at_1"], series["novelty2"]["rca_mrr"]],
        "Novelty 3": [series["novelty3"]["rca_hr_at_1"], series["novelty3"]["rca_mrr"]],
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    _plot_metric(
        axes[0],
        labels,
        {
            "AD F1": ad_series["AD F1"],
            "AD Precision": ad_series["AD Precision"],
            "AD Recall": ad_series["AD Recall"],
        },
        "BGL anomaly detection vs paper",
        "Score",
    )
    _plot_metric(
        axes[1],
        ["Paper", "Baseline", "Novelty 1", "Novelty 2", "Novelty 3"],
        {
            "HR@1": rca_series["Paper"][:1] + [rca_series[k][0] for k in ["Baseline", "Novelty 1", "Novelty 2", "Novelty 3"]],
            "MRR@20": [series["paper"]["rca_mrr"], series["baseline"]["rca_mrr"], series["novelty1"]["rca_mrr"], series["novelty2"]["rca_mrr"], series["novelty3"]["rca_mrr"]],
        },
        "BGL root-cause localization vs paper",
        "Score",
    )
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("BGL paper comparison across baseline and novelty variants")
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_MAIN, dpi=160)
    plt.close(fig)

    # Dynamic auxiliary metrics
    ours = payload.get("ours", {})
    aux_labels = ["Failure acc", "Remediation acc", "Impact RMSE"]
    aux_series = {}
    for nov in ["novelty2", "novelty3"]:
        m = ours.get(f"{nov}_eval", {}).get("aux_weak_labels")
        if m:
            aux_series[nov.capitalize().replace("y", "y ")] = [
                m.get("failure_acc", 0),
                m.get("remediation_acc", 0),
                m.get("impact_rmse", 0)
            ]

    if aux_series:
        fig2, ax2 = plt.subplots(figsize=(10, 4.8))
        _plot_aux_metric(ax2, aux_labels, aux_series, "Extra auxiliary heads added beyond the paper", "Metric")
        ax2.legend(frameon=False, loc="upper right")
        ax2.set_title("Auxiliary feature-head performance (novelty 2 and novelty 3)")
        fig2.tight_layout()
        fig2.savefig(OUT_AUX, dpi=160)
        plt.close(fig2)
        print(f"Wrote {OUT_AUX}")
    else:
        print("Skipping auxiliary plot (no aux metrics found).")

    print(f"Wrote {OUT_MAIN}")
    print(f"Wrote {OUT_AUX}")


if __name__ == "__main__":
    main()