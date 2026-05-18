# FaultGuard-AI — results summary (auto-generated)

This note is meant for humans: metrics, plots, and what the project accomplished.

*Generated:* **2026-05-09 20:30 UTC** • *Benchmark dataset:* **BGL** (held-out `n_test` + `an_test`)

## TL;DR (read this first)

**FaultGuard-AI** extends the Chimera-style log anomaly + root-cause stack with optional cross-domain training, multi-task heads, richer task interactions, and a single JSON benchmark format.

- **Infrastructure achieved:** reproducible trains/evals (`main.py`), one command to regenerate this report (`scripts/generate_project_report.py`), and metrics that match `main.run_eval` / `metrics_eval`.
- **Scientific caveat:** richer models need careful training and checkpoints; stronger AD/RCA than the slim baseline is a *hypothesis*, not guaranteed out of the box.

## What has been achieved (deliverables)

| Area | Concrete outcome |
|------|------------------|
| **Detection & diagnosis** | Same evaluation protocol as the original code (balanced loaders, anomaly F1 + root-cause rank metrics). |
| **Research extensions** | Four modes wired into `main.py` (cross-domain, multi-task, advanced interaction, unified benchmark). See `NOVELTIES.md` for formulas. |
| **Reporting** | Figures + CSV + JSON under `report_output/`; this Markdown file summarizes them in plain language. |
| **Reproducibility** | Paths are relative to `chimera/data/<dataset>/…`; rerun with `--dataset` when another corpus is prepared (e.g. Thunderbird splits). |

## 1. What we implemented

| Novelty | Description | Key files |
|--------|-------------|-----------|
| **1 – Cross-domain** | Domain-adversarial training (DANN-style), few-shot head adaptation | `src/domain_adaptation.py`, `src/cross_domain_dataset.py`, `main.py` modes `cd_train`, `cd_eval`, `few_shot` |
| **2 – Multi-task** | Four extra heads: parsing, failure class, impact, remediation | `src/multitask_expansion.py`, `src/multitask_dataset.py`, `main.py` `mt_train` / `mt_eval` |
| **3 – Advanced interaction** | Graph task interaction, hierarchical mixing, dynamic loss weights | `src/advanced_interaction.py`, `main.py` `ai_train` / `ai_eval` |
| **4 – Unified benchmark** | Standard metrics JSON across datasets | `src/unified_benchmark.py`, `main.py` `unified_benchmark` |

Full formulas and CLI: see `NOVELTIES.md`.

## 2. How novelties are achieved (summary)

- **N1:** Shared encoder features are pushed to be domain-invariant via a gradient-reversal domain discriminator.
- **N2:** Shared sequence representations feed auxiliary heads with multi-task losses (explicit `*_mt.txt` labels when present, else weak labels from AD/RCA).
- **N3:** Task embeddings pass through a learned graph, hierarchical grouping, then a prioritizer yields softmax weights multiplied into per-task losses.
- **N4:** Same AD/RCA protocol plus auxiliary metrics; results written to JSON for cross-dataset tables.

## 3. Training telemetry (Lightning CSV)

Figures:

- `report_output/fig_validation_loss.png` – validation loss by epoch.
- `report_output/fig_adv_step_loss.png` – novelty-3 step loss sample (if logged).
- `report_output/fig_task_weights_final.png` – dynamic task weights at last epoch (if logged).
- `report_output/fig_bgl_paper_vs_novelties.png` – BGL paper vs. baseline + novelty variants (AD/RCA).
- `report_output/fig_bgl_novelty_auxiliary_heads.png` – auxiliary feature-head metrics for novelties 2 and 3.

## 4. Test results — anomaly detection & root cause

### Metric cheat sheet

| Symbol | Meaning in one line | Higher / lower |
|--------|---------------------|----------------|
| **Anomaly F1** | Blend of precision and recall for “is this window faulty?” vs normal | Higher is usually better |
| **HR@1** | Fraction of anomaly test windows where the **top** ranked component is truly failing | Higher is better |
| **MRR@20** | Average inverse rank when the truth appears in the **top 20** ranked components | Higher is better |
| **Infer. time** | Seconds to score the test loaders on your machine | Lower is faster (hardware-dependent) |

Raw confusion counts (**tp / fp / tn / fn**) live in `report_output/metrics_summary.json` — use them whenever F1 alone looks confusing.

Machine-readable copy: **`report_output/metrics_summary.json`** (and `metrics_summary_flat.csv`).

### Snapshot table

| What you ran | Anomaly F1 | Root HR@1 | MRR top-20 | Inference (s) |
|--------------|-----------|-----------|-----------|---------------|
| Baseline (original Chimera — anomaly + root-cause only) | 0.9615 | 0.8757 | 0.9169 | 5.52 |
| Novelty 2 (multi-task: failure/impact/remediation/etc.) | 0.9695 | 0.9494 | 0.9594 | 5.43 |
| Novelty 3 (graph + dynamic loss weights) | 0.9660 | 0.8757 | 0.9089 | 5.36 |

### Plain-language readout for this run

- **Baseline (original Chimera — anomaly + root-cause only)** — Mix of hits and misses on both splits—see ratios in the JSON file.
- **Novelty 2 (multi-task: failure/impact/remediation/etc.)** — Mix of hits and misses on both splits—see ratios in the JSON file.
- **Novelty 3 (graph + dynamic loss weights)** — Mix of hits and misses on both splits—see ratios in the JSON file.

### Compared to the original two-task model

The saved **baseline** checkpoint reached anomaly F1 **0.9615**. The strongest extension in this folder is **0.9695** (about **+0.0079** absolute, ~**+0.8%** relative).

### Mathematical Foundation

The Chimera framework and its extensions rely on a multi-objective loss function:

$$ \mathcal{L}_{total} = \mathcal{L}_{ad} + \lambda_2 \mathcal{L}_{localizer} + \lambda_3 \mathcal{L}_{diff} + \lambda_4 \mathcal{L}_{jsd} + \sum \mathcal{L}_{aux} $$

#### 1. Anomaly Detection (AD) Loss
Uses Cross-Entropy on the pooled representations $z$ from the shared and task-specific encoders.

#### 2. Root Cause Localization (Ranking Loss)
Uses a hinge-based ranking loss to ensure that the anomalous window score ($s_{an}$) is higher than the normal window score ($s_n$):
$$ \mathcal{L}_{localizer} = \max(0, 1 - s_{an} + s_n) $$

#### 3. Difference (Orthogonality) Loss
To ensure the shared and private encoders learn non-redundant features, we minimize the cosine similarity between their feature matrices:
$$ \mathcal{L}_{diff} = ||H_{shared}^\top H_{private}||^2_F $$

#### 4. JS-Divergence Alignment
Aligns the attention weights ($A$) with the predicted root-cause probabilities ($S$):
$$ \mathcal{L}_{jsd} = JSD(A || S) $$

### Architectural Innovations

#### Novelty 1: Domain Adaptation
Introduces a **Gradient Reversal Layer (GRL)** and a domain discriminator to ensure features are robust across different system distributions. Weight is scheduled via $\lambda_{domain} \cdot \min(1, epoch/warmup)$.

#### Novelty 2: Multi-Task & GAN
Uses auxiliary heads (Failure Class, Impact Score) and a Generative Adversarial Network to augment the training set with synthetic anomalies, solving the data-scarcity problem in log analysis.

#### Novelty 3: Advanced Interaction
Uses a Graph-based interaction layer to allow the detection head and localization head to exchange state information before final inference.

### Technical notes (for debugging)

- **F1 = 0** usually means the model never raises an anomaly flag on the anomaly split, or never accepts the normal split—always open `tp/fp/tn/fn`.
- **High HR@1 with low F1** can happen if the model fires on anomalies but floods normal traffic with alerts; the table above does not replace operations dashboards.
- **Original Chimera** is the slim two-head network; extended models add capacity and losses, so retrain/tune rather than expecting a free win.

## 5. Figures (visual summary)

- `report_output/fig_bgl_paper_vs_novelties.png` — BGL paper-vs-baseline-vs-novelty comparison for AD F1, precision, recall, HR@1, and MRR@20.
- `report_output/fig_bgl_novelty_auxiliary_heads.png` — auxiliary heads introduced by novelties 2 and 3 (failure, remediation, impact).
- `report_output/fig_model_comparison_ad_rca.png` — bar chart of the same F1 / HR@1 / MRR values as the snapshot table.
- `report_output/fig_aux_failure_acc.png` — optional: weak-label accuracy for the failure head (only when multi-task style models were evaluated).
- `report_output/fig_validation_loss.png` — did training loss trend down (from Lightning logs)?

## 6. How to regenerate

```bash
cd chimera
python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
```