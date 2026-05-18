# FaultGuard-AI Novelty Implementations

This document describes the implemented novelty modules in the current repository, where they live, how to run them, and the fresh BGL results produced by the latest test run.

## Overview

The repository currently implements three novelty variants on top of the baseline model, plus a unified evaluation path used for reporting:

1. **Novelty 1 — Domain-adaptive FaultGuard-AI**
2. **Novelty 2 — Multi-task FaultGuard-AI**
3. **Novelty 3 — Advanced interaction FaultGuard-AI**
4. **Unified benchmark and report generation**

The active evaluation/training entrypoint is `main.py`.

## Novelty 1: Domain-Adaptive FaultGuard-AI

### Goal
Improve robustness across domains by learning features that are less sensitive to domain-specific noise.

### Implemented idea
- Domain-aware training path using a domain-adaptation variant of FaultGuard-AI.
- Validation and evaluation are run through the same BGL split loaders as the baseline.

### Code locations
- `src/domain_adaptation.py`
- `main.py` modes: `novelty1_train`, `novelty1_eval`

### What it adds
- A domain-adaptive objective on top of the base anomaly-detection and RCA losses.
- A dedicated novelty checkpoint and evaluation path.

### Latest BGL result
- AD F1: **0.7556**
- AD Precision: **0.7234**
- AD Recall: **0.7909**
- RCA HR@1: **0.7001**
- RCA MRR@20: **0.7132**

## Novelty 2: Multi-task FaultGuard-AI

### Goal
Expand the original two-task setup with extra auxiliary supervision so the shared representation can capture more structure from the log stream.

### Implemented idea
- Auxiliary heads are added to FaultGuard-AI.
- The current code path supports weak-label style multitask training/evaluation.
- This novelty is the strongest current run by AD F1 and is the dynamic best checkpoint selected from the saved models.

### Code locations
- `src/multitask_expansion.py`
- `src/multitask_dataset.py`
- `main.py` modes: `novelty2_train`, `novelty2_eval`

### What it adds
- Additional auxiliary prediction branches beyond baseline AD + RCA.
- Dynamic checkpoint selection based on validation/test AD F1 rather than hardcoding a filename.

### Latest BGL result
- AD F1: **0.8541**
- AD Precision: **0.7640**
- AD Recall: **0.9683**
- AD Accuracy: **0.9716**
- RCA HR@1: **0.8209**
- RCA MRR@20: **0.8546**

### Why it matters
- This is the best-performing novelty in the current repository state.
- It is also the checkpoint chosen by `scripts/select_best_checkpoint.py`.

## Novelty 3: Advanced Interaction FaultGuard-AI

### Goal
Make task interaction more adaptive by introducing richer coupling between the model’s diagnostic branches.

### Implemented idea
- Dynamic interaction mechanisms are added to FaultGuard-AI.
- The novelty is designed to change how task signals influence each other during inference/training.

### Code locations
- `src/advanced_interaction.py`
- `main.py` modes: `novelty3_train`, `novelty3_eval`

### What it adds
- Stronger interaction modeling between AD and RCA-related pathways.
- A separate model checkpoint and evaluation path for the advanced interaction variant.

### Latest BGL result
- AD F1: **0.4390**
- AD Precision: **0.2909**
- AD Recall: **0.8937**
- AD Accuracy: **0.8037**
- RCA HR@1: **0.7652**
- RCA MRR@20: **0.7937**

## Unified evaluation and reporting

### Goal
Provide a repeatable way to evaluate saved checkpoints and regenerate comparison artifacts.

### Code locations
- `src/unified_benchmark.py`
- `scripts/select_best_checkpoint.py`
- `scripts/plot_paper_comparison.py`
- `scripts/generate_project_report.py`
- `main.py` mode: `unified_benchmark`

### Generated artifacts
- `report_output/selected_checkpoint_summary.json`
- `report_output/paper_comparison_metrics.json`
- `report_output/fig_bgl_paper_vs_novelties.png`
- `report_output/fig_bgl_novelty_auxiliary_heads.png`
- `report_output/PROJECT_REPORT.md`

## Current model ranking on BGL

From the latest fresh run:

| Variant | AD F1 | RCA HR@1 | RCA MRR@20 |
|---|---:|---:|---:|
| Novelty 1 | 0.7556 | 0.7001 | 0.7132 |
| **Novelty 2** | **0.8541** | **0.8209** | **0.8546** |
| Novelty 3 | 0.4390 | 0.7652 | 0.7937 |

Novelty 2 is the strongest implemented novelty in the current checkpoint set.

## How to run the novelties

### Train
```bash
python main.py --mode novelty1_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256
```

### Evaluate
```bash
python main.py --mode novelty1_eval --dataset BGL --load_checkpoint True
python main.py --mode novelty2_eval --dataset BGL --load_checkpoint True
python main.py --mode novelty3_eval --dataset BGL --load_checkpoint True
```

### Select the best checkpoint
```bash
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint
```

### Regenerate plots and report
```bash
python scripts/plot_paper_comparison.py
python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
```

## Related documentation

- [docs/COMMANDS.md](docs/COMMANDS.md) — full command reference
- [docs/RESULTS_AND_REPRODUCTION.md](docs/RESULTS_AND_REPRODUCTION.md) — reproduction guide
- [TEST_RESULTS.md](TEST_RESULTS.md) — fresh verification run
- [report_output/PAPER_COMPARISON_REPORT.md](report_output/PAPER_COMPARISON_REPORT.md) — paper comparison summary

## CLI quick reference

### Novelty 1

```bash
python main.py --mode novelty1_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty1_eval --dataset BGL --load_checkpoint True
```

### Novelty 2

```bash
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty2_eval --dataset BGL --load_checkpoint True
```

### Novelty 3

```bash
python main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty3_eval --dataset BGL --load_checkpoint True
```

### Unified benchmark

```bash
python main.py --mode unified_benchmark --dataset BGL
```
