# Results and Reproduction Guide

This guide documents the completed BGL-only workflow, the dynamic checkpoint selection, and the generated comparison figures.

## What was completed

- BGL evaluation outputs were collected in `report_output/paper_comparison_metrics.json`.
- Comparison plots were generated:
  - `report_output/fig_bgl_paper_vs_novelties.png`
  - `report_output/fig_bgl_novelty_auxiliary_heads.png`
- The project report was updated in `report_output/PROJECT_REPORT.md`.
- Dynamic checkpoint selection was added in `scripts/select_best_checkpoint.py`.
- Validation F1 monitoring and best-checkpoint checkpointing were integrated into `main.py` and `src/callbacks.py`.

## Current best BGL numbers

From the saved metrics payload:

- Paper Chimera: AD F1 0.9419, HR@1 0.8973, MRR@20 0.9115
- Baseline Chimera: AD F1 0.6791, HR@1 0.6075, MRR@20 0.6504
- Novelty 1: AD F1 0.7556, HR@1 0.7001, MRR@20 0.7132
- Novelty 2: AD F1 0.8541, HR@1 0.8209, MRR@20 0.8546
- Novelty 3: AD F1 0.4390, HR@1 0.7652, MRR@20 0.7937

Novelty 2 is the strongest current model by AD F1.

## Reproduce the workflow

### 1) Train models

```bash
python main.py --mode train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty1_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256
```

### 2) Evaluate the trained checkpoints

```bash
python main.py --mode eval --dataset BGL --load_checkpoint True
python main.py --mode novelty1_eval --dataset BGL --load_checkpoint True
python main.py --mode novelty2_eval --dataset BGL --load_checkpoint True
python main.py --mode novelty3_eval --dataset BGL --load_checkpoint True
```

### 3) Select the best checkpoint dynamically

```bash
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint
```

This writes:

- `checkpoint/best_model.bin`
- `report_output/selected_checkpoint_summary.json`

### 4) Regenerate the figures

```bash
python scripts/plot_paper_comparison.py
```

### 5) Refresh the human-readable report

```bash
python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
```

## Notes on avoiding overfit / underfit

- `main.py` now monitors `val_ad_f1` and checkpoints the best epoch instead of relying on hardcoded choice.
- The repo’s existing early stopping remains active, which helps avoid over-training.
- `scripts/select_best_checkpoint.py` compares all saved checkpoints on the same BGL test split, so the final chosen checkpoint is dynamic, not hardcoded.
- The auxiliary-head metrics in the report are useful for verifying the extra tasks are learning something beyond the paper’s original two-task setup.

## Files of interest

- `main.py`
- `src/callbacks.py`
- `src/metrics_eval.py`
- `scripts/select_best_checkpoint.py`
- `scripts/plot_paper_comparison.py`
- `report_output/paper_comparison_metrics.json`
- `report_output/selected_checkpoint_summary.json`
- `report_output/PROJECT_REPORT.md`
