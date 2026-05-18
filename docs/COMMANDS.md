# Command Reference

This file documents the user-facing commands available in the repository.

## 1) Install dependencies

```bash
pip install -r requirements.txt
```

## 2) Main training and evaluation entrypoint

File: `main.py`

### Shared flags

- `--hard_device`: `cuda` or `cpu` (default: `cuda`)
- `--gpu_index`: CUDA device index when using GPU (default: `0`)
- `--mode`: selects the workflow
- `--dataset`: dataset folder under `data/` (default: `BGL`)
- `--epochs`: training epochs (default: `10`)
- `--batch_size`: batch size (default: `256`)
- `--lr`: learning rate (default: `1e-3`)
- `--warmup_epochs`: warmup epochs (default: `8`)
- `--model_save_path`: checkpoint output directory (default: `checkpoint`)
- `--load_checkpoint`: load the saved model before evaluation or resumed use
- `--threshold`: anomaly decision threshold (default: `0.5`)
- `--auto_threshold`: auto-calibrate the decision threshold during eval (default: `True`)

### Available modes

#### Base Chimera

```bash
python main.py --mode train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode eval --dataset BGL --load_checkpoint True
```

#### Novelty 1: domain adaptation

```bash
python main.py --mode novelty1_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty1_eval --dataset BGL --load_checkpoint True
```

#### Novelty 2: multitask expansion

```bash
python main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty2_eval --dataset BGL --load_checkpoint True
```

#### Novelty 3: advanced interaction

```bash
python main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256
python main.py --mode novelty3_eval --dataset BGL --load_checkpoint True
```

#### Unified benchmark

```bash
python main.py --mode unified_benchmark --dataset BGL
```

### Notes

- Training uses validation AD F1 (`val_ad_f1`) for checkpointing and early stopping.
- Evaluation uses the saved checkpoint from `checkpoint/<ModelName>_model.bin` when `--load_checkpoint True` is provided.
- If `--mode` is not one of the explicit train/eval/unified options, `main.py` falls back to testing on the BGL loaders.

## 3) Cross-domain / few-shot entrypoint

File: `nov_1.py`

### Shared flags

- `--hard_device`: `cuda` or `cpu` (default: `cuda`)
- `--gpu_index`: CUDA device index (default: `0`)
- `--load_checkpoint`: load the saved cross-domain checkpoint
- `--model_save_path`: checkpoint output directory (default: `checkpoint`)
- `--epochs`: training epochs (default: `50`)
- `--batch_size`: batch size (default: `128`)
- `--warmup_epochs`: warmup epochs (default: `10`)
- `--lr`: learning rate (default: `1e-3`)
- `--accumulate_grad_batches`: gradient accumulation steps (default: `1`)
- `--mode`: one of `cd_train`, `cd_eval`, `few_shot`
- `--dataset`: source dataset name (default: `BGL`)
- `--source_datasets`: list of source datasets for cross-domain training
- `--target_dataset`: target dataset for evaluation or few-shot adaptation (default: `GAIA`)
- `--num_domains`: number of domains
- `--domain_lambda`: domain loss weight (default: `0.1`)
- `--domain_lambda_max`: max domain loss weight (default: `1.0`)
- `--shots`: number of few-shot samples (default: `50`)
- `--adaptation_steps`: few-shot adaptation steps (default: `200`)
- `--adaptation_lr`: few-shot adaptation learning rate (default: `1e-4`)
- `--rca_dim`: RCA class count (default: `20`)

### Available modes

#### Cross-domain training

```bash
python nov_1.py --mode cd_train --source_datasets BGL --dataset BGL --epochs 50 --batch_size 128
```

#### Cross-domain evaluation

```bash
python nov_1.py --mode cd_eval --source_datasets BGL --target_dataset GAIA --dataset BGL --load_checkpoint
```

#### Few-shot adaptation

```bash
python nov_1.py --mode few_shot --source_datasets BGL --target_dataset GAIA --dataset BGL --load_checkpoint
```

## 4) Utility scripts

### Dynamic checkpoint selection

```bash
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint
```

What it does:

- scans every `*_model.bin` file in `checkpoint/`
- evaluates each checkpoint on the BGL test loaders
- writes the best run to `checkpoint/best_model.bin`
- saves the full summary to `report_output/selected_checkpoint_summary.json`

### Paper comparison plots

```bash
python scripts/plot_paper_comparison.py
```

What it does:

- generates `report_output/fig_bgl_paper_vs_novelties.png`
- generates `report_output/fig_bgl_novelty_auxiliary_heads.png`

### Project report regeneration

```bash
python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
```

What it does:

- refreshes `report_output/PROJECT_REPORT.md`
- refreshes `report_output/metrics_summary.json`
- refreshes `report_output/metrics_summary_flat.csv`
- refreshes `report_output/lightning_metrics_concat.csv` when Lightning logs are present
- copies the report to the repository root as `PROJECT_REPORT.md`

## 5) Recommended end-to-end order

```bash
pip install -r requirements.txt
python main.py --mode train --dataset BGL --epochs 150 --batch_size 256
python scripts/select_best_checkpoint.py --dataset BGL --device cpu --batch-size 128 --checkpoint-dir checkpoint
python main.py --mode eval --dataset BGL --load_checkpoint True
python scripts/plot_paper_comparison.py
python scripts/generate_project_report.py --dataset BGL --device cpu --batch-size 128
```
