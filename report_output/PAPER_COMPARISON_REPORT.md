# FaultGuard-AI Complete Documentation and Paper Comparison

Generated (UTC): 2026-05-08 13:39:34

## 1) Scope
This report documents implementation, run protocol, measured results, and comparison against the reference Chimera paper.

## 2) Implemented components
- Baseline `Chimera` model from original repo
- Novelty 1: `DomainAdaptiveChimera` (domain adversarial module)
- Novelty 2: `MultiTaskChimera` (auxiliary tasks + latent GAN)
- Novelty 3: `AdvancedInteractionChimera` (dynamic interaction weighting)
- Unified benchmarking path and staged result export

Key code files:
- `main.py`
- `src/models.py`
- `src/domain_adaptation.py`
- `src/multitask_expansion.py`
- `src/advanced_interaction.py`
- `src/multitask_dataset.py`
- `src/unified_benchmark.py`

## 3) Experimental protocol used
- Dataset: BGL split files in `data/BGL`
- Hardware: NVIDIA RTX 4090 (CUDA enabled in current environment)
- Loader style: same existing project loaders and evaluation function `compute_ad_rca_metrics`
- Thresholding: auto-calibrated in eval path when enabled

Commands executed (representative):
- `python main.py --mode train --dataset BGL --epochs 10 --batch_size 256 --gpu_index 0`
- `python main.py --mode novelty1_train --dataset BGL --epochs 30 --batch_size 256 --gpu_index 0 --load_checkpoint true`
- `python main.py --mode novelty2_train --dataset BGL --epochs 20 --batch_size 256 --gpu_index 0`
- `python main.py --mode novelty3_train --dataset BGL --epochs 20 --batch_size 256 --gpu_index 0`
- `python main.py --mode eval/novelty*_eval --dataset BGL --batch_size 256 --gpu_index 0 --load_checkpoint true`

## 4) Our measured staged results (BGL)

| Model | AD Accuracy | AD F1 | AD Precision | AD Recall | RCA HR@1 | RCA MRR |
|---|---:|---:|---:|---:|---:|---:|
| baseline_chimera_eval | 93.28% | 67.91% | 57.61% | 82.69% | 60.75% | 65.04% |
| novelty1_eval | 95.60% | 75.56% | 72.34% | 79.09% | 70.01% | 71.32% |
| novelty2_eval | 97.16% | 85.41% | 76.40% | 96.83% | 82.09% | 85.46% |
| novelty3_eval | 80.37% | 43.90% | 29.09% | 89.37% | 76.52% | 79.37% |

Best AD run in this series: **novelty2_eval** with 97.16% AD accuracy and 85.41% AD F1.

## 5) Reference paper values used for comparison
Source: `https://arxiv.org/html/2509.24364v1` (tables in Experimental Evaluation)

BGL, Chimera (paper):
- AD Precision: 91.22%
- AD Recall: 97.37%
- AD F1: 94.19%
- RCA HR@1: 89.73%
- RCA MRR: 91.15%

## 6) Side-by-side comparison: best ours vs paper Chimera on BGL

| Metric | Paper Chimera | Best Ours (novelty2_eval) | Delta (ours - paper) |
|---|---:|---:|---:|
| AD F1 | 94.19% | 85.41% | -8.78 pp |
| RCA HR@1 | 89.73% | 82.09% | -7.64 pp |
| RCA MRR | 91.15% | 85.46% | -5.69 pp |

## 7) Interpretation
- In this run series, novelty2 is the strongest variant among implemented novelties.
- It improves strongly over the local baseline Chimera checkpoint from this session.
- It does **not** surpass the paper Chimera BGL headline metrics yet.
- Therefore, the >98% target is only reached for AD accuracy in novelty2? **No** (97.16% < 98%).

## 8) Reproducibility artifacts
- `report_output/staged_novelty_results.json` (staged metrics)
- `report_output/paper_comparison_metrics.json` (paper-vs-ours structured comparison)
- `checkpoint/*.bin` (saved checkpoints)

## 9) Caveats
- The paper compares against fully tuned settings and 5-run means ± std; this run is a single operational run series.
- Novelty modules were integrated into an evolving codebase and may need additional hyperparameter sweeps for fair best-of-best comparison.