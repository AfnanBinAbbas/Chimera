# FaultGuard-AI Novelty Implementations

This document describes the implemented novelties in this repository and their mathematical formulations.

## Novelty 1: Cross-Domain Adaptation Module

### Goal
Learn domain-invariant representations so the model can generalize across software system types (zero-shot), and adapt quickly with few labeled samples (few-shot).

### Architecture Additions
- **Gradient Reversal Layer (GRL)** between shared encoder and domain discriminator
- **Domain Discriminator** predicting domain label from shared features
- **CrossDomainChimera** training with adversarial domain objective

Implemented in:
- `src/domain_adaptation.py`
- `src/cross_domain_dataset.py`
- `main.py` modes: `cd_train`, `cd_eval`, `few_shot`

### Formulation

Let:
- \(x\) = input log sequence
- \(y^{ad}\) = anomaly label
- \(y^{rca}\) = root-cause localization labels
- \(d\) = domain label
- \(E_s(\cdot)\) = shared encoder
- \(E_{ad}(\cdot), E_{rca}(\cdot)\) = task-specific encoders
- \(D(\cdot)\) = domain discriminator

Shared feature:
\[
z_s = E_s(x)
\]

Domain adversarial term using GRL:
\[
\mathcal{L}_{dom} = \text{CE}(D(\text{GRL}(z_s)), d)
\]

Base Chimera objective:
\[
\mathcal{L}_{chimera} = \mathcal{L}_{cls} + 2\mathcal{L}_{rank} + 0.001\mathcal{L}_{diff} + 0.5\mathcal{L}_{jsd}
\]

Total objective (implemented):
\[
\mathcal{L}_{total}^{(N1)} = \mathcal{L}_{chimera} + \lambda_{dom}\mathcal{L}_{dom}
\]

where \(\lambda_{dom}\) is linearly warmed up during training.

### Few-shot Adaptation
Given support set \(S_t\) from target domain:
- Freeze shared encoder \(E_s\)
- Fine-tune task heads on \(S_t\) for a few steps

Optimization:
\[
\theta_{heads}^{*} = \arg\min_{\theta_{heads}} \sum_{(x,y)\in S_t} \mathcal{L}_{task}(x,y)
\]

---

## Novelty 2: Multi-Task Expansion

### Goal
Extend Chimera from 2 tasks (AD, RCA) to 6 tasks by adding:
1. Log parsing
2. Failure type classification
3. Impact assessment
4. Remediation suggestion

Implemented in:
- `src/multitask_expansion.py`
- `src/multitask_dataset.py`
- `main.py` modes: `mt_train`, `mt_eval`

### Added Task Heads
- **Log Parsing Head**: sequence-level reconstruction in embedding space
- **Failure Classification Head**: multi-class classifier
- **Impact Head**: scalar regression with sigmoid output
- **Remediation Head**: action classifier

### Multi-task Objective

Let:
- \(\hat{p}\) be predicted parsing sequence embeddings
- \(p\) be parsing targets
- \(\hat{f}, f\) be predicted/true failure class
- \(\hat{i}, i\) be predicted/true impact score
- \(\hat{r}, r\) be predicted/true remediation class

Auxiliary losses:
\[
\mathcal{L}_{parse} = \text{MSE}(\hat{p}, p)
\]
\[
\mathcal{L}_{fail} = \text{CE}(\hat{f}, f)
\]
\[
\mathcal{L}_{impact} = \text{MSE}(\hat{i}, i)
\]
\[
\mathcal{L}_{rem} = \text{CE}(\hat{r}, r)
\]

Total objective (implemented):
\[
\mathcal{L}_{total}^{(N2)} =
\mathcal{L}_{chimera}
\lambda_{parse}\mathcal{L}_{parse}
\lambda_{fail}\mathcal{L}_{fail}
\lambda_{impact}\mathcal{L}_{impact}
\lambda_{rem}\mathcal{L}_{rem}
\]

where each \(\lambda\) is configurable from CLI:
- `--mt_logparse_lambda`
- `--mt_failure_lambda`
- `--mt_impact_lambda`
- `--mt_remediation_lambda`

### Explicit Multi-task Label Format

For true labeled training, create files:
- `data/<DATASET>/n_train_mt.txt`, `an_train_mt.txt`
- `data/<DATASET>/n_dev_mt.txt`, `an_dev_mt.txt`
- `data/<DATASET>/n_test_mt.txt`, `an_test_mt.txt`

Each line must be:
`ad_label:src_ids:rca_mask:parse_ids:failure_type:impact_score:remediation_action`

Example:
`1:12 40 8 ...:0 1 0 ...:901 901 17 ...:3:0.74:5`

Meaning:
- `ad_label`: 0 or 1
- `src_ids`: tokenized log event IDs (space-separated)
- `rca_mask`: binary relevance vector (space-separated)
- `parse_ids`: parser target event/template IDs (space-separated)
- `failure_type`: integer class ID
- `impact_score`: float severity in [0,1]
- `remediation_action`: integer class ID

If these files are absent, `mt_train`/`mt_eval` automatically fallback to weak-label mode.

---

## Novelty 3: Advanced Task Interaction Mechanisms

### Goal
Replace static interactions with dynamic mechanisms that adapt to system state.

Implemented in:
- `src/advanced_interaction.py`
- `main.py` modes: `ai_train`, `ai_eval`

### Components
1. **Graph-based interaction**
   - Tasks are nodes in a learned graph.
   - Adjacency matrix is learned from task embeddings by attention.
2. **Hierarchical grouping**
   - Grouped as detection, localization, and analysis tasks.
   - Dense intra-group and sparse inter-group message mixing.
3. **Dynamic task prioritization**
   - A system-state encoder predicts per-task weights.
   - Weights modulate each task’s contribution to total loss.

### Formulation

Let \(H \in \mathbb{R}^{B\times T\times D}\) be task embeddings.

Graph interaction:
\[
A = \text{softmax}\left(\frac{QK^\top}{\sqrt{D}}\right),\quad
\tilde{H} = \text{LN}(H + AV)
\]

Hierarchical interaction:
\[
\hat{H} = \text{HierMix}(\tilde{H})
\]

Dynamic prioritization:
\[
w = \text{softmax}(g(s)),\quad s=\text{pool}(\hat{H})
\]

with \(w=[w_{ad},w_{rca},w_{parse},w_{fail},w_{impact},w_{rem}]\).

Final objective:
\[
\mathcal{L}_{total}^{(N3)} =
w_{ad}\mathcal{L}_{ad}
w_{rca}\mathcal{L}_{rca}
w_{parse}\lambda_{parse}\mathcal{L}_{parse}
w_{fail}\lambda_{fail}\mathcal{L}_{fail}
w_{impact}\lambda_{impact}\mathcal{L}_{impact}
w_{rem}\lambda_{rem}\mathcal{L}_{rem}
 + 0.001\mathcal{L}_{diff}+0.5\mathcal{L}_{jsd}
\]

---

## Novelty 4: Unified Evaluation Benchmark

### Goal
Provide one standardized protocol to evaluate all tasks across multiple datasets/domains.

Implemented in:
- `src/unified_benchmark.py`
- `main.py` mode: `unified_benchmark`

### Benchmark Outputs
- AD: Precision, Recall, F1
- RCA: HR@1, MRR
- Failure classification: Accuracy, Macro-F1
- Impact assessment: MSE
- Remediation suggestion: Accuracy
- Efficiency: inference time

### Standardized Protocol
For each dataset \(d \in \mathcal{D}\), compute:
\[
\mathbf{m}_d =
[\text{F1}_{ad}, \text{HR@1}_{rca}, \text{MRR}_{rca}, \text{Acc}_{fail},
\text{F1}^{macro}_{fail}, \text{MSE}_{impact}, \text{Acc}_{rem}, t_{inf}]
\]

and save:
\[
\mathcal{R} = \{\mathbf{m}_d \mid d \in \mathcal{D}\}
\]
as JSON report (`benchmark/results.json` by default).

### Dataset note in current repo
- `data/BGL/` is already in split format (`n_train.txt`, `an_train.txt`, etc.).
- `data/tbird2/` currently contains a single raw file (`tbird2`), so it must be preprocessed into split files before full benchmark coverage.

---

## CLI Quick Reference

### Novelty 1
- Cross-domain train:
  - `python main.py --mode cd_train --source_datasets BGL Thunderbird --epochs 150`
- Zero-shot eval:
  - `python main.py --mode cd_eval --source_datasets BGL Thunderbird --target_dataset GAIA --load_checkpoint True`
- Few-shot adaptation:
  - `python main.py --mode few_shot --source_datasets BGL Thunderbird --target_dataset GAIA --shots 50 --adaptation_steps 200 --load_checkpoint True`

### Novelty 2
- Multi-task train:
  - `python main.py --mode mt_train --dataset BGL --epochs 150`
- Multi-task eval:
  - `python main.py --mode mt_eval --dataset BGL --load_checkpoint True`

### Novelty 3
- Advanced interaction train:
  - `python main.py --mode ai_train --dataset BGL --epochs 150`
- Advanced interaction eval:
  - `python main.py --mode ai_eval --dataset BGL --load_checkpoint True`

### Novelty 4
- Unified benchmark:
  - `python main.py --mode unified_benchmark --benchmark_datasets BGL tbird2 --benchmark_output benchmark/results.json --load_checkpoint True`
