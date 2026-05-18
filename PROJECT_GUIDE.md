# FaultGuard-AI: End-to-End Log Fault Diagnosis

This project is an advanced implementation and extension of the **Chimera** framework for automated log anomaly detection and root cause localization. It introduces three major novelties beyond the original research paper to improve cross-dataset generalization and diagnostic depth.

---

## 1. Environment Setup

The project is optimized for **Python 3.10.12** and requires a GPU (e.g., **NVIDIA RTX 4090**) for efficient training of the 150-epoch models.

### Installation
```bash
# 1. Ensure you are using Python 3.10
# 2. Install dependencies
pip install -r requirements.txt
```

### Key Dependencies
* **PyTorch (>=1.13.1):** Core deep learning framework.
* **PyTorch Lightning (==1.1.2):** High-level training orchestration.
* **Scikit-Learn (>=1.0.2):** Used for evaluation metrics (ndcg, f1).
* **NumPy (<2.0.0):** Essential for compatibility with legacy data structures.

---

## 2. Model Architectures (The Novelties)

The project includes four distinct model variants:

### Baseline (Original Chimera)
Implements the core paper architecture:
* **Shared-Private Encoder (SPE):** Separates shared features from task-specific (AD/RCA) features.
* **JS-Divergence Alignment:** Syncs attention weights with root-cause probabilities.
* **Metrics:** Focused only on Anomaly F1 and RCA HR@1.

### Novelty 1: Domain-Adaptive Chimera
Designed for **cross-domain generalization** (e.g., training on BGL and deploying on Thunderbird).
* Adds a **Domain Discriminator** head with a **Gradient Reversal Layer (GRL)**.
* Forces the encoder to learn features that are "domain-invariant," meaning the model focuses on fault patterns rather than specific system log styles.

### Novelty 2: Multi-Task Expansion (Best Performer)
Expands the diagnosis from "if and where" to "what and how bad."
* **Extra Heads:**
    * **Failure Class:** Categorizes the fault type (e.g., Network, Disk).
    * **Impact Score:** Predicts the severity of the incident.
    * **Remediation:** Suggests a fix action.
    * **Semantic Parsing:** Ensures the internal vectors represent log meaning accurately.
* **GAN Augmentation:** Uses a Generative Adversarial Network to create synthetic anomalies, improving training on unbalanced datasets.

### Novelty 3: Advanced Interaction Chimera
Introduces a "knowledge graph" between different task heads.
* **Graph Interaction:** Allows the Anomaly Detection head to "talk" to the Root Cause head before making a final decision.
* **Dynamic Loss Weighting:** Automatically adjusts the importance of each task during training based on their difficulty.

---

## 3. Dataset Preparation

The project currently supports two primary datasets:

### **BGL (Blue Gene/L)**
* **Status:** Fully prepared.
* **Location:** `data/BGL/`
* **Files:** `n_train.txt`, `an_train.txt`, `emd_dict.json`, etc.

### **Thunderbird (tbird2)**
* **Status:** Raw data only.
* **Location:** `data/tbird2` (31GB raw file).
* **Preparation Needed:** You must run the **Drain3** parser to split the earliest 10 million logs into the expected `n_test.txt` and `an_test.txt` format before this dataset can be used in the scripts.

---

## 4. Running the Pipeline

The pipeline is now **fully automated**. Follow these steps in order:

### Step 1: Training
Choose a novelty and run for 150 epochs (recommended for accuracy > 94%).
```bash
# Multi-Task (Best Accuracy)
python3.10 main.py --mode novelty2_train --dataset BGL --epochs 150 --batch_size 256 --hard_device cuda

# Graph Interaction
python3.10 main.py --mode novelty3_train --dataset BGL --epochs 150 --batch_size 256 --hard_device cuda
```

### Step 2: Generate Metrics
Run the evaluation script to calculate F1, HR@1, and MRR. This step now **automatically syncs** your results for the plotting script.
```bash
python3.10 scripts/generate_project_report.py --dataset BGL --device cuda --batch-size 128
```

### Step 3: Update Graphs
Run the plotting script to generate the comparison figures against the original research paper.
```bash
python3.10 scripts/plot_paper_comparison.py --dataset BGL --device cuda --batch-size 128
```

---

## 5. Achieved Results vs. Paper (BGL)

By following the 150-epoch training protocol on GPU, this implementation achieves the following results:

| Metric | Paper (Chimera) | Our Implementation (Novelty 2) | Status |
| :--- | :---: | :---: | :---: |
| **Anomaly Detection F1** | 94.19% | **98.55%** | ** Surpassed Paper** |
| **Root HR@1** | 89.73% | 85.43% | Near State-of-the-art |
| **Root MRR** | 91.15% | 90.19% | Competitive |

---

## 6. Automation Technical Details

* **`sync_paper_comparison`**: This new function in `generate_project_report.py` ensures that `report_output/paper_comparison_metrics.json` is always up-to-date with your latest training weights.
* **Dynamic Plotting**: `plot_paper_comparison.py` now extracts your real auxiliary head scores (Failure/Remediation) directly from JSON, ensuring the "Extra Heads" graph matches your training data perfectly.