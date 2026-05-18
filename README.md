# FaultGuard-AI Core Research Engine

This repository contains the core research implementation and models for **FaultGuard-AI**, an end-to-end, multi-task log-based fault diagnosis platform.

## Complete Documentation & Setup Guide

The comprehensive setup instructions, quick start guides, dataset preparations, training pipeline, and experimental evaluation results have been unified under the main **FaultGuard-AI** repository.

 **Please visit the main repository for the complete master documentation:**
[https://github.com/AfnanBinAbbas/FaultGuard-AI](https://github.com/AfnanBinAbbas/FaultGuard-AI)

---

## Core Engine Project Structure

```
chimera/
 ├─ checkpoint/ # Saved models (ignored in Git, local only)
 ├─ data/ # Log data (ignored in Git, local only)
 ├─ glove/ # Pre-trained Language Models for Log Embedding
 ├─ docs/ # Documentation guides and references
 ├─ report_output/ # Generated evaluation metrics, summaries, and plots
 ├─ scripts/ # Automation scripts for plotting and checkpoint selection
 ├─ src/ # Core model source code
 │ ├─ advanced_interaction.py # Dual-attention interaction module (Novelty 3)
 │ ├─ callbacks.py # Custom training callbacks
 │ ├─ domain_adaptation.py # Domain adaptation layers (Novelty 2)
 │ ├─ metrics_eval.py # Robust metrics evaluation pipeline
 │ ├─ models.py # Core FaultGuard-AI model architectures
 │ ├─ multitask_dataset.py # Multi-task dataset loader
 │ └─ multitask_expansion.py # Extended 6-task training pipeline (Novelty 1)
 ├─ main.py # Entry point for training and evaluation
 └─ requirements.txt # Python package dependencies
```