Security & Privacy of Machine Learning (CS-452)
Project Proposal
Group Members:
1. Afnan Bin Abbas (2022048)
2. Rooshan Riaz (2022506)
3. Shameer Awais (2022 428 )
4. Naqi Raza (2022574)
5. Yasir Khan (2022455)
Submission Date: 1 1 /03/
Ghulam Ishaq Khan Institute of Engineering Sciences and Technology
1 EXECUTIVE SUMMARY

This project proposes significant extensions to Chimera , a state-of-the-art log-based fault
diagnosis framework introduced in the paper "United We Stand: Towards End-to-End Log-
based Fault Diagnosis via Interactive Multi-Task Learning". While Chimera successfully
combines anomaly detection and root cause localization through interactive multi-task
learning, this project will extend its capabilities in three novel directions:

    Cross-Domain Generalization : Adapting Chimera to work across diverse software
    system types (microservices, cloud platforms, IoT systems)
    Multi-Task Expansion : Incorporating additional diagnostic tasks including log
    parsing, failure type classification, impact assessment, and remediation suggestion
    Advanced Task Interaction : Designing novel interaction mechanisms including
    graph-based, hierarchical, and dynamic task prioritization

The proposed Chimera-X framework aims to achieve 15 - 25% improvement over the
original Chimera, establishing a new state-of-the-art in comprehensive log-based system
reliability engineering.
2 BACKGROUND AND LITERATURE REVIEW
2.1 LOG-BASED FAULT DIAGNOSIS

Software systems generate massive volumes of logs that capture runtime behaviors. Analyzing
these logs is crucial for:

    Anomaly Detection (AD) : Identifying when system behavior deviates from normal
    patterns
    Root Cause Localization (RCL) : Pinpointing the specific logs/events that caused
    failures

Current approaches treat these as separate tasks, leading to diagnostic bias accumulation and
suboptimal performance.
2.2 THE CHIMERA FRAMEWORK

The original Chimera paper introduced several innovations:

Component Description Limitation

Interactive Multi-
Task Learning

AD and RCL share knowledge
bidirectionally

Limited to only 2 tasks

Shared-Private
Encoder (SPE)

Disentangles task-specific and
shared features

Tested only on 3 datasets

Cross-granularity
Alignment

Aligns diagnostic results via JS
divergence

Static interaction only

Sequence-driven
Localizer

Trains without root cause labels No dynamic task prioritization

2.3 RESEARCH GAP

Despite Chimera's success, significant gaps remain:

Gap Description Why Important

Limited Domain
Coverage

Tested only on BGL,
Thunderbird, System A

Real-world systems are
diverse

Task Incompleteness Only 2 diagnostic tasks SREs need comprehensive
diagnosis

Static Interaction Fixed interaction mechanism System context changes
dynamically

No Cross-Domain
Adaptation

Assumes training/test from same
domain

New systems lack labeled
data

2.4 RELATED RECENT WORK

Paper Approach Limitation

LogLM (2024) LLM-based log analysis Computationally expensive

MIRTL (2025) Multi-task representation learning No task interaction mechanism

RIT (2024) Relational interaction transformer Fixed task relations

Eadro (2023) End-to-end troubleshooting Multi-source data required

3 PROBLEM STATEMENT
3.1 CORE PROBLEM

Current log-based fault diagnosis systems fail to provide comprehensive, adaptable, and
efficient diagnosis due to:

    Diagnostic Bias Accumulation : Errors from anomaly detection propagate to root cause
    localization
    Domain Specificity : Models trained on one system fail on others (up to 17%
    performance drop)

    Task Isolation : Related diagnostic tasks (parsing, classification, remediation) are
    treated separately
    Static Architectures : Task interaction mechanisms don't adapt to changing system
    conditions

3.2 RESEARCH QUESTIONS

This project addresses the following research questions:

RQ Question Novelty Contribution

RQ1 How can Chimera be extended to work across
diverse software system types without
retraining?

Cross-domain adaptation

RQ2 Can incorporating additional diagnostic tasks
improve overall fault diagnosis performance?

Multi-task expansion

RQ3 What interaction mechanisms enable optimal
knowledge transfer among multiple diagnostic
tasks?

Advanced interaction design

RQ4 How can task priorities be dynamically adjusted
based on system state?

Dynamic task weighting

4 PROPOSED NOVELTY AND CONTRIBUTIONS
4.1 NOVELTY 1: CROSS-DOMAIN ADAPTATION MODULE

Problem : Chimera assumes training and testing data come from the same system distribution. Real-
world deployment requires models that work on new, unseen systems.

Solution : A domain adaptation layer that learns system-invariant representations.

class CrossDomainChimera:

    Domain-adversarial training to remove system-specific features
    Few-shot adaptation for new systems with minimal labeled data
    Zero-shot inference capability

4.2 NOVELTY 2: MULTI-TASK EXPANSION

Problem : Real SRE workflows require more than just anomaly detection and root cause
localization.

Solution : Extend Chimera to handle 4 additional tasks simultaneously:

New Task Description Output Format Related
Work

Log Parsing Extract structured
templates

Event templates Drain, Spell

Failure
Classification

Categorize failure
types

10+ failure categories LogClass

Impact
Assessment

Estimate severity and
scope

Severity score (0-1) +
affected components

ImpactRank

Remediation
Suggestion

Recommend
recovery actions

Action sequence RemediateRL

4.3 NOVELTY 3: ADVANCED TASK INTERACTION MECHANISMS

Problem : Chimera uses a single, static interaction mechanism. Different system states require different
interaction patterns.

Solution : Three novel interaction mechanisms:

A. Graph-Based Task Interaction

    Model tasks as nodes in a graph
    Learn task relations dynamically during training
    Message passing between related tasks

B. Hierarchical Task Grouping

    Group related tasks (detection tasks, localization tasks, analysis tasks)
    Intra-group dense interaction, inter-group sparse interaction
    Inspired by cognitive architectures

C. Dynamic Task Prioritization

    System state encoder monitors current logs
    Predicts which tasks are most important now
    Allocates computational resources accordingly

4.4 NOVELTY 4: UNIFIED EVALUATION BENCHMARK

Problem : No comprehensive benchmark exists for multi-task log diagnosis across domains.

Solution : Create and release:

Component Details

Cross-Domain Dataset Suite 5 system types × 100K logs each

Multi-Task Labels All 6 tasks labeled for 50K logs

Evaluation Protocol Standardized metrics and splits

Baseline Implementations All comparison methods open-sourced

5 METHODOLOGY

Phase 1: Foundation

    Set up Chimera codebase and reproduce results
    Collect and preprocess 5 new datasets (microservices, e-commerce, cloud, IoT, HPC)
    Implement unified data loader interface

Phase 2: Cross-Domain Adaptation

    Implement domain-adversarial training
    Design few-shot adaptation protocol
    Evaluate zero-shot performance across domains

Phase 3: Multi-Task Expansion

    Design and implement 4 additional task heads
    Create labeling pipeline for new tasks
    Implement task relation learning

Phase 4: Advanced Interaction

    Implement graph-based interaction
    Implement hierarchical grouping
    Implement dynamic prioritization
    Ablation studies and comparison

Phase 5: Evaluation and Documentation

    Comprehensive experiments
    Paper writing
    Code release and documentation

5.1 DATASETS

Dataset System Type Size

BGL HPC Supercomputer 4M logs

Thunderbird HPC Supercomputer 10M logs

System A Cloud Platform 2M logs

GAIA Microservices 1.5M logs

OpenStack Cloud Infrastructure 3M logs

IoT- 23 IoT Systems 2M logs

5.2 EVALUATION METRICS

Task Metrics

Anomaly Detection Precision, Recall, F1-Score

Root Cause Localization HR@k, MAP@k, MRR (k=1,3,5)

Log Parsing Accuracy, F1 (template-level)

Failure Classification Accuracy, Macro-F

Impact Assessment MSE, Spearman Correlation

Remediation Suggestion BLEU Score, Execution Success Rate

Cross-Domain Zero-shot F1, Few-shot F1 (shots=10,50,100)

Efficiency Inference time, Parameter count, FLOPs

6 CONCLUSION

This project proposes Chimera-X , a significant extension to the state-of-the-art Chimera framework
for log-based fault diagnosis. By addressing three critical limitations—limited domain coverage,
incomplete task set, and static interaction mechanisms—this project will deliver:

    A cross-domain adaptation module enabling zero-shot deployment on new systems
    A multi-task expansion incorporating 4 additional diagnostic tasks
    Three novel interaction mechanisms for optimal knowledge transfer
    A comprehensive benchmark for multi-task log diagnosis

The proposed extensions are expected to achieve 15 - 25% improvement over the original Chimera,
establishing a new state-of-the-art in comprehensive system reliability engineering. All code, datasets,
and models will be open-sourced to enable reproduction and future research.
7 REFERENCES

    He, M., et al. " United We Stand: Towards End-to-End Log-based Fault Diagnosis via
    Interactive Multi-Task Learning ." arXiv:2509.24364v1.
    Zhang, X., et al. " Robust log-based anomaly detection on unstable log data. " ESEC/FSE
    Lee, C., et al. " Eadro: An end-to-end troubleshooting framework for microservices on
    multi-source data. " ICSE 2023.
    Wittkopp, T., et al. " LogRCA: Log-based root cause analysis for distributed services. "
    Euro-Par 2024.
    Zhang, C., et al. " Metalog: Generalizable cross-system anomaly detection from logs with
    meta-learning. " ICSE 2024.
    Zhang, Y., & Yang, Q. " A survey on multi-task learning. " IEEE TKDE 2021.
    He, M., et al. " Weakly-supervised log-based anomaly detection with inexact labels via
    multi-instance learning ." ICSE 2025.
    Duan, C., et al. " AFAFormer: A general augmentation framework for log-based anomaly
    detection. " ISSRE 2023.
