---
title: "NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models"
collection: publications
permalink: /publications/2026-nsrboost
category: conferences
date: 2026-01-31
venue: 'Arxiv'
excerpt: ''
citation: '<u>Dabiao Ma</u>, <u><strong>Ziming Dai</strong></u>, Zhimin Xin, Shu Wang, Jian Yang, and Haojun Fei. "TS-PEFT: Unveiling Token-Level Redundancy in Parameter-Efficient Fine-Tuning." arXiv preprint arXiv:2511.16147 (2026).'
---

<img src="../images/nsrboost.png" alt="Comparison between standard PEFT and our TS-PEFT framework." style="max-width: 100%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">


### ✨ Proposed Method
This paper introduces **NSR-Boost**, a neuro-symbolic residual boosting framework specifically designed to upgrade industrial legacy models in high-concurrency production environments without prohibitive retraining costs. The core advantage of NSR-Boost is its "non-intrusive" nature: it treats the legacy model as a frozen entity and performs targeted repairs exclusively on "hard regions" where predictions fail. The framework operates in three key stages: first, it locates hard regions through residual analysis; second, it generates interpretable experts using a bi-level approach (generating symbolic code structures via an LLM and fine-tuning parameters using Tree-structured Parzen Estimator optimization); finally, it dynamically integrates these experts with the legacy model's output using a lightweight aggregator.

### 📊 Experimental Results
* **Superior Performance:** Experimental results demonstrate that the NSR-Boost framework significantly outperforms state-of-the-art (SOTA) baselines.
* **Cost-Effective & Safe Upgrades:** It successfully avoids the systemic risks and prohibitive retraining costs typically associated with upgrading legacy models in production environments.
* **High Interpretability:** By utilizing an LLM to generate symbolic code structures (such as Python functions) as experts, the framework maintains crucial interpretability and provides actionable feedback constraints for industrial applications.

### 🤝 Collaborating Institutions

Tianjin University; Qfin Holdings, Inc.


<div style="max-width: 80%; margin: 0 auto 30px auto; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 50px; width: auto; margin: 0;">
  <img src="../images/logo/logo_qifu.png" alt="Qfin Holdings, Inc." style="height: 50px; width: auto; margin: 0;">
</div>