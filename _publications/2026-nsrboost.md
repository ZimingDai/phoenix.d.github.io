---
title: "NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models"
collection: publications
permalink: /publications/2026-nsrboost
category: conferences
date: 2026-01-31
sort_date: 2026-08-09
venue: 'SIGKDD'
rank: a
rank_label: 'CCF-A'
teaser: 'nsrboost.png'
excerpt: ''
authors: '<strong>Ziming Dai</strong>, Dabiao Ma, Jinle Tong, Mengyuan Han, Jian Yang, Hongtao Liu, Haojun Fei, and Qing Yang'
highlight: 'A neuro-symbolic residual boosting framework for upgrading industrial legacy models without full retraining.'
keywords: ['Neuro-Symbolic AI', 'Residual Boosting', 'Legacy Models']
paperurl: 'https://arxiv.org/abs/2601.10457'
citation: '<strong>Ziming Dai</strong>, Dabiao Ma, Jinle Tong, Mengyuan Han, Jian Yang, Hongtao Liu, Haojun Fei, and Qing Yang. "NSR-Boost: A Neuro-Symbolic Residual Boosting Framework for Industrial Legacy Models." Accepted by the ACM SIGKDD Conference on Knowledge Discovery and Data Mining (SIGKDD 2026).'
---

<img src="../images/nsrboost.png" alt="Overview of the NSR-Boost neuro-symbolic residual boosting framework." style="max-width: 100%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">


### ✨ Proposed Method
This paper introduces **NSR-Boost**, a neuro-symbolic residual boosting framework specifically designed to upgrade industrial legacy models in high-concurrency production environments without prohibitive retraining costs. The core advantage of NSR-Boost is its "non-intrusive" nature: it treats the legacy model as a frozen entity and performs targeted repairs exclusively on "hard regions" where predictions fail. The framework operates in three key stages: first, it locates hard regions through residual analysis; second, it generates interpretable experts using a bi-level approach (generating symbolic code structures via an LLM and fine-tuning parameters using Tree-structured Parzen Estimator optimization); finally, it dynamically integrates these experts with the legacy model's output using a lightweight aggregator.

### 📊 Experimental Results
* **Superior Performance:** Experimental results demonstrate that the NSR-Boost framework significantly outperforms state-of-the-art (SOTA) baselines.
* **Cost-Effective & Safe Upgrades:** It successfully avoids the systemic risks and prohibitive retraining costs typically associated with upgrading legacy models in production environments.
* **High Interpretability:** By utilizing an LLM to generate symbolic code structures (such as Python functions) as experts, the framework maintains crucial interpretability and provides actionable feedback constraints for industrial applications.

### 🤝 Collaborating Institutions

Tianjin University; Qfin Holdings, Inc.


<div style="max-width: 80%; margin: 0 auto 30px 0; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 70px; width: auto; margin: 0;">
  <img src="../images/logo/logo_qifu.png" alt="Qfin Holdings, Inc." style="height: 70px; width: auto; margin: 0;">
</div>
