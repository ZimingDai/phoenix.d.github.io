---
title: "TS-PEFT: Unveiling Token-Level Redundancy in Parameter-Efficient Fine-Tuning"
collection: publications
permalink: /publications/2026-tspeft
category: conferences
date: 2026-01-29
sort_date: 2026-08-15
venue: 'IJCAI-ECAI'
rank: a
rank_label: 'CCF-A'
teaser: 'tspeft.png'
excerpt: ''
authors: 'Dabiao Ma<sup>*</sup>, <strong>Ziming Dai</strong><sup>*</sup>, Zhimin Xin, Shu Wang, Jian Yang, and Haojun Fei'
highlight: 'A token-level sparsity perspective for reducing redundant updates in parameter-efficient fine-tuning.'
keywords: ['PEFT', 'Token Sparsity', 'LLM Fine-Tuning']
citation: '<u>Dabiao Ma</u>, <u><strong>Ziming Dai</strong></u>, Zhimin Xin, Shu Wang, Jian Yang, and Haojun Fei. "TS-PEFT: Unveiling Token-Level Redundancy in Parameter-Efficient Fine-Tuning." In the 35th International Joint Conference on Artificial Intelligence. 2026.'
---

<img src="../images/tspeft.png" alt="Comparison between standard PEFT and our TS-PEFT framework." style="max-width: 100%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
### ✨ Proposed Method
This paper introduces **TS-PEFT**, a theoretical framework utilizing proximal optimization that acts as a dynamic probe to identify token-level redundancy during the fine-tuning process of large models. Current Parameter-Efficient Fine-Tuning (PEFT) methods typically operate under the assumption that every token passing through a selected target module contributes equally and requires a parameter update. TS-PEFT challenges this convention by dynamically identifying and removing unnecessary token updates to optimize the adaptation mechanism.

### 📊 Experimental Results
* **Efficiency and Performance:** By discarding 30% to 70% of token updates, TS-PEFT consistently matches or exceeds the performance of dense baselines such as LoRA and DoRA.
* **Noise Reduction:** Extensive experiments demonstrate that indiscriminately updating all tokens is not only computationally superfluous but often introduces optimization noise.
* **Module Importance Indicator:** In-depth analysis shows that the learned token-level sparsity is a superior indicator of module importance compared to traditional weight criteria, providing a novel data-driven perspective on large models.

### 🤝 Collaborating Institutions

Tianjin University; Qfin Holdings, Inc.


<div style="max-width: 80%; margin: 0 auto 30px 0; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 70px; width: auto; margin: 0;">
  <img src="../images/logo/logo_qifu.png" alt="Qfin Holdings, Inc." style="height: 70px; width: auto; margin: 0;">
</div>
