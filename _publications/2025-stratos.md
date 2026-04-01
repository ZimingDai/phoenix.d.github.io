---
title: "Stratos: An End-to-End Distillation Pipeline for Customized LLMs under Distributed Cloud Environments"
collection: publications
permalink: /publications/2025-stratos
category: conferences
date: 2025-03-22
venue: 'AAAI'
excerpt: ''
citation: '<strong>Ziming Dai</strong>, Tuo Zhang, Fei Gao, Xingyi Cai, Xiaofei Wang, Cheng Zhang, Wenyu Wang, and Chengjie Zang. "Stratos: An End-to-End Distillation Pipeline for Customized LLMs Under Distributed Cloud Environments." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 47, pp. 40506-40512. 2026.'
---

<img src="../images/stratos.png" alt="System overview of the proposed end-to-end distillation pipeline." style="max-width: 90%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

### ✨ Proposed Method
This paper introduces **Stratos**, an end-to-end Large Language Model (LLM) distillation pipeline designed to automate server/model selection, knowledge distillation, and deployment in distributed cloud environments. To optimize cloud hosting under user-defined budget and latency constraints, Stratos automatically selects Pareto-optimal servers and dynamically matches teacher-student model pairs. Furthermore, it features an adaptive dual-mode knowledge distillation framework that intelligently switches between "Knowledge Alignment" and "Knowledge Injection" based on the task's complexity and the teacher model's domain coverage.

### 📊 Experimental Results
* **Exceptional Domain Adaptation:** On a rare, domain-specific Mahjong reasoning task, the Stratos-distilled student model achieves four times (4x) the accuracy of its GPT-4o teacher baseline.
* **Optimal Resource Efficiency:** By successfully balancing accuracy, latency, training time, and cost, Stratos improves the overall deployment score by at least 22% compared to traditional single-objective (e.g., accuracy-first or cost-first) selection methods.
* **Strong Generalization:** The pipeline demonstrates significant reasoning enhancements and robust generalization capabilities across complex mathematical reasoning benchmarks, such as GSM8K and AIME.

### 🤝 Collaborating Institutions
Tianjin University; University of Southern California; Tianjin University of Finance and Economics; Paiou Cloud Computing (Shanghai) Co., Ltd

<div style="max-width: 80%; margin: 0 auto 30px 0; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 70px; width: auto; margin: 0;">
  <img src="../images/logo/logo_usc.png" alt="Guangming Laboratory" style="height: 70px; width: auto; margin: 0;">
  <img src="../images/logo/logo_tjufe.png" alt="Guangming Laboratory" style="height: 70px; width: auto; margin: 0;">
  <img src="../images/logo/logo_ppio.png" alt="Tianjin University of Finance" style="height: 70px; width: auto; margin: 0;">
</div>

