---
title: "Multi-Granularity Federated Learning by Graph-Partitioning"
collection: publications
category: manuscripts
permalink: /publications/2024-gpmgfl
date: 2024-11-11
excerpt: ''
venue: 'IEEE Transactions on Cloud Computing'
paperurl: 'https://doi.org/10.1109/TCC.2024.3494765'
citation: '<strong>Ziming Dai</strong>, Yunfeng Zhao, Chao Qiu, Xiaofei Wang, Haipeng Yao, and Dusit Niyato. "Multi-Granularity Federated Learning by Graph-Partitioning." IEEE Transactions on Cloud Computing 13, no. 1 (2024): 18-33.'
---

<img src="../images/gpmgfl.png" alt="The procedure of graph-partitioning multi-granularity FL on consortium blockchain." style="max-width: 90%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

### ✨ Proposed Method
This paper proposes a **Graph-Partitioning Multi-Granularity Federated Learning (GP-MGFL)** method built on a consortium blockchain. To reduce overall communication overhead, the framework uses a balanced graph partitioning algorithm to group edge clients, which minimizes high-cost communications while ensuring effective intra-group guidance. Furthermore, the system introduces a cross-granularity guidance mechanism where fine-granularity models guide coarse-granularity models to fully leverage data heterogeneity and enhance accuracy. To maintain security, a dynamic credit model is implemented to adjust clients' contributions to the global model and automatically select group leaders for model aggregation.

### 📊 Experimental Results
* **Accuracy Enhancement:** The GP-MGFL algorithm achieves an accuracy that is 5.6% higher than that of ordinary blockchain-based federated learning (BFL) algorithms.
* **Superior Grouping:** Compared to other grouping methods, such as greedy grouping, the proposed GP-MGFL approach improves accuracy by approximately 1.5%.
* **Robust Security:** In scenarios involving malicious clients, the method demonstrates strong robustness, achieving a maximum accuracy improvement of 11.1% over baseline models.

### 🤝 Collaborating Institutions
Tianjin University; Guangming Laboratory of Artificial Intelligence and Digital Economy (SZ); Beijing University of Posts and Telecommunications; Nanyang Technological University

<div style="max-width: 80%; margin: 0 auto; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 50px; width: auto; margin: 0;">
  <img src="../images/logo/logo_gml.png" alt="Guangming Laboratory" style="height: 50px; width: auto; margin: 0;">
  <img src="../images/logo/logo_bupt.png" alt="Guangming Laboratory" style="height: 50px; width: auto; margin: 0;">
  <img src="../images/logo/logo_ntu.png" alt="Tianjin University of Finance" style="height: 50px; width: auto; margin: 0;">
</div>