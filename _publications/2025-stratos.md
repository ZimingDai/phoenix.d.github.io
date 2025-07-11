---
title: "Stratos: An End-to-End Distillation Pipeline for Customized LLMs under Distributed Cloud Environments"
collection: publications
permalink: /publications/2025-stratos
category: conferences
date: 2025-03-22
venue: 'NaN'
excerpt: ''
citation: '<strong>Ziming Dai</strong> "Stratos: An End-to-End Distillation Pipeline for Customized LLMs under Distributed Cloud Environments"'
---

In this work, we introduce **Stratos**, a fully automated end-to-end LLM distillation and deployment pipeline tailored for distributed cloud environments. Stratos dynamically selects optimal teacher-student model pairs and matches them with appropriate cloud servers based on user-defined performance and budget constraints.

The system leverages two core strategies—**Knowledge Alignment** and **Knowledge Injection**—to transfer reasoning capabilities from large models to smaller, cost-effective student models. In challenging domains where the teacher model lacks pretraining exposure (e.g., Mahjong reasoning), Stratos employs prompt engineering and synthetic data generation to enable knowledge transfer beyond the teacher’s limitations.

Empirical results show that Stratos significantly improves student model performance, achieving a 4× accuracy gain over GPT-4o on domain-specific tasks. The system has been deployed across a 3,700+ node commercial distributed cloud, demonstrating practical feasibility.

Stratos is envisioned as a versatile toolkit to empower researchers and developers in building customized, efficient, and affordable LLM solutions for vertical applications.
