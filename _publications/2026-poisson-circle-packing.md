---
title: "Single pass Poisson disk sampling via circle packing"
collection: publications
category: manuscripts
permalink: /publications/2026-poisson-circle-packing
date: 2026-04-01
sort_date: 2026-04-01
venue: 'Computers & Graphics'
rank: q2
rank_label: 'JCR Q2'
teaser: 'poisson-circle-packing.jpg'
excerpt: ''
authors: 'Jun Cui, Zeyu Li, Yuxiao Li, Ziheng Guo, <strong>Ziming Dai</strong>, and Jiawan Zhang'
highlight: 'A single-pass Poisson-disk sampling method via circle packing for efficient blue-noise sample generation.'
keywords: ['Poisson-Disk Sampling', 'Circle Packing', 'Blue Noise']
paperurl: 'https://doi.org/10.1016/j.cag.2026.104548'
citation: 'Jun Cui, Zeyu Li, Yuxiao Li, Ziheng Guo, <strong>Ziming Dai</strong>, and Jiawan Zhang. "Single pass Poisson disk sampling via circle packing." Computers & Graphics 135 (2026): 104548.'
---

<img src="../images/poisson-circle-packing.jpg" alt="Pipeline of the proposed Poisson-disk sampling method via circle packing." style="max-width: 100%; display: block; margin: 20px auto; box-shadow: 0 4px 8px rgba(0,0,0,0.1);">

### ✨ Proposed Method
This paper proposes a **spatial covering model based on constrained cells** for Poisson-disk sampling. The model maintains both minimum distance and maximal coverage properties within local cells, then constructs the sample distribution in a single-pass manner. Guided by this geometric model, the method uses circle packing to generate high-quality blue-noise samples efficiently while allowing a controllable trade-off between noise and aliasing.

### 📊 Experimental Results
* **Single-Pass Efficiency:** The method avoids expensive gap tracking in many sampling scenarios and generates high-quality distributions with extreme efficiency.
* **Adaptive Sampling:** The framework extends to arbitrary density functions in linear time, making it suitable for practical adaptive sampling tasks.
* **Application Quality:** Experiments demonstrate competitive blue-noise properties and application results in image stippling and surface remeshing.

### 🤝 Collaborating Institutions
Tianjin University; Communication University of China

<div style="max-width: 80%; margin: 0 auto 30px 0; display: flex; flex-wrap: wrap; justify-content: left; gap: 30px; align-items: center;">
  <img src="../images/logo/logo_tju.png" alt="Tianjin University" style="height: 70px; width: auto; margin: 0;">
</div>
