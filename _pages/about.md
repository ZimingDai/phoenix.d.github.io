---
layout: home
permalink: /
title: "Ziming (Phoenix) Dai"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% include base_path %}

<div class="home-page">

<section class="home-hero">
  <p class="home-hero__eyebrow">Federated Learning · Edge Intelligence · LLM Personalization</p>
  <h1>Ziming (Phoenix) Dai</h1>
  <p class="home-hero__role">Ph.D. Student, City University of Hong Kong</p>
  <p class="home-hero__lead">
    I work on personalized and deployable AI systems, with a focus on federated learning, edge intelligence, and large language model adaptation under privacy, latency, and resource constraints.
  </p>
  <div class="home-tag-list">
    <span>Federated Learning</span>
    <span>Edge Intelligence</span>
    <span>LLM Personalization</span>
    <span>Efficient Deployment</span>
  </div>
  <div class="home-hero__actions">
    <a href="{{ '/publications/' | prepend: base_path }}" class="home-button home-button--primary">Publications</a>
    <a href="{{ '/assets/CV_ZimingDai.pdf' | prepend: base_path }}" class="home-button">CV</a>
  </div>
</section>

<section class="home-section">
  <div class="home-section__header">
    <h2>Current Focus</h2>
    <p>
      My current research explores how intelligent models can adapt to local contexts while remaining efficient, private, and practical for real-world edge and distributed environments.
    </p>
  </div>
</section>

<section class="home-section">
  <div class="home-section__header">
    <h2>Experience</h2>
  </div>

  <div class="home-experience-group">
    <h3 class="home-experience-group__title">Academic Experience</h3>
    <div class="home-experience-grid">
      <div class="home-experience-item home-experience-item--education">
        <div class="home-experience-item__mark home-experience-item__mark--logo">
          <img src="{{ '/images/logo/logo_cityu.png' | prepend: base_path }}" alt="City University of Hong Kong">
        </div>
        <div>
          <span class="home-experience-item__date">2026 - Present</span>
          <h3>City University of Hong Kong</h3>
          <p>Ph.D. Student in Data Science, advised by <a href="https://zhouzimu.github.io/">Prof. Zimu Zhou</a>.</p>
        </div>
      </div>

      <div class="home-experience-item home-experience-item--education">
        <div class="home-experience-item__mark home-experience-item__mark--logo">
          <img src="{{ '/images/logo/logo_tju.png' | prepend: base_path }}" alt="Tianjin University">
        </div>
        <div>
          <span class="home-experience-item__date">2023 - 2026</span>
          <h3>Tianjin University</h3>
          <p>M.Eng. in Computer Technology at <a href="http://www.drxiaofei.wang/">Edge Big Bang Lab</a>, advised by <a href="http://qiuchao.fei8s.com/">Prof. Chao Qiu</a> and <a href="https://cic.tju.edu.cn/faculty/wangxiaofei/index.html">Prof. Xiaofei Wang</a>.</p>
        </div>
      </div>

      <div class="home-experience-item home-experience-item--education">
        <div class="home-experience-item__mark home-experience-item__mark--logo">
          <img src="{{ '/images/logo/logo_tju.png' | prepend: base_path }}" alt="Tianjin University">
        </div>
        <div>
          <span class="home-experience-item__date">2019 - 2023</span>
          <h3>Tianjin University</h3>
          <p>B.Eng. in Artificial Intelligence. Awarded Outstanding Undergraduate Thesis.</p>
        </div>
      </div>
    </div>
  </div>

  <div class="home-experience-group">
    <h3 class="home-experience-group__title">Internship Experience</h3>
    <div class="home-experience-grid">
      <div class="home-experience-item home-experience-item--industry">
        <div class="home-experience-item__mark home-experience-item__mark--logo">
          <img src="{{ '/images/logo/logo_360.png' | prepend: base_path }}" alt="Qfin Holdings (360 DigiTech)">
        </div>
        <div>
          <span class="home-experience-item__date">2025 - 2026</span>
          <h3>Qfin Holdings (360 DigiTech)</h3>
          <p>NLP Engineer Intern, working on industrial language models and model enhancement.</p>
        </div>
      </div>

      <div class="home-experience-item home-experience-item--industry">
        <div class="home-experience-item__mark home-experience-item__mark--logo">
          <img src="{{ '/images/logo/logo_wenge.png' | prepend: base_path }}" alt="Beijing Wenge Technology">
        </div>
        <div>
          <span class="home-experience-item__date">2022</span>
          <h3>Beijing Wenge Technology</h3>
          <p>Algorithm Engineer Intern, focusing on named entity recognition for low-resource languages.</p>
        </div>
      </div>
    </div>
  </div>
</section>

<section class="home-section">
  <div class="home-section__header home-section__header--split">
    <div>
      <h2>Selected Publications</h2>
    </div>
    <a href="{{ '/publications/' | prepend: base_path }}" class="home-link">All publications</a>
  </div>

  <div class="home-publication-list">
    {% assign homepage_publications = site.publications | sort: "sort_date" | reverse %}
    {% assign selected_count = 0 %}
    {% for post in homepage_publications %}
      {% if post.rank == "a" and post.venue != "AAAI" and selected_count < 3 %}
        {% assign selected_count = selected_count | plus: 1 %}
        <a class="home-publication home-publication--a" href="{{ post.url | prepend: base_path }}">
          <span class="home-publication__meta">
            <strong>{{ post.rank_label }}</strong>
            <span>{{ post.venue }}</span>
            <span>{{ post.date | date: "%Y" }}</span>
          </span>
          <span class="home-publication__title">{{ post.title }}</span>
        </a>
      {% endif %}
    {% endfor %}
  </div>
</section>

<section class="home-section home-section--connect">
  <h2>Connect</h2>
  <p>
    You can view my <a href="{{ '/assets/CV_ZimingDai.pdf' | prepend: base_path }}">CV</a>, visit my <a href="https://zimingdai.github.io/">blog</a>, or reach out by email.
  </p>
</section>

<section class="home-section home-gallery-section">
  <div class="home-section__header">
    <h2>Gallery</h2>
  </div>

  {% assign gallery_photos = site.static_files | sort: "path" %}
  <div class="home-gallery" aria-label="Photo gallery">
    <div class="home-gallery__track">
      {% for image in gallery_photos %}
        {% assign image_ext = image.extname | downcase %}
        {% if image.path contains '/images/gallery/' %}
          {% if image_ext == '.jpg' or image_ext == '.jpeg' or image_ext == '.png' or image_ext == '.webp' %}
            <figure class="home-gallery__item">
              <img src="{{ image.path | prepend: base_path }}" alt="Gallery photo" loading="lazy">
            </figure>
          {% endif %}
        {% endif %}
      {% endfor %}
      {% for image in gallery_photos %}
        {% assign image_ext = image.extname | downcase %}
        {% if image.path contains '/images/gallery/' %}
          {% if image_ext == '.jpg' or image_ext == '.jpeg' or image_ext == '.png' or image_ext == '.webp' %}
            <figure class="home-gallery__item" aria-hidden="true">
              <img src="{{ image.path | prepend: base_path }}" alt="" loading="lazy">
            </figure>
          {% endif %}
        {% endif %}
      {% endfor %}
    </div>
  </div>
</section>

</div>
