---
title: 'Handscript of 《Lyapunov-Guided Deep Reinforcement Learning...》'
date: 2022-05-04
permalink: /posts/2022/05/lyapunov-guided/
tags:
  - top papers
---

> Bi, Suzhi, et al. "Lyapunov-guided deep reinforcement learning for stable online computation offloading in mobile-edge computing networks." *IEEE Transactions on Wireless Communications* 20.11 (2021): 7519-7537.

Opportunistic computation offloading is an effective method to improve computational performance in mobile-edge computing (MEC) networks under dynamic edge environments.

In this paper, we consider a **multi-user MEC network** with time-varying wireless channels and randomly arriving task data across continuous time frames.

Specifically, our goal is to design an online computation load algorithm to maximize network processing throughput while ensuring long-term data queue stability and average power constraints. We formulate the problem as a multi-stage stochastic Mixed-Integer Nonlinear Programming (MINLP) problem, which jointly determines binary offloading (i.e., whether each user's task is processed locally or at the edge server) and system resource allocation across time frames.

To address the temporal coupling of decisions, we propose a novel framework called **LyDROO**, which combines Lyapunov optimization and Deep Reinforcement Learning (DRL).

# Background Knowledge

Generally, edge computing systems adopt two common computation models:

1. Binary: The entire dataset of a task is either processed locally at the wireless device or offloaded entirely to the edge server.
2. Partial: The dataset can be partitioned for parallel processing on both the device and the edge server.

## Lyapunov

> See [Lyapunov Optimization: An Introduction](http://cslabcms.nju.edu.cn/problem_solving/images/c/c0/2018-zhao.pdf)

Lyapunov functions are used to control dynamic systems. The system state at a given time is represented by a **multi-dimensional vector**, and the Lyapunov function is a **non-negative scalar representation** of this state. A common approach is to take the **weighted sum of squares** of all state components.

If the system state trends in an undesired direction, the value of the function increases. By minimizing this function over time, we stabilize the system.

### Example

Let the random event in the $$t$$-th time slot be $$w(t) = [w_1(t),w_2(t),...,w_n(t)] \in \Omega^n$$, where each $$w$$ is i.i.d. and $$\Omega$$ is the event space. The control decision vector is $$\alpha(t) = [\alpha_1(t),...,\alpha_n(t)] \in A^m$$, where $$A$$ is the action set.

In the $$t$$-th time slot, the target function is:
$$
p(t) = P(w(t),\alpha(t))
$$

System variables $$y_k(t)$$ affected by the decision are:
$$
y_k(t) = Y_k(w(t),\alpha(t))
$$

The stochastic optimization problem is:
$$
\min_{\alpha(t)\in A^m} \lim_{T\to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[p(t)]
$$

$$
\text{s.t. } \lim_{T\to \infty} \frac{1}{T} \sum_{t=0}^{T-1} \mathbb{E}[y_k(t)] \leq 0
$$

We define virtual queues for each constraint $$y_k(t)$$:
$$
Q_k(t+1) = \max\{Q_k(t) + y_k(t), 0\}
$$

From here, we derive:
$$
\lim_{T \to \infty} \frac{\mathbb{E}[Q_k(T)]}{T} = 0
$$

This becomes the new constraint.

---

The Lyapunov function is defined as:
$$
L(\Theta(t)) = \frac{1}{2} \sum_{k=1}^K Q_k(t)^2
$$

Define the Lyapunov drift as:
$$
\Delta(\Theta(t)) = L(\Theta(t+1)) - L(\Theta(t))
$$

We aim to minimize:
$$
\Delta(\Theta(t)) + V \cdot p(t)
$$

The final expression becomes:
$$
\min_{\alpha(t)\in A^m} \mathbb{E}[B + V \cdot p(t) + \sum_{k=1}^K Q_k(t)y_k(t) \mid \Theta(t)]
$$

# Model & Problem Formulation

<img src="https://raw.githubusercontent.com/ZimingDai/Picture/main/img/20220504125540.png"/>

We consider a scenario where an edge server (ES) helps $$N$$ wireless devices (WDs) in equal time frames $$T$$.

- $$A_i^t$$: Task arrivals to the queue of WD $$i$$ at time $$t$$. Each $$A_i$$ is i.i.d. with $$\mathbb{E}[(A_i^t)^2] = \eta_i$$.
- $$h_i^t$$: Channel gain between WD $$i$$ and the ES. Fixed within a time frame, varies across frames.
- $$D_i^t$$: Bits processed by WD $$i$$ during frame $$t$$.
- $$x_i^t$$: Offloading decision, 1 for ES, 0 for local.

Local processing:
$$
D_{i,L}^t = f_i^t \cdot \frac{T}{\phi}
$$

$$
E_{i,L}^t = \kappa (f_i^t)^2 \cdot T
$$

Where:
- $$f_i^t$$: Local CPU frequency
- $$\phi > 0$$: Number of cycles per bit
- $$\kappa > 0$$: Energy efficiency coefficient

Offloading:
$$
D_{i,O}^t = \frac{W \cdot \tau_i^t T}{v_u} \log_2\left(1 + \frac{E_{i,O}^t h_i^t}{\tau_i^t T N_0}\right)
$$

$$
D_i^t = (1 - x) D_{i,L}^t + x D_{i,O}^t
$$

$$
E_i^t = (1 - x) E_{i,L}^t + x E_{i,O}^t
$$

Queue update:
$$
Q_i(t+1) = \max\{Q_i(t) - \tilde{D}_i^t + A_i^t, 0\}
$$

With:
$$
\tilde{D}_i^t = \min(Q_i(t), D_i^t)
$$

If we assume infinite queue capacity:
$$
Q_i(t+1) = Q_i(t) - D_i^t + A_i^t
$$

**By Little's Law, average delay is proportional to average queue length.**

## Problem Formulation

We aim to maximize the **long-term average weighted sum computation rate** for all WDs under queue stability and average power constraints. Decisions are made online at each time frame, without knowing future channel states or task arrivals.

The stochastic MINLP formulation includes:

- $$c_i$$: Fixed weight of WD $$i$$
- $$r_i^t$$: Processing rate

Constraints:
- (6a) Offloading time ≤ 1
- (6b) Causality: processed data ≤ queued data
- (6c) Average power constraint with threshold $$\gamma_i$$
- (6d) Queue stability

# Lyapunov-based Decomposition

To handle average power constraints (6c), we define a virtual energy queue:
$$
Y_i(t+1) = \max(Y_i(t) + v e_i^t - v \gamma_i, 0)
$$

Combine queues $$Q$$ and $$Y$$ into a Lyapunov function and compute the drift-plus-penalty:
$$
\Lambda(Z(t)) = \Delta L(Z(t)) - V \cdot \sum_{i=1}^N \mathbb{E}[c_i r_i^t \mid Z(t)]
$$

Then:
$$
\Delta L(Q(t)) \leq B_1 + \sum_{i=1}^N Q_i(t) \mathbb{E}[A_i^t - D_i^t \mid Z(t)]
$$

$$
\Delta L(Y(t)) \leq B_2 + \sum_{i=1}^N Y_i(t) \mathbb{E}[e_i^t - \gamma_i \mid Z(t)]
$$

Combine both:
$$
\Delta L(Z(t)) \leq \hat{B} + \sum_{i=1}^N Q_i(t) \mathbb{E}[A_i^t - D_i^t \mid Z(t)] + \sum_{i=1}^N Y_i(t) \mathbb{E}[e_i^t - \gamma_i \mid Z(t)]
$$

Final optimization objective:
$$
\sum_{i=1}^N (Q_i(t) + V c_i) r_i^t - \sum_{i=1}^N Y_i(t) e_i^t
$$

# Lyapunov-Guided DRL

At each time frame, the observation is:
$$
\xi^t = \{h_i^t, Q_i(t), Y_i(t)\}_{i=1}^N
$$

Control decisions:
$$
\{x^t, y^t\}
$$

$$
y^t = \{\tau_i^t, f_i^t, e_{i,O}^t, r_{i,O}^t\}
$$

Even though the objective is non-convex, given a fixed $$x$$, it becomes convex. Thus, we use DRL to learn a policy $$\pi$$ to map $$\xi^t$$ to optimal $$x^t$$.

## Actor

Input: $$\xi^t$$ → output: relaxed offloading decisions $$\hat{x}^t \in [0,1]^N$$ using DNN with parameters $$\theta_t$$. Quantized to $$x_j^t \in \{0,1\}^N$$ using nearest thresholding.

### Critic

Evaluate each $$x_j^t$$ in candidate set $$\Omega_t$$ and select:
$$
x^t = \arg\max_{x_j^t \in \Omega_t} G(x_j^t, \xi^t)
$$

## Policy Update

Update actor policy based on critic evaluation.

## Queue Update

After execution, update system queues $$Q$$ and $$Y$$.
