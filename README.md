# Decision-Aware Curiosity for Reinforcement Learning

## Overview

This project investigates whether an RL agent can distinguish between **uncertainty that is useful for future decisions** and uncertainty that is merely novel or informative.

Standard curiosity-driven reinforcement learning encourages an agent to explore unfamiliar states. However, not all information is equally valuable. An agent may spend substantial effort exploring information that does not improve its future decisions.

This project studies whether exploration can instead be guided by the **decision relevance of information**.

## Research Question

> Can an RL agent distinguish between uncertainty that is worth exploring and uncertainty that should be ignored?

A more specific question is:

> When is information worth acquiring because it can improve future decisions?

## Motivation

Curiosity is an important mechanism for exploration in uncertain environments. Prediction-error-based curiosity can encourage agents to seek novel experiences, but novelty does not necessarily imply usefulness.

This creates an important distinction:

**Novel information ≠ useful information**

An agent should ideally explore when reducing uncertainty can improve its future decisions while avoiding exploration that provides little decision benefit.

This problem is closely related to research on reinforcement learning, Bayesian inference, uncertainty, exploration, and decision making.

## Hypothesis

Exploration guided by the expected decision benefit of information should reduce unnecessary exploration and improve learning efficiency compared with curiosity based only on novelty or prediction error.

## Approach

The project uses small controlled environments to isolate different forms of uncertainty.

The main components are:

* Q-learning as the reinforcement learning framework
* Prediction-error/novelty-based exploration as a baseline
* Decision-aware intrinsic rewards
* Relevant and irrelevant information sources
* Multiple random seeds for evaluation
* External reward and success rate as performance measures
* Exploration counts to measure information-seeking behavior

## Experimental Progression

### Experiment 1 — Noisy Partial Grid World

The first experiment compared:

1. Standard Q-learning
2. Prediction-error curiosity
3. Decision-aware curiosity

The environment contained observation noise and obstacles.

The result did not show an advantage for the proposed decision-aware mechanism.

Prediction curiosity performed better in this environment.

This established an initial negative result and motivated a more controlled environment.

### Experiment 2 — Decision-Relevant vs Decision-Irrelevant Information

The second experiment introduced two types of investigation:

* **Relevant information:** could help determine the correct final action.
* **Irrelevant information:** did not improve the final decision.

Prediction curiosity explored extensively, but this did not translate into better performance.

This demonstrated that high exploration does not necessarily produce useful information-seeking behavior.

### Experiment 2A — Cost-Aware Information Value

The next experiment incorporated an estimated **Expected Value of Information (EVI)** and an investigation cost.

The goal was to reward investigation only when the expected information benefit justified its cost.

However, the decision-aware method still did not outperform the baseline.

### Experiment 2B — Controlled Decision-Relevance Environment

The final experiment further isolated relevant and irrelevant investigation.

The decision-aware mechanism was evaluated against standard Q-learning using multiple random seeds.

The final evaluation produced:

| Method         | Average Reward | Success Rate | Relevant Probes | Irrelevant Probes |
| -------------- | -------------: | -----------: | --------------: | ----------------: |
| Baseline       |          -1.20 |       49.52% |             700 |               400 |
| Decision-aware |          -1.83 |       44.87% |             600 |               600 |

The proposed mechanism therefore **did not outperform the baseline**.

## Results

The experiments did not support the original hypothesis.

Instead, they revealed an important limitation:

> Assigning intrinsic value to decision-relevant information alone is insufficient to produce effective information-seeking behavior.

In particular, an agent needs not only to recognize potentially useful information, but also to **represent that information and incorporate it into subsequent decisions**.

This distinction became increasingly clear across the experiments.

## Key Observation

The experiments highlight three different concepts:

```text
Novel information
       ↓
Information that reduces uncertainty
       ↓
Information that changes a decision
       ↓
Information that improves the final outcome
```

These are not necessarily equivalent.

A curiosity mechanism may successfully encourage exploration without improving the agent's actual decisions.

## Limitations

Several limitations remain.

### 1. Simplified environments

The experiments use small synthetic environments rather than complex real-world tasks.

### 2. Limited state representation

The current agents use tabular Q-learning, which restricts the complexity of the representations that can be learned.

### 3. Information integration

The experiments reveal that obtaining information and using information are separate problems. The current mechanism does not provide a sufficiently sophisticated representation of acquired information.

### 4. Environment-specific decision value

The current decision-aware mechanism uses simplified assumptions about which information is relevant. A more general mechanism should estimate decision value from the agent's learned model rather than manually specifying relevance.

## Future Work

Future versions could investigate:

* Bayesian belief-state representations
* POMDP-based environments
* Learned models of environmental uncertainty
* Expected Value of Information based on changes in future action value
* Information-theoretic intrinsic rewards
* Model-based reinforcement learning
* Uncertainty-aware arbitration between exploration and exploitation
* More complex environments with changing dynamics

A particularly interesting direction is to connect information acquisition directly to **changes in predicted future action values**, rather than rewarding information simply because it reduces uncertainty.

## Relevance to Reinforcement Learning Research

The project explores the intersection of:

* Reinforcement learning
* Curiosity-driven exploration
* Bayesian inference
* Uncertainty estimation
* Information-seeking behavior
* Decision making

The investigation is motivated by the broader question of how an intelligent agent should decide **what information is worth seeking** when learning and acting in uncertain environments.

## Conclusion

This project did not produce a universally superior curiosity algorithm.

Instead, the experiments demonstrate a more fundamental result:

> **Useful exploration requires more than curiosity about uncertainty; an agent must understand how acquired information affects future decisions.**

The negative experimental results therefore motivate a future shift from simple intrinsic-reward design toward integrated **belief representation, information valuation, and decision making**.

This provides a foundation for further investigation into flexible learning and uncertainty-driven behavior in reinforcement learning systems.
