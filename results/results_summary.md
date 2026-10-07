# Experimental Results Summary

## Research Objective

The experiments investigate whether an RL agent can distinguish between information that is useful for future decisions and information that is merely novel or uncertain.

---

## Experiment 1 — Noisy Partial Grid World

| Method                   | Average Reward | Success Rate |
| ------------------------ | -------------: | -----------: |
| Baseline                 |         -18.34 |       97.00% |
| Prediction Curiosity     |     **-12.17** |   **99.00%** |
| Decision-Aware Curiosity |         -19.68 |       97.00% |

### Observation

Prediction-based curiosity performed better than the decision-aware mechanism in this environment.

The result suggested that the initial decision-aware formulation was insufficient and motivated a more controlled experimental environment.

---

## Experiment 2 — Decision-Relevant vs Decision-Irrelevant Exploration

| Method               | Average Reward | Success Rate | Relevant Probes | Irrelevant Probes |
| -------------------- | -------------: | -----------: | --------------: | ----------------: |
| Baseline             |      **-0.86** |   **49.02%** |           272.6 |             158.9 |
| Prediction Curiosity |          -3.98 |        0.88% |      **1807.5** |             172.7 |
| Decision-Aware       |          -1.49 |       40.98% |           613.7 |             236.7 |

### Observation

Prediction curiosity produced very high exploration but extremely poor task performance.

This demonstrated that increased exploration does not necessarily correspond to useful information-seeking.

---

## Experiment 2A — Cost-Aware Decision-Aware Exploration

| Method         | Average Reward | Success Rate | Relevant Probes | Irrelevant Probes |
| -------------- | -------------: | -----------: | --------------: | ----------------: |
| Baseline       |      **-0.78** |   **49.98%** |           200.0 |             189.9 |
| Decision-Aware |          -1.52 |       48.88% |           400.0 |             290.2 |

### Observation

Introducing an estimated information value and investigation cost did not produce an improvement over the baseline.

The result suggested that simply assigning a higher intrinsic reward to potentially useful information was insufficient.

---

## Experiment 2B — Controlled Decision-Relevance Environment

| Method         | Average Reward | Success Rate | Relevant Probes | Irrelevant Probes |
| -------------- | -------------: | -----------: | --------------: | ----------------: |
| Baseline       |      **-1.20** |   **49.52%** |             700 |               400 |
| Decision-Aware |          -1.83 |       44.87% |             600 |               600 |

### Observation

The decision-aware mechanism did not outperform standard Q-learning.

Although the mechanism changed exploration behavior, this did not translate into improved final decisions.

---

## Overall Finding

Across the experiments, decision-aware intrinsic rewards did not consistently improve external task performance.

The results suggest that effective information-seeking requires more than identifying potentially relevant uncertainty.

An agent must also be able to:

1. Acquire information.
2. Represent the acquired information.
3. Update its belief about the environment.
4. Determine how that information changes future actions.
5. Convert the information into improved decisions.

### Main Research Insight

> Useful exploration requires understanding how acquired information affects future decisions, rather than simply rewarding uncertainty reduction or information acquisition.

---

## Main Limitation

The final controlled experiments use simplified tabular environments. In particular, the current formulation does not provide a sufficiently sophisticated belief representation for integrating acquired information into subsequent action selection.

This limitation motivates future work using belief-state representations, model-based reinforcement learning, and Bayesian approaches.

---

## Reproducibility

Experiments were repeated across multiple random seeds.

Experiment 1 used 5 seeds.

Experiments 2, 2A, and 2B used 10 seeds.

The reported values are averages across the corresponding seeds.
