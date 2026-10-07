# Decision-Aware Curiosity for Reinforcement Learning

## Research Question

Can an RL agent distinguish between uncertainty that is useful for future decisions and uncertainty that is irrelevant to its decisions?

## Motivation

Curiosity can encourage an agent to explore uncertain states.

However, not all uncertainty is useful.

An agent may encounter information that is surprising or uncertain but does not help it make better decisions.

This project investigates whether exploration can be guided by the expected decision value of information.

## Hypothesis

A decision-aware curiosity mechanism should:

1. Explore decision-relevant uncertainty more effectively.
2. Reduce unnecessary exploration of decision-irrelevant uncertainty.
3. Maintain or improve external reward.
4. Improve exploration efficiency.

## Experiment

The environment contains two hidden variables.

### Decision-Relevant Uncertainty

A hidden context determines whether LEFT or RIGHT is the correct action.

The agent can investigate this uncertainty before making its decision.

Learning the hidden context can therefore improve future reward.

### Decision-Irrelevant Uncertainty

The agent can also investigate another hidden variable.

This information does not affect the final reward or optimal action.

Therefore, investigating it is unnecessary.

## Compared Methods

### 1. Baseline

Standard Q-learning without intrinsic curiosity.

### 2. Prediction Curiosity

Exploration is encouraged using novelty/prediction-style curiosity.

### 3. Decision-Aware Curiosity

Exploration is rewarded according to the estimated Expected Value of Information (EVI).

Information receives a curiosity bonus when reducing the uncertainty can improve the future decision.

## Main Metrics

- External reward
- Success rate
- Decision-relevant investigations
- Decision-irrelevant investigations
- Exploration efficiency

## Expected Result

The proposed method is expected to investigate decision-relevant uncertainty more selectively while avoiding unnecessary investigation of decision-irrelevant uncertainty.

The experiment is designed to determine whether this hypothesis is supported by empirical results rather than assuming that the proposed method will outperform the baselines.

## Important Research Limitation

The current decision-aware mechanism uses a structural Expected Value of Information calculation based on the experimental environment.

Therefore, this is a proof-of-concept implementation rather than a fully general decision-aware curiosity algorithm.

A future version should learn decision relevance from experience without relying on environment-specific information.

## Technologies

- Python
- NumPy
- Matplotlib
- Reinforcement Learning
- Q-learning
- Partial Observability
- Curiosity-driven Exploration
- Expected Value of Information