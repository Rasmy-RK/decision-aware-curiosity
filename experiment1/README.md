# Decision-Aware Curiosity for Reinforcement Learning

## Research Question

Can an RL agent distinguish uncertainty that is useful for future decisions from uncertainty that is merely unpredictable?

## Motivation

Curiosity can encourage reinforcement learning agents to explore unfamiliar states.

However, not all uncertainty is useful.

An agent may waste exploration on unpredictable observations even when learning about them does not improve its future decisions.

This project investigates a decision-aware exploration mechanism.

## Methods

Three approaches are compared:

1. Standard Q-learning
2. Q-learning with prediction-error curiosity
3. Q-learning with decision-aware curiosity

## Environment

A partially observable grid world is used.

The environment contains:

- Hidden true states
- Noisy observations
- Obstacles
- A goal state

## Hypothesis

Decision-aware curiosity should reduce unnecessary exploration while maintaining or improving learning efficiency in uncertain environments.

## Evaluation

The methods are evaluated using:

- External reward
- Success rate
- Exploration behaviour
- Robustness to observation noise
- Learning efficiency

## Technologies

- Python
- NumPy
- Matplotlib
- Reinforcement Learning
- Q-learning
- Curiosity-driven exploration
- Partial observability