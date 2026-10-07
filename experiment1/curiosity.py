import numpy as np


class CuriosityModule:

    def __init__(self):
        self.visits = np.zeros((5, 5))
        self.transitions = {}

    def reset(self):
        self.visits.fill(0)
        self.transitions.clear()

    # ---------------------------------
    # Baseline: prediction-error style
    # ---------------------------------

    def prediction_error(self, next_state):
        row, col = next_state

        visits = self.visits[row, col]

        curiosity = 1.0 / np.sqrt(visits + 1)

        return curiosity

    # ---------------------------------
    # Decision-aware curiosity
    # ---------------------------------

    def decision_aware(self, state, next_state):

        row, col = next_state

        visits = self.visits[row, col]

        # Epistemic uncertainty:
        # unfamiliar states have higher uncertainty
        uncertainty = 1.0 / np.sqrt(visits + 1)

        # Estimate whether information about
        # this state can affect future decisions.

        state_key = state
        next_key = next_state

        previous = self.transitions.get(
            state_key, {}
        )

        transition_count = previous.get(
            next_key, 0
        )

        total = sum(previous.values()) + 1

        probability = (
            transition_count + 1
        ) / total

        # Low confidence in transition =
        # higher uncertainty.

        transition_uncertainty = 1.0 - probability

        # Decision relevance:
        # uncertainty is more valuable when
        # the transition itself is uncertain.

        decision_relevance = (
            uncertainty * transition_uncertainty
        )

        return decision_relevance

    def update(self, state, next_state):

        row, col = next_state

        self.visits[row, col] += 1

        if state not in self.transitions:
            self.transitions[state] = {}

        if next_state not in self.transitions[state]:
            self.transitions[state][next_state] = 0

        self.transitions[state][next_state] += 1