import numpy as np


class CuriosityModule:

    def __init__(self):
        self.visits = np.zeros(4)

    def reset(self):
        self.visits.fill(0)

    def prediction_error(self, state):

        belief, relevant, irrelevant = state

        index = (
            belief
            + 4 * relevant
            + 8 * irrelevant
        )

        novelty = 1.0 / np.sqrt(
            self.visits[belief] + 1
        )

        self.visits[belief] += 1

        return novelty

    def expected_value_of_information(self, belief):

        # Convert belief bin into representative probability

        probabilities = [
            0.125,
            0.375,
            0.625,
            0.875
        ]

        p = probabilities[belief]

        # Current best expected reward
        left_value = p * 10 + (1 - p) * (-10)
        right_value = p * (-10) + (1 - p) * 10

        current_value = max(
            left_value,
            right_value
        )

        # Value if hidden context became perfectly known
        perfect_information_value = 10

        evi = (
            perfect_information_value
            - current_value
        )

        return max(evi, 0)

    def decision_aware_bonus(self, state, action):

        belief, relevant, irrelevant = state

        # Action 2 = investigate relevant uncertainty
        if action == 2:

            evi = self.expected_value_of_information(
                belief
            )

            return evi / 10.0

        # Investigating irrelevant uncertainty
        # should receive no decision-aware bonus.
        return 0.0