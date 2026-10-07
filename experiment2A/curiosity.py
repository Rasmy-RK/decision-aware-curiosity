class CuriosityModule:

    def expected_value_of_information(self, belief):

        probabilities = [
            0.125,
            0.375,
            0.625,
            0.875
        ]

        p = probabilities[belief]

        left_value = p * 10 + (1 - p) * (-10)
        right_value = p * (-10) + (1 - p) * 10

        current_value = max(
            left_value,
            right_value
        )

        perfect_information_value = 10

        evi = (
            perfect_information_value
            - current_value
        )

        return max(evi, 0)

    def decision_aware_bonus(self, state, action):

        belief, relevant, irrelevant = state

        if action == 2 and relevant == 0:

            evi = self.expected_value_of_information(
                belief
            )

            investigation_cost = 1.0

            net_value = evi - investigation_cost

            return max(net_value, 0) / 10.0

        return 0.0