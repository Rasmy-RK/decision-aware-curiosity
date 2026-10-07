class CuriosityModule:

    def decision_value(self, state, action):

        relevant, irrelevant = state

        # Relevant information can change
        # the final decision.
        if action == 2 and relevant == 0:
            return 1.0

        # Irrelevant information cannot
        # improve the final decision.
        return 0.0