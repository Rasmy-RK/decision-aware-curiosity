import numpy as np


class DecisionEnvironment:

    def __init__(self, clue_accuracy=0.80):
        self.clue_accuracy = clue_accuracy
        self.reset()

    def reset(self):
        # Hidden context that actually affects the final decision
        self.relevant_context = np.random.randint(2)

        # Hidden context that does NOT affect the final decision
        self.irrelevant_context = np.random.randint(2)

        self.belief = 0.5
        self.relevant_used = 0
        self.irrelevant_used = 0

        return self.get_state()

    def get_state(self):
        # Convert belief into simple discrete bins
        if self.belief < 0.25:
            belief_bin = 0
        elif self.belief < 0.50:
            belief_bin = 1
        elif self.belief < 0.75:
            belief_bin = 2
        else:
            belief_bin = 3

        return (
            belief_bin,
            self.relevant_used,
            self.irrelevant_used
        )

    def update_belief(self, clue):
        # Bayesian-style belief update

        p = self.belief
        accuracy = self.clue_accuracy

        if clue == self.relevant_context:
            numerator = p * accuracy
            denominator = numerator + (1 - p) * (1 - accuracy)
        else:
            numerator = p * (1 - accuracy)
            denominator = numerator + (1 - p) * accuracy

        if denominator > 0:
            self.belief = numerator / denominator

    def step(self, action):

        # 0 = choose LEFT
        # 1 = choose RIGHT
        # 2 = investigate relevant uncertainty
        # 3 = investigate irrelevant uncertainty

        reward = 0
        done = False
        info = "none"

        if action == 0:

            # If hidden context = 0, LEFT is correct
            if self.relevant_context == 0:
                reward = 10
            else:
                reward = -10

            done = True
            info = "decision"

        elif action == 1:

            # If hidden context = 1, RIGHT is correct
            if self.relevant_context == 1:
                reward = 10
            else:
                reward = -10

            done = True
            info = "decision"

        elif action == 2:

            # Relevant investigation
            reward = -1
            self.relevant_used = 1

            if np.random.random() < self.clue_accuracy:
                clue = self.relevant_context
            else:
                clue = 1 - self.relevant_context

            self.update_belief(clue)

            info = "relevant"

        elif action == 3:

            # Irrelevant investigation
            reward = -1
            self.irrelevant_used = 1

            # This information is deliberately unrelated
            # to the final decision.
            info = "irrelevant"

        return self.get_state(), reward, done, info