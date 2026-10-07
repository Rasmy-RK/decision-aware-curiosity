import numpy as np


class DecisionEnvironment:

    def __init__(self, clue_accuracy=0.80):
        self.clue_accuracy = clue_accuracy
        self.reset()

    def reset(self):
        self.relevant_context = np.random.randint(2)
        self.irrelevant_context = np.random.randint(2)

        self.belief = 0.5

        self.relevant_used = 0
        self.irrelevant_used = 0

        return self.get_state()

    def get_state(self):

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

        p = self.belief
        accuracy = self.clue_accuracy

        if clue == self.relevant_context:
            numerator = p * accuracy
            denominator = (
                numerator
                + (1 - p) * (1 - accuracy)
            )
        else:
            numerator = p * (1 - accuracy)
            denominator = (
                numerator
                + (1 - p) * accuracy
            )

        if denominator > 0:
            self.belief = numerator / denominator

    def step(self, action):

        # 0 = LEFT
        # 1 = RIGHT
        # 2 = investigate relevant information
        # 3 = investigate irrelevant information

        reward = 0
        done = False
        info = "none"

        # Final decision
        if action == 0:

            if self.relevant_context == 0:
                reward = 10
            else:
                reward = -10

            done = True
            info = "decision"

        elif action == 1:

            if self.relevant_context == 1:
                reward = 10
            else:
                reward = -10

            done = True
            info = "decision"

        # Investigate relevant uncertainty
        elif action == 2:

            if self.relevant_used == 0:

                reward = -1

                self.relevant_used = 1

                if np.random.random() < self.clue_accuracy:
                    clue = self.relevant_context
                else:
                    clue = 1 - self.relevant_context

                self.update_belief(clue)

                info = "relevant"

            else:

                # Repeated investigation gives no new information
                reward = -2
                info = "wasted_relevant"

        # Investigate irrelevant uncertainty
        elif action == 3:

            if self.irrelevant_used == 0:

                reward = -1

                self.irrelevant_used = 1

                # This information does not affect
                # the final decision.
                info = "irrelevant"

            else:

                # Repeated investigation is wasted
                reward = -2
                info = "wasted_irrelevant"

        return self.get_state(), reward, done, info