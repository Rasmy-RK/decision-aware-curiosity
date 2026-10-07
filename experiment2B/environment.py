import numpy as np


class DecisionEnvironment:

    def __init__(self):
        self.reset()

    def reset(self):
        self.relevant_state = np.random.randint(2)
        self.irrelevant_state = np.random.randint(2)

        self.relevant_known = 0
        self.irrelevant_known = 0

        return self.get_state()

    def get_state(self):
        return (
            self.relevant_known,
            self.irrelevant_known
        )

    def step(self, action):

        # 0 = LEFT
        # 1 = RIGHT
        # 2 = investigate relevant information
        # 3 = investigate irrelevant information

        reward = 0
        done = False
        info = "none"

        if action == 0:
            reward = 10 if self.relevant_state == 0 else -10
            done = True
            info = "decision"

        elif action == 1:
            reward = 10 if self.relevant_state == 1 else -10
            done = True
            info = "decision"

        elif action == 2:

            if self.relevant_known == 0:
                self.relevant_known = 1
                reward = -1
                info = "relevant"
            else:
                reward = -2
                info = "wasted_relevant"

        elif action == 3:

            if self.irrelevant_known == 0:
                self.irrelevant_known = 1
                reward = -1
                info = "irrelevant"
            else:
                reward = -2
                info = "wasted_irrelevant"

        return self.get_state(), reward, done, info