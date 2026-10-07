import numpy as np


class QLearningAgent:
    def __init__(
        self,
        learning_rate=0.1,
        discount=0.9,
        epsilon=0.15
    ):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        self.q_table = np.zeros((5, 5, 4))

    def choose_action(self, state):
        row, col = state

        if np.random.random() < self.epsilon:
            return np.random.randint(4)

        return np.argmax(self.q_table[row, col])

    def update(self, state, action, reward, next_state):
        row, col = state
        next_row, next_col = next_state

        old_value = self.q_table[row, col, action]

        next_value = np.max(
            self.q_table[next_row, next_col]
        )

        target = reward + self.discount * next_value

        self.q_table[row, col, action] += (
            self.learning_rate
            * (target - old_value)
        )