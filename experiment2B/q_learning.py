import numpy as np


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.1,
        discount=0.95,
        epsilon=0.15
    ):
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon

        # relevant_known × irrelevant_known × actions
        self.q_table = np.zeros((2, 2, 4))

    def choose_action(self, state, training=True):

        relevant, irrelevant = state

        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)

        return np.argmax(
            self.q_table[relevant, irrelevant]
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        relevant, irrelevant = state
        next_relevant, next_irrelevant = next_state

        old_value = self.q_table[
            relevant,
            irrelevant,
            action
        ]

        if done:
            next_value = 0
        else:
            next_value = np.max(
                self.q_table[
                    next_relevant,
                    next_irrelevant
                ]
            )

        target = reward + self.discount * next_value

        self.q_table[
            relevant,
            irrelevant,
            action
        ] += self.learning_rate * (
            target - old_value
        )