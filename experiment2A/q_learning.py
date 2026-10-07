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

        self.q_table = np.zeros((4, 2, 2, 4))

    def choose_action(self, state, training=True):

        belief, relevant, irrelevant = state

        if training and np.random.random() < self.epsilon:
            return np.random.randint(4)

        return np.argmax(
            self.q_table[
                belief,
                relevant,
                irrelevant
            ]
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        belief, relevant, irrelevant = state

        next_belief, next_relevant, next_irrelevant = next_state

        old_value = self.q_table[
            belief,
            relevant,
            irrelevant,
            action
        ]

        if done:
            next_value = 0
        else:
            next_value = np.max(
                self.q_table[
                    next_belief,
                    next_relevant,
                    next_irrelevant
                ]
            )

        target = reward + self.discount * next_value

        self.q_table[
            belief,
            relevant,
            irrelevant,
            action
        ] += self.learning_rate * (
            target - old_value
        )