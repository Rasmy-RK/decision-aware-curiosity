import numpy as np
import matplotlib.pyplot as plt

from environment import PartialGridWorld
from q_learning import QLearningAgent
from curiosity import CuriosityModule


EPISODES = 500
STEPS = 100
SEEDS = 5


def train(method, seed):

    np.random.seed(seed)

    env = PartialGridWorld(noise=0.20)
    agent = QLearningAgent()
    curiosity = CuriosityModule()

    rewards = []
    exploration = []

    for episode in range(EPISODES):

        state = env.reset()

        total_reward = 0
        total_exploration = 0

        for step in range(STEPS):

            # Observation is used as the state
            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            if method == "baseline":

                intrinsic_reward = 0

            elif method == "prediction":

                intrinsic_reward = (
                    curiosity.prediction_error(
                        next_state
                    )
                )

            else:

                intrinsic_reward = (
                    curiosity.decision_aware(
                        state,
                        next_state
                    )
                )

            learning_reward = (
                reward + 0.5 * intrinsic_reward
            )

            agent.update(
                state,
                action,
                learning_reward,
                next_state
            )

            curiosity.update(
                state,
                next_state
            )

            state = next_state

            total_reward += reward
            total_exploration += intrinsic_reward

            if done:
                break

        rewards.append(total_reward)
        exploration.append(total_exploration)

    return rewards, exploration


def main():

    methods = [
        "baseline",
        "prediction",
        "decision"
    ]

    results = {}

    for method in methods:

        all_rewards = []

        print("Training:", method)

        for seed in range(SEEDS):

            rewards, _ = train(
                method,
                seed
            )

            all_rewards.append(rewards)

        results[method] = np.mean(
            all_rewards,
            axis=0
        )

    # -------------------------
    # Reward comparison
    # -------------------------

    plt.figure(figsize=(10, 5))

    for method in methods:

        plt.plot(
            results[method],
            label=method
        )

    plt.xlabel("Episode")
    plt.ylabel("External Reward")
    plt.title(
        "Decision-Aware Curiosity vs Baselines"
    )

    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()