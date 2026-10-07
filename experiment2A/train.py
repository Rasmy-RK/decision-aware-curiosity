import numpy as np
import matplotlib.pyplot as plt

from environment import DecisionEnvironment
from q_learning import QLearningAgent
from curiosity import CuriosityModule


EPISODES = 3000
SEEDS = 10
STEPS = 4


def train(method, seed):

    np.random.seed(seed)

    env = DecisionEnvironment()
    agent = QLearningAgent()
    curiosity = CuriosityModule()

    rewards = []
    relevant_exploration = []
    irrelevant_exploration = []

    for episode in range(EPISODES):

        state = env.reset()

        total_reward = 0
        relevant_count = 0
        irrelevant_count = 0

        for step in range(STEPS):

            action = agent.choose_action(
                state,
                training=True
            )

            next_state, reward, done, info = env.step(
                action
            )

            intrinsic_reward = 0

            if method == "decision":

                intrinsic_reward = (
                    curiosity.decision_aware_bonus(
                        state,
                        action
                    )
                )

            learning_reward = (
                reward
                + 2.0 * intrinsic_reward
            )

            agent.update(
                state,
                action,
                learning_reward,
                next_state,
                done
            )

            if info == "relevant":
                relevant_count += 1

            if info == "irrelevant":
                irrelevant_count += 1

            total_reward += reward

            state = next_state

            if done:
                break

        rewards.append(total_reward)
        relevant_exploration.append(relevant_count)
        irrelevant_exploration.append(irrelevant_count)

    return (
        rewards,
        relevant_exploration,
        irrelevant_exploration
    )


def main():

    methods = [
        "baseline",
        "decision"
    ]

    results = {}

    for method in methods:

        print("Training:", method)

        all_rewards = []
        all_relevant = []
        all_irrelevant = []

        for seed in range(SEEDS):

            rewards, relevant, irrelevant = train(
                method,
                seed
            )

            all_rewards.append(rewards)
            all_relevant.append(relevant)
            all_irrelevant.append(irrelevant)

        results[method] = {
            "reward": np.mean(
                all_rewards,
                axis=0
            ),

            "relevant": np.mean(
                all_relevant,
                axis=0
            ),

            "irrelevant": np.mean(
                all_irrelevant,
                axis=0
            )
        }

    plt.figure(figsize=(10, 5))

    for method in methods:

        plt.plot(
            results[method]["reward"],
            label=method
        )

    plt.xlabel("Episode")
    plt.ylabel("External Reward")
    plt.title(
        "Experiment 2A: Decision-Aware Curiosity"
    )

    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))

    for method in methods:

        plt.plot(
            results[method]["relevant"],
            label=method + " - relevant"
        )

        plt.plot(
            results[method]["irrelevant"],
            linestyle="--",
            label=method + " - irrelevant"
        )

    plt.xlabel("Episode")
    plt.ylabel("Investigation Count")

    plt.title(
        "Decision-Relevant vs Decision-Irrelevant Exploration"
    )

    plt.legend()
    plt.grid()
    plt.show()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()