import numpy as np

from environment import PartialGridWorld
from q_learning import QLearningAgent
from curiosity import CuriosityModule


def evaluate_agent(method, episodes=100):

    env = PartialGridWorld(noise=0.20)

    agent = QLearningAgent()
    curiosity = CuriosityModule()

    total_rewards = []
    successes = 0

    for episode in range(episodes):

        state = env.reset()
        episode_reward = 0

        for step in range(100):

            action = agent.choose_action(state)

            next_state, reward, done = env.step(action)

            if method == "baseline":

                intrinsic = 0

            elif method == "prediction":

                intrinsic = curiosity.prediction_error(
                    next_state
                )

            else:

                intrinsic = curiosity.decision_aware(
                    state,
                    next_state
                )

            agent.update(
                state,
                action,
                reward + 0.5 * intrinsic,
                next_state
            )

            curiosity.update(
                state,
                next_state
            )

            state = next_state
            episode_reward += reward

            if done:
                successes += 1
                break

        total_rewards.append(
            episode_reward
        )

    average_reward = np.mean(
        total_rewards
    )

    success_rate = (
        successes / episodes
    ) * 100

    return average_reward, success_rate


if __name__ == "__main__":

    methods = [
        "baseline",
        "prediction",
        "decision"
    ]

    for method in methods:

        reward, success = evaluate_agent(
            method
        )

        print(
            f"{method}: "
            f"Reward = {reward:.2f}, "
            f"Success = {success:.2f}%"
        )