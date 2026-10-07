import numpy as np

from environment import DecisionEnvironment
from q_learning import QLearningAgent
from curiosity import CuriosityModule


TRAINING_EPISODES = 5000
TEST_EPISODES = 1000
STEPS = 4
SEEDS = 10


def train_agent(method, seed):

    np.random.seed(seed)

    env = DecisionEnvironment()
    agent = QLearningAgent()
    curiosity = CuriosityModule()

    for episode in range(TRAINING_EPISODES):

        state = env.reset()

        for step in range(STEPS):

            action = agent.choose_action(
                state,
                training=True
            )

            next_state, reward, done, info = env.step(action)

            intrinsic = 0

            if method == "decision":
                intrinsic = curiosity.decision_value(
                    state,
                    action
                )

            agent.update(
                state,
                action,
                reward + 2.0 * intrinsic,
                next_state,
                done
            )

            state = next_state

            if done:
                break

    return agent


def evaluate_agent(method, seed):

    agent = train_agent(method, seed)
    env = DecisionEnvironment()

    total_reward = 0
    successes = 0
    relevant = 0
    irrelevant = 0

    for episode in range(TEST_EPISODES):

        state = env.reset()

        for step in range(STEPS):

            action = agent.choose_action(
                state,
                training=False
            )

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            if info == "relevant":
                relevant += 1

            if info == "irrelevant":
                irrelevant += 1

            if done and reward == 10:
                successes += 1

            state = next_state

            if done:
                break

    return (
        total_reward / TEST_EPISODES,
        successes / TEST_EPISODES * 100,
        relevant,
        irrelevant
    )


def main():

    print("\nEXPERIMENT 2B - FINAL EVALUATION")
    print("-" * 55)

    for method in ["baseline", "decision"]:

        rewards = []
        success = []
        relevant = []
        irrelevant = []

        for seed in range(SEEDS):

            result = evaluate_agent(
                method,
                seed
            )

            rewards.append(result[0])
            success.append(result[1])
            relevant.append(result[2])
            irrelevant.append(result[3])

        print("\n", method)
        print("Average Reward:", round(np.mean(rewards), 2))
        print("Success Rate:", round(np.mean(success), 2), "%")
        print("Relevant Probes:", round(np.mean(relevant), 2))
        print("Irrelevant Probes:", round(np.mean(irrelevant), 2))


if __name__ == "__main__":
    main()