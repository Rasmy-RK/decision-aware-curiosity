import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("experiment_results.csv")


# --------------------------------
# 1. Average Reward
# --------------------------------

plt.figure(figsize=(10, 6))

for experiment in data["Experiment"].unique():

    subset = data[data["Experiment"] == experiment]

    plt.bar(
        subset["Method"],
        subset["Average Reward"]
    )

    plt.title(experiment + " - Average Reward")
    plt.ylabel("Average Reward")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        experiment.lower().replace(" ", "_") + "_reward.png"
    )

    plt.show()


# --------------------------------
# 2. Success Rate
# --------------------------------

plt.figure(figsize=(10, 6))

for experiment in data["Experiment"].unique():

    subset = data[data["Experiment"] == experiment]

    plt.bar(
        subset["Method"],
        subset["Success Rate"]
    )

    plt.title(experiment + " - Success Rate")
    plt.ylabel("Success Rate (%)")
    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig(
        experiment.lower().replace(" ", "_") + "_success.png"
    )

    plt.show()


# --------------------------------
# 3. Exploration
# --------------------------------

exploration_data = data.dropna(
    subset=["Relevant Probes", "Irrelevant Probes"]
)

for experiment in exploration_data["Experiment"].unique():

    subset = exploration_data[
        exploration_data["Experiment"] == experiment
    ]

    x = range(len(subset))

    plt.figure(figsize=(10, 6))

    plt.bar(
        [i - 0.2 for i in x],
        subset["Relevant Probes"],
        width=0.4,
        label="Relevant"
    )

    plt.bar(
        [i + 0.2 for i in x],
        subset["Irrelevant Probes"],
        width=0.4,
        label="Irrelevant"
    )

    plt.xticks(
        x,
        subset["Method"],
        rotation=20
    )

    plt.ylabel("Number of Probes")
    plt.title(
        experiment + " - Exploration Behavior"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        experiment.lower().replace(" ", "_")
        + "_exploration.png"
    )

    plt.show()


print("All result graphs generated.")