import os.path
import pickle
import sys
import numpy
import matplotlib.pyplot as plt

from agents.myagent import *
from experiments.episodicexperiment import EpisodicExperiment
from ga.controller import Controller
from tasks.mariotask import MarioTask


class IndividualReward:
    def __init__(self, individual, reward):
        self.individual = individual
        self.reward = reward

    def __str__(self):
        return str(self.reward)


# スコア記録用リスト
max_scores = []  # 各世代の最高スコア
avg_scores = []  # 各世代の平均スコア


def make_next_generation(experiment, individuals):
    n_individuals = len(individuals)
    rewards = []

    for individual in individuals:
        experiment.agent.individual = individual
        experiment.doEpisodes(1)
        rewards.append(IndividualReward(individual, experiment.task.reward))
        print("reward: {0}".format(experiment.task.reward))

    def tournament_selection(pop, k=5):
        selected = []
        for _ in range(len(pop)):
            aspirants = [pop[i] for i in numpy.random.randint(len(pop), size=k)]
            selected.append(max(aspirants, key=lambda ind: ind.reward))
        return selected

    numberOfElites = 5
    sorted_rewards = sorted(rewards, key=lambda individual_reward: individual_reward.reward, reverse=True)
    best_reward = sorted_rewards[0].reward
    avg_reward = sum(r.reward for r in rewards) / len(rewards)
    max_scores.append(best_reward)
    avg_scores.append(avg_reward)

    print(f"Generation best: {best_reward}, average: {avg_reward}")

    elite_individuals = list(map(lambda e: e.individual, sorted_rewards[:numberOfElites]))
    next_individuals = elite_individuals
    tournament_selected = tournament_selection(rewards)

    while len(next_individuals) < n_individuals:
        father, mother = Controller.select(
            list(map(lambda individual_reward: individual_reward.individual, tournament_selected)),
            list(map(lambda individual_reward: individual_reward.reward, tournament_selected))
        )
        child1, child2 = Controller.two_points_cross(father, mother)
        next_individuals.append(child1)
        next_individuals.append(child2)

    next_individuals = next_individuals[:n_individuals]
    mutation_rate = max(0.1, 0.3 - len(next_individuals) * 0.01)
    Controller.mutate(next_individuals, mutation_rate=mutation_rate)

    return next_individuals


def plot_scores(max_scores, avg_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(max_scores, label="Max Score")
    plt.plot(avg_scores, label="Average Score")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title("Score Progression Across Generations")
    plt.legend()
    plt.grid()
    plt.show()


def main():
    agent = MyAgent(None)
    task = MarioTask(agent.name)
    task.env.initMarioMode = 2
    task.env.levelDifficulty = 5
    task.env.levelType = 1
    experiment = EpisodicExperiment(task, agent)

    n_individuals = 50
    filename = "learned_individuals_{0}_{1}.pkl".format(task.env.levelDifficulty, task.env.levelType)
    if os.path.exists(filename):
        initial_individuals = load(filename)
    else:
        initial_individuals = [Individual(random=True) for _ in range(n_individuals)]
    current_individuals = initial_individuals

    n_generations = 30
    for generation in range(n_generations):
        print("generation #{0} playing...".format(generation))
        task.env.visualization = generation % 10 == 0
        current_individuals = make_next_generation(experiment, current_individuals)
        save(current_individuals, filename)

    # 世代ごとのスコア推移をプロット
    plot_scores(max_scores, avg_scores)


def save(individuals, filename):
    l = list(map(lambda x: x.to_list(), individuals))
    with open(filename, "wb") as f:
        pickle.dump(l, f)


def load(filename):
    with open(filename, "rb") as f:
        l = pickle.load(f)
        return list(map(lambda x: Individual.from_list(x), l))


if __name__ == "__main__":
    main()
else:
    print("This is module to be run rather than imported.")
