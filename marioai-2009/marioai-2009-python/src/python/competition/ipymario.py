import os.path
import pickle

from agents.myagent import *
from tasks.mariotask import MarioTask
from experiments.episodicexperiment import EpisodicExperiment

def load(filename):
    with open(filename, "rb") as f:
        l = pickle.load(f)
        return list(map(lambda x: Individual.from_list(x), l))

def main():
    agent = MyAgent(None)
    filename = "learned_individuals_5_1.pkl"
    trained_individuals = load(filename)
    agent.individual = trained_individuals[0]

    task = MarioTask(agent.name, initMarioMode=2)
    task.env.levelDifficulty = 5
    task.env.levelType = 1

    exp = EpisodicExperiment(task, agent)
    print('Task Ready')
    exp.doEpisodes(1)
    print('mm 2, ld 5, lt 1:', task.reward)

if __name__ == "__main__":
    main()
else:
    print("This is module to be run rather than imported.")
