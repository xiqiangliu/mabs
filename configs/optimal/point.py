import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 10
ps = np.linspace(0, 1, 20)


def experiment_optimal(p, n):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.BernoulliArm(p),
            mabs.arms.DeterministicArm(0.5),
        ]

        env = mabs.envs.bayesian.BayesianPointOneArmBernoulli(n)
        env.arms = arms
        env.compute_policy()

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, "TEST"


def launch_experiment(args):
    return experiment_optimal(*args)


def experiment(save_path: Path):
    args = [(p, n) for p in ps]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_optimal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(Path(sys.argv[1]).resolve())
