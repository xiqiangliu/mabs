import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
ps = np.linspace(0, 1, 100)
bernoulli_priors = np.array([[1, 1], [1, 3], [10, 10], [10, 30]])


def experiment_optimal(p, n, alpha, beta):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.BernoulliArm(p),
            mabs.arms.DeterministicArm(0.5),
        ]

        env = mabs.envs.bayesian.BayesianDensityOneArmBernoulli(alpha, beta, n)
        env.arms = arms
        env.compute_policy()

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, f"[{alpha}, {beta}]"


def experiment(save_path: Path):
    args = []
    for prior in bernoulli_priors:
        args += [(p, n, prior[0], prior[1]) for p in ps]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_optimal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(Path(sys.argv[1]).resolve())
