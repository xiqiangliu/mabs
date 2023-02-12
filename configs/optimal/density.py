import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import mabs

n = 1000
sim_count = 200
ps = np.linspace(0, 1, 500)
bernoulli_priors = np.array([[1, 1], [1, 3], [10, 10], [10, 30]])


def experiment_optimal(p, n, alpha, beta, rank):
    exp_regret = np.empty(sim_count)

    arms = [
        mabs.arms.BernoulliArm(p),
        mabs.arms.DeterministicArm(0.5),
    ]

    env = mabs.envs.bayesian.BayesianDensityOneArmBernoulli(alpha, beta, n)
    env.arms = arms
    policy = env.compute_policy()

    for i in tqdm(range(sim_count), position=rank, leave=False, dynamic_ncols=True):
        env = mabs.envs.bayesian.BayesianDensityOneArmBernoulli(alpha, beta, n)
        env.arms = arms
        env.policy = policy.copy()

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, f"[{alpha}, {beta}]"


def launch_experiment(args):
    return experiment_optimal(*args)


def experiment(save_path: Path):
    args = []

    current_i = 0
    for prior in bernoulli_priors:
        for p in ps:
            args.append((p, n, prior[0], prior[1], current_i % mp.cpu_count()))
            current_i += 1

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_optimal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(Path(sys.argv[1]).resolve())
