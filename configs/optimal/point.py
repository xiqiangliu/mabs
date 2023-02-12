import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import mabs

mp.set_start_method("spawn", force=True)
n = 1000
sim_count = 1000
ps = np.linspace(0, 1, 500)


def experiment_optimal(p, n, rank):
    exp_regret = np.empty(sim_count)
    arms = [
        mabs.arms.BernoulliArm(p),
        mabs.arms.DeterministicArm(0.5),
    ]

    env = mabs.envs.bayesian.BayesianPointOneArmBernoulli(n)
    env.arms = arms
    policy = env.compute_policy()

    for i in tqdm(range(sim_count), position=rank, leave=False, dynamic_ncols=True):
        env = mabs.envs.bayesian.BayesianPointOneArmBernoulli(n)
        env.arms = arms
        env.policy = policy.copy()

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, ""


def launch_experiment(args):
    return experiment_optimal(*args)


def experiment(save_path: Path):
    args = [(p, n, i % mp.cpu_count()) for i, p in enumerate(ps)]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_optimal, args)

    save_path.parent.mkdir(exist_ok=True, parents=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(Path(sys.argv[1]).resolve())
