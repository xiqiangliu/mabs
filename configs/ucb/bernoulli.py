import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
ps = np.linspace(0, 1, 100)


def experiment_bernoulli(p, n, env_cls):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.BernoulliArm(0.5),
            mabs.arms.BernoulliArm(p),
        ]
        env = env_cls(n)
        env.arms = arms

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, env_cls.__name__


def experiment(save_path: Path):
    args = []
    for env in [mabs.envs.ucb.KLUCB, mabs.envs.ucb.MossUCB, mabs.envs.ucb.UCB]:
        args += [(p, n, env) for p in ps]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_bernoulli, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(save_path=Path(sys.argv[1]).resolve())
