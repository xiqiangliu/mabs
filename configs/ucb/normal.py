import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
mus = np.linspace(0, 1, 100)


def experiment_normal(mu, n, env_cls):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.NormalArm(0, 1),
            mabs.arms.NormalArm(mu, 1),
        ]
        env = env_cls(n)
        env.arms = arms

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, mu, env_cls.__name__


def experiment(save_path: Path):
    args = []
    for env in [mabs.envs.ucb.AsymptoticUCB, mabs.envs.ucb.MossUCB, mabs.envs.ucb.UCB]:
        args += [(mu, n, env) for mu in mus]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_normal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(save_path=Path(sys.argv[1]).resolve())
