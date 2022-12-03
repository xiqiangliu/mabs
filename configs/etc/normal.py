import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
mus = np.linspace(0, 1, 100)
ms = [25, 50, 75, 100, "optimal"]


def experiment_normal(mu, m, n):
    m_raw = m
    if m == "optimal":
        m = np.maximum(1, 4 / mu**2 * np.log(n * mu**2 / 4)) if mu > 0 else 1
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.NormalArm(0, 1),
            mabs.arms.NormalArm(mu, 1),
        ]
        env = mabs.envs.simple.ETC(m, n)
        env.arms = arms

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, mu, m_raw


def experiment(save_path: Path):
    args = []

    for mu in mus:
        for m in ms:
            args.append([mu, m, n])
    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_normal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(save_path=Path(sys.argv[1]).resolve())
