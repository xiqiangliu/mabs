import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
combinations = [[0.1, -0.1], [0.1, -0.2], [0.1, 0.2]]
nus = np.linspace(-0.5, 0.5, 100)


def experiment_linucb(a, b, nu, n):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.contextual.LinearArm(nu, a),
            mabs.arms.contextual.LinearArm(nu, b),
        ]
        env = mabs.envs.contextual.LinearUCB(n, 0.1, arms)

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret
    return exp_regret, nu, f"[{a}, {b}]"


def experiment(save_path: Path):
    args = []

    for a, b in combinations:
        args += [(a, b, nu, n) for nu in nus]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_linucb, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(Path(sys.argv[1]).resolve())
