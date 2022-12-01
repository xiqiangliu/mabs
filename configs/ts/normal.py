import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
mus = np.linspace(0, 1, 100)
normal_priors = np.array(
    [
        [[0, 1], [0, 1]],
        [[0, 1], [0.5, 1]],
        [[0.5, 1], [0, 1]],
        [[0, 0.1], [0.5, 0.1]],
        [[0.5, 0.1], [0, 0.1]],
    ]
)


def experiment_normal(mu, n, prior_a, prior_b):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.NormalArm(0, 1),
            mabs.arms.NormalArm(mu, 1),
        ]
        env = mabs.envs.ThompsonSampling("normal")
        env.arms = arms
        env.prior_params = {
            arms[0]: {"mu": prior_a[0], "sigma": prior_a[1] ** 0.5},
            arms[1]: {"mu": prior_b[0], "sigma": prior_b[1] ** 0.5},
        }

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, mu, f"[{prior_a}, {prior_b}]"


def experiment(save_path: Path):
    args = []
    for prior in normal_priors:
        args += [(mu, n, prior[0], prior[1]) for mu in mus]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_normal, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(save_path=Path(sys.argv[1]).resolve())
