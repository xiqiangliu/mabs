import multiprocessing as mp
import pickle
import sys
from pathlib import Path

import numpy as np

import mabs

n = 1000
sim_count = 1000
ps = np.linspace(0, 1, 100)
bernoulli_priors = np.array(
    [[[1, 1], [1, 1]], [[1, 1], [1, 3]], [[10, 10], [10, 10]], [[10, 10], [10, 30]]]
)


def experiment_bernoulli(p, n, prior_a, prior_b):
    exp_regret = np.empty(sim_count)
    for i in range(sim_count):
        arms = [
            mabs.arms.BernoulliArm(0.5),
            mabs.arms.BernoulliArm(p),
        ]
        priors = {
            arms[0]: {"alpha": prior_a[0], "beta": prior_a[1]},
            arms[1]: {"alpha": prior_b[0], "beta": prior_b[1]},
        }
        env = mabs.envs.bayesian.ThompsonSampling("bernoulli")
        env.arms = arms
        env.prior_params = priors

        for _ in range(n):
            env.act()
        exp_regret[i] = env.regret

    return exp_regret, p, f"[{prior_a}, {prior_b}]"


def experiment(save_path: Path):
    args = []
    for prior in bernoulli_priors:
        args += [(p, n, prior[0], prior[1]) for p in ps]

    with mp.Pool(mp.cpu_count()) as p:
        result = p.starmap(experiment_bernoulli, args)

    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":
    experiment(save_path=Path(sys.argv[1]).resolve())
