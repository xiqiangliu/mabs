import logging

import numpy as np

try:
    from numba import njit, prange
except ImportError:
    prange = range

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


from ....arms import BernoulliArm, DeterministicArm
from ...base import BaseEnv

logger = logging.getLogger(__name__)


# @njit(parallel=True)
# def compute_policy(alpha: int, beta: int, n: int, p2: float):
#     def p(s, t):
#         return (alpha + s) / (alpha + beta + t)

#     w = np.zeros((n + 1, n + 1, 2), dtype=np.float32)
#     for t_raw in prange(n):
#         for s in prange(n):
#             t = n - 1 - t_raw
#             w[t, s, 1] = p2 + w[t + 1, s].max()
#             w[t, s, 0] = (
#                 p(s, t)
#                 + p(s, t) * w[t + 1, s + 1].max()
#                 + (1 - p(s, t)) * w[t + 1, s].max()
#             )

#     return w


def compute_policy(alpha: int, beta: int, n: int, p2: float) -> np.ndarray:
    w = np.zeros(n + 1, dtype=np.float32)
    s_star = np.zeros(n, dtype=np.float32)
    for t_raw in range(n - 1):
        t = n - t_raw
        s = np.arange(t)
        temp2 = w[:-1] + p2
        p = (alpha + s) / (alpha + beta + t)
        temp1 = p + w[:-1] * (1 - p) + w[1:] * p
        w = np.zeros_like(p, dtype=np.float32)
        idx = temp1 < temp2
        w[idx] = temp2[idx]
        s_star[t - 1] = np.max(s[idx]) if idx.any() else -1
        idx = ~idx
        w[idx] = temp1[idx]
    s_star[0] = -1
    return s_star


class BayesianDensityOneArmBernoulli(BaseEnv):
    def __init__(self, alpha: int, beta: int, n: int, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.n = n
        self.s = 0
        self.omega = None

    def compute_policy(self) -> np.ndarray:
        assert len(self.arms) == 2
        assert isinstance(self.arms[0], BernoulliArm)
        assert isinstance(self.arms[1], DeterministicArm)

        self.policy = compute_policy(
            self.alpha, self.beta, self.n, self.arms[1].mean_reward
        )
        return self.policy

    def act(self):
        if self.policy is None:
            raise RuntimeError("Must compute policy before acting.")

        optimal_arm = self.arms[
            int(self.log.actions[self.arms[0]]["rewards"] <= self.policy[self.t])
        ]
        reward = optimal_arm.pull()
        self.t += 1

        self.log.record(arm=optimal_arm, reward=reward)
        return reward
