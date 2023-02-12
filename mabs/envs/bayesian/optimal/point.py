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


@njit(parallel=True)
def compute_policy(n: int, p2: float) -> np.ndarray:
    w = np.zeros((n + 1, n + 1, 2), dtype=np.float32)
    for t_raw in prange(n):
        for s in prange(n):
                t = n - 1 - t_raw
                w[t, s, 1] = p2 + w[t + 1, s].max()
                w[t, s, 0] = (
                    (s + 1) / (t + 2)
                    + (s + 1) / (t + 2) * w[t + 1, s + 1].max()
                    + (1 - (s + 1) / (t + 2)) * w[t + 1, s].max()
                )
    return w


class BayesianPointOneArmBernoulli(BaseEnv):
    def __init__(self, n: int, **kwargs):
        super().__init__(**kwargs)

        self.n = n
        self.s = 0
        self.q = 0

    def compute_policy(self) -> np.ndarray:
        assert len(self.arms) == 2
        assert isinstance(self.arms[0], BernoulliArm)
        assert isinstance(self.arms[1], DeterministicArm)

        self.policy = compute_policy(self.n, self.arms[1].mean_reward).argmax(axis=-1)
        return self.policy

    def act(self):
        if self.policy is None:
            raise RuntimeError("Must compute policy before acting.")
        action = self.policy[self.t, self.s]
        self.t += 1
        optimal_arm = self.arms[action]
        reward = optimal_arm.pull()
        if optimal_arm == self.arms[0]:
            self.s += reward
            self.q += 1

        self.log.record(arm=optimal_arm, reward=reward)
        return reward
