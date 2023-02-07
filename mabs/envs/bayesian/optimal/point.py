import logging

import numpy as np

from ....arms import BernoulliArm, DeterministicArm
from ...base import BaseEnv

logger = logging.getLogger(__name__)


def compute_policy(n: int, p2: float) -> np.ndarray:
    w = np.zeros((n + 1, n + 1, n + 1, 2), dtype=np.float32)
    for t in np.arange(n - 1, -1, -1):
        for s in np.arange(n):
            for q in np.arange(n):
                w[t, s, q, 1] = p2 + w[t + 1, s, q].max()
                w[t, s, q, 0] = (
                    (s + 1) / (q + 2)
                    + (s + 1) / (q + 2) * w[t + 1, s + 1, q + 1].max()
                    + (1 - (s + 1) / (q + 2)) * w[t + 1, s, q + 1].max()
                )

    return w.argmax(axis=-1)


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

        self.policy = compute_policy(self.n, self.arms[1].mean_reward)
        return self.policy

    def act(self):
        if self.policy is None:
            raise RuntimeError("Must compute policy before acting.")
        action = self.policy[self.t, self.s, self.q]
        self.t += 1
        optimal_arm = self.arms[action]
        reward = optimal_arm.pull()
        if optimal_arm == self.arms[0]:
            self.s += reward
            self.q += 1

        self.log.record(arm=optimal_arm, reward=reward)
        return reward
