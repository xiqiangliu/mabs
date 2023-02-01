import logging

import numpy as np

from ....arms import BernoulliArm, DeterministicArm
from ...base import BaseEnv

logger = logging.getLogger(__name__)


def compute_policy(n: int, p2: float) -> np.ndarray:
    w = np.zeros((n + 1, n, n, 2))
    for t in np.arange(n - 1, -1, -1):
        for s in np.arange(n - 1, 1, -1):
            for q in np.arange(n - 1, 1, -1):
                w[t - 1, s - 1, q - 1, 1] = p2 + w[t, s, q].max()
                w[t - 1, s - 1, q - 1, 0] = (
                    (s - 1) / (q - 1)
                    + (s - 1) / (q - 1) * w[t, s, q].max()
                    + (1 - (s - 1) / (q - 1)) * w[t, s - 1, q].max()
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
