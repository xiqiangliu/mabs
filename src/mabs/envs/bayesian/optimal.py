import numpy as np

from ...arms import BernoulliArm, DeterministicArm
from ..base import BaseEnv


def compute_policy(alpha: int, beta: int, n: int, p2: float) -> np.ndarray:
    def p(s, t):
        return (alpha + s) / (alpha + beta + t - 1)

    omega = np.zeros((n + 1, n, 2))
    for t in np.arange(n, -1, -1):
        for s in np.arange(n, 0, -1):
            omega[t, s, 1] = (n - t + 1) * p2
            omega[t, s, 0] = (
                p(s, t)
                + p(s, t) * omega[t + 1, s + 1].max()
                + (1 - p(s, t)) * omega[t + 1, s].max()
            )

    return omega.argmax(axis=-1)


class BayesianOneArmBernoulli(BaseEnv):
    def __init__(self, alpha: int, beta: int, n: int, **kwargs):
        super().__init__(**kwargs)

        self.alpha = alpha
        self.beta = beta
        self.n = n

        self.t = 0
        self.s = 0

        self.omega = None

    def compute_policy(self) -> np.ndarray:
        assert len(self.arms) == 2
        assert isinstance(self.arms[0], BernoulliArm)
        assert isinstance(self.arms[1], DeterministicArm)

        self.policy = compute_policy(self.alpha, self.beta, self.n, self.arms[1].p)
        return self.policy

    def act(self):
        assert self.policy is not None
        action = self.policy[self.t, self.s]
        self.t += 1
        optimal_arm = self.arms[action]
        if reward := optimal_arm.pull():
            self.s += 1
        self.log.record(arm=optimal_arm, reward=reward)
        return reward

    @property
    def regret(self) -> float:
        optimal_reward = max([arm.p for arm in self.arms]) * self.n
        actual_reward = sum([self.log[arm]["actions"] * arm.p for arm in self.arms])
        return optimal_reward - actual_reward
