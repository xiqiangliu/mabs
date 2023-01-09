import numpy as np

from ...arms import BaseArm
from .base import SimpleEnv


class ETC(SimpleEnv):
    """
    Environment with Thompson Sampling Policy. Currently supports arms with normal and bernoulli distributions.
    """

    def __init__(self, m: int, n: int, **kwargs):
        """
        Args:
            m: Number of rounds to explore
            n: Total number of rounds
        """
        super().__init__(**kwargs)

        assert m < n

        self.m = m
        self.n = n

    def act(self):
        if self.t < self.m:
            arm: BaseArm = self.arms[self.t % len(self.arms)]
        else:
            arm = self.arms[np.argmax(self.log.empirical_mu)]

        reward = arm.pull()
        self.log.record(arm=arm, reward=reward)
        self.t += 1

        return reward
