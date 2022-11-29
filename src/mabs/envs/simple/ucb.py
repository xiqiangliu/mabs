import numpy as np

from .base import SimpleEnv


class UCB(SimpleEnv):
    """
    Environment with UCB-Delta policy.
    """

    def __init__(self, n: int, **kwargs):
        """
        Args:
            m: Number of rounds to explore
            n: Total number of rounds
        """
        super().__init__(**kwargs)
        self.n = n
        self.delta = 1 / self.n**2

    def act(self):
        arm = self.arms[self.ucb_value.argmax()]
        reward = arm.pull()
        self.log.record(arm=arm, reward=reward)
        self.t += 1

        return reward

    @property
    def ucb_value(self):
        return np.array(self.log.empirical_mu) + (
            2 * np.log(1 / self.delta) / len(self.arms)
        ) ** (1 / 2)
