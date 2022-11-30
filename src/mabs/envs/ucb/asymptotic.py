import numpy as np

from .base import UCB


class AsymptoticUCB(UCB):
    """
    Implements UCB policy with asymptotic value function.
    """

    @property
    def ucb_value(self):
        action_counts = np.array([self.log[arm]["actions"] for arm in self.arms])

        return np.array(self.log.empirical_mu) + (
            2 * np.log(1 + self.t * np.log(self.t) ** 2) / action_counts
        ) ** (1 / 2)
