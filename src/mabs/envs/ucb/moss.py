import numpy as np

from .base import UCB


class MossUCB(UCB):
    """
    Implements UCB policy with MOSS value function.
    """

    @property
    def ucb_value(self):
        action_counts = np.array([self.log[arm]["actions"] for arm in self.arms])

        return np.array(self.log.empirical_mu) + (
            4
            / action_counts
            * np.log(np.maximum(1, self.n / (len(self.arms) * action_counts)))
        ) ** (1 / 2)
