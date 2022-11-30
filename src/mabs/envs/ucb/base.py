import numpy as np

from ...arms import BaseArm
from ..base import BaseEnv
from ..record import BanditRecord


class BaseUCBEnv(BaseEnv):
    """
    Base environment for UCB policy variants.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t = 0

    @property
    def regret(self):
        optimal_rewards = max([arm.mean_reward for arm in self.arms]) * self.t
        actual_rewards = sum(
            [self.log[arm]["actions"] * arm.mean_reward for arm in self.arms]
        )

        return optimal_rewards - actual_rewards


class UCB(BaseUCBEnv):
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
        self._finish_init = False

    def act(self):
        if not self._finish_init:
            for arm in self.arms:
                if self.log[arm]["actions"] == 0:
                    selected_arm = arm
                    break
            else:
                selected_arm = self.arms[self.ucb_value.argmax()]
                self._finish_init = True
        else:
            selected_arm = self.arms[self.ucb_value.argmax()]
        reward = selected_arm.pull()
        self.log.record(arm=selected_arm, reward=reward)
        self.t += 1

        return reward

    @property
    def arms(self) -> list[BaseArm]:
        return self._arms

    @arms.setter
    def arms(self, arms: list[BaseArm]):
        self._arms = arms

        # Trick to initialize logger
        self.log = BanditRecord()
        for arm in self._arms:
            self.log.actions[arm]

        self._finish_init = False

    @property
    def ucb_value(self):
        action_counts = np.array([self.log[arm]["actions"] for arm in self.arms])
        return np.array(self.log.empirical_mu) + np.sqrt(
            2 * np.log(1 / self.delta) / action_counts
        )
