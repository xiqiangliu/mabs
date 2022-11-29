from ..base import BaseEnv


class SimpleEnv(BaseEnv):
    """
    Base environment for simple multi-armed bandit policies.
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
