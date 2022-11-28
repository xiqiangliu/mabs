from ..base import BaseEnv


class ContextualEnv(BaseEnv):
    """
    Base environment for contextual bandits.
    Implements regret calculation for all policies.
    """

    @property
    def regret(self):
        optimal_rewards = max([arm.deterministic_reward for arm in self.arms]) * self.t
        actual_rewards = sum(
            [self.log[arm]["actions"] * arm.deterministic_reward for arm in self.arms]
        )

        return optimal_rewards - actual_rewards
