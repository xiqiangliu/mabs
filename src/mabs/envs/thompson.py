from typing import Union

import numpy as np

from ..arms.base import BaseArm
from .base import BaseEnv


class ThompsonSampling(BaseEnv):
    """
    Environment with Thompson Sampling Policy. Currently supports arms with normal and bernoulli distributions.
    """
    def __init__(self, reward_dist: str, prior_params: list[dict], **kwargs):
        """
        Args:
            reward_dist: Can be either "normal" or "bernoulli"
        """
        super().__init__(**kwargs)
        self.reward_dist = reward_dist
        self.prior_params = prior_params

    def _estimate_mean(self, arm: BaseArm, prior: dict) -> Union[float, np.ndarray]:
        arm_log = self.log[arm]

        if self.reward_dist == "normal":
            mu_prime = (
                (prior["mu"] / prior["sigma"] ** 2)
                + arm_log["rewards"] / arm.sigma**2
            ) / (arm.sigma**-2 + prior["sigma"] ** -2)
            sigma_prime = 1 / (prior["sigma"] ** -2 + arm.sigma**-2)
            return self._rng.normal(loc=mu_prime, scale=sigma_prime)

        elif self.reward_dist == "bernoulli":
            alpha_prime = prior["alpha"] + arm_log["rewards"]
            beta_prime = prior["alpha"] + (arm_log["actions"] - arm_log["rewards"])
            return self._rng.beta(a=alpha_prime, b=beta_prime)
        else:
            raise NotImplementedError

    def act(self):
        samples = np.asarray(
            [
                self._estimate_mean(arm, prior)
                for arm, prior in zip(self.arms, self.prior_params)
            ]
        )
        optimal_arm = self.arms[samples.argmax()]
        reward = optimal_arm.pull()
        self.log.record(arm=optimal_arm, reward=reward)
        return reward

    @property
    def regret(self):
        if self.reward_dist == "bernoulli":
            optimal_reward = (
                np.max([arm.p for arm in self.arms]) * self.log.total_rounds
            )
            policy_reward = sum([arm.p * self.log[arm]["actions"] for arm in self.arms])
            return optimal_reward - policy_reward
        elif self.reward_dist == "normal":
            optimal_reward = (
                np.max([arm.mu for arm in self.arms]) * self.log.total_rounds
            )
            policy_reward = sum(
                [arm.mu * self.log[arm]["actions"] for arm in self.arms]
            )
            return optimal_reward - policy_reward
        else:
            raise NotImplementedError
