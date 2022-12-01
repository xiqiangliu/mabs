from typing import Union

import numpy as np

from ...arms import BaseArm
from ..base import BaseEnv


class ThompsonSampling(BaseEnv):
    """
    Environment with Thompson Sampling Policy. Currently supports arms with normal and bernoulli distributions.
    """

    def __init__(self, reward_dist: str, **kwargs):
        """
        Args:
            reward_dist: Can be either "normal" or "bernoulli"
        """
        super().__init__(**kwargs)
        self.reward_dist = reward_dist

    def _estimate_mean(self, arm: BaseArm, prior: dict) -> Union[float, np.ndarray]:
        """
        Estimate a arm's expected reward.

        Args:
            arm: A bandit arm.
            prior: A dict with prior parameters as keys.
        """
        if not isinstance(arm, BaseArm):
            raise ValueError(f"{arm} is not a bandit arm.")

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
            beta_prime = prior["beta"] + (arm_log["actions"] - arm_log["rewards"])
            return self._rng.beta(a=alpha_prime, b=beta_prime)
        else:
            raise NotImplementedError

    def act(self):
        samples = np.asarray(
            [
                self._estimate_mean(arm, self.prior_params[arm])
                for arm in self.prior_params
            ]
        )
        optimal_arm = self.arms[samples.argmax()]
        reward = optimal_arm.pull()

        self.log.record(arm=optimal_arm, reward=reward)
        self.t += 1

        return reward

    @property
    def prior_params(self):
        if not hasattr(self, "_prior_params"):
            raise ValueError("Prior parameters have not been set for each arm.")
        return self._prior_params

    @prior_params.setter
    def prior_params(self, prior_params: dict[BaseArm, dict]):
        if set(prior_params.keys()) != set(self.arms):
            raise ValueError(
                "Number of prior parameters do not match the number of arms"
            )
        self._prior_params = prior_params
