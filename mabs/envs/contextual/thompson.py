import numpy as np
from numpy.typing import ArrayLike

from ...arms.contextual import LinearArm
from .base import ContextualEnv


class LinearThompson(ContextualEnv):
    """
    Thompson Sampling for stochastic linear bandits.
    """

    def __init__(
        self, mu: ArrayLike, sigma: ArrayLike, arms: list[LinearArm], **kwargs
    ):
        super().__init__(**kwargs)

        mu = np.atleast_1d(mu)
        sigma = np.atleast_2d(sigma)
        assert mu.ndim <= 1 and sigma.ndim <= mu.ndim + 1

        if mu.size != sigma.shape[0]:
            raise ValueError(
                f"The mean and convariance's dimension for Thompson Sampling must match. Mean has shape {mu.shape} while covariance matrix has shape {sigma.shape}"
            )

        self.mu = mu
        self.sigma = sigma

        self.arms = arms
        self.contexts = np.array([arm.context for arm in self.arms])

    def act(self):
        sampled_theta = self._rng.multivariate_normal(mean=self.mu, cov=self.sigma)
        estimated_reward = self.contexts @ sampled_theta

        selected_arm = self.arms[estimated_reward.argmax()]
        selected_context = selected_arm.context[..., np.newaxis]

        reward = selected_arm.pull()
        self.log.record(arm=selected_arm, reward=reward)

        self.mu = (
            np.linalg.inv(
                np.linalg.inv(self.sigma) + selected_context @ selected_context.T
            )
            @ (np.linalg.inv(self.sigma) @ sampled_theta + reward * selected_context)
        ).ravel()

        self.sigma = np.linalg.inv(
            np.linalg.inv(self.sigma) + selected_context @ selected_context.T
        )

        self.t += 1
        return selected_context.squeeze(), reward
