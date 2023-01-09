import numpy as np

from ...arms.contextual import LinearArm
from .base import ContextualEnv


class LinearUCB(ContextualEnv):
    """
    UCB for stochastic linear bandits.
    """

    def __init__(self, n: int, lamb: float, arms: list[LinearArm], **kwargs):
        super().__init__(**kwargs)

        self.n = n
        self.delta = 1 / self.n
        self.d = arms[0].context.shape[0]
        self.lamb = lamb

        self.V = self.lamb * np.eye(self.d)
        self.theta = np.zeros(self.d)
        self.empirical_mean = np.zeros_like(self.theta)

        self.arms = arms
        self.contexts = np.array([arm.context for arm in self.arms])

    def act(self):
        beta: float = self.lamb**0.5 + (
            2 * np.log(self.n) + self.d * np.log(1 + self.t / (self.delta * self.d))
        )

        estimated_reward = self.contexts @ self.theta + beta * np.diag(
            self.contexts @ np.linalg.inv(self.V) @ self.contexts.T
        )

        assert estimated_reward.ndim <= 1
        selected_arm_idx: int = estimated_reward.argmax(axis=0)

        selected_arm = self.arms[selected_arm_idx]
        selected_context = self.contexts[selected_arm_idx][..., None]
        reward = selected_arm.pull()

        self.V += selected_context @ selected_context.T
        self.empirical_mean += reward * selected_context.squeeze()
        self.theta = np.linalg.inv(self.V) @ self.empirical_mean

        self.log.record(selected_arm, reward)
        self.t += 1

        return selected_context.squeeze(), reward
