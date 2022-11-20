import numpy as np
from numpy.typing import ArrayLike

from ..base import BaseArm


class LinearArm(BaseArm):
    """
    Stochastic linear arm with standard normal noise.
    """

    def __init__(self, theta: ArrayLike, context: ArrayLike, **kwargs):
        """
        Args:
            theta: Environment's theta parameter.
            context: Arm's action vector.
        """
        super().__init__(**kwargs)

        self._theta = np.array(theta)
        self._context = np.array(context)

        if self._theta.shape != self._context.shape:
            raise ValueError(
                f"Arm's action shape {self._context.shape} does not match theta's shape {self._theta.shape}"
            )

        self._compute_reward()

    def pull(self):
        return self.deterministic_reward + self._rng.standard_normal()

    def _compute_reward(self):
        if np.ndim(self._theta) == np.ndim(self._context) == 0:
            self.deterministic_reward = self._theta * self._context
        else:
            self.deterministic_reward = self._theta @ self._context

    @property
    def theta(self):
        """
        Get environment's theta parameter.
        """
        return self._theta

    @theta.setter
    def theta(self, theta: ArrayLike):
        if theta.shape != self.context.shape:
            raise ValueError(
                f"Arm's action shape {self.context.shape} does not match theta's shape {theta.shape}"
            )

        self._theta = theta
        self._compute_reward()

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, context: ArrayLike):
        if context.shape != self.theta.shape:
            raise ValueError(
                f"Arm's action shape {context.shape} does not match theta's shape {self.theta.shape}"
            )

        self._context = context
        self._compute_reward()
