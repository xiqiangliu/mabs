from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class BaseArm(ABC):
    """
    Base class for all bandit arms. Not meant to be instantiated.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Optional. If passed with an integer, it sets the seed for the random number generator of the arm.
        """
        self._rng = np.random.default_rng(seed=seed)

    @abstractmethod
    def pull(self):
        """
        Pull the arm one time.

        Returns:
            Reward of the arm as a result of this pull.
        """
        pass

    @property
    @abstractmethod
    def mean_reward(self) -> float:
        """Expected reward of the arm"""
        pass
