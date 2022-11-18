from abc import ABC, abstractmethod

import numpy as np

from ..arms.base import BaseArm
from .record import BanditRecord


class BaseEnv(ABC):
    """
    Base class for multi-armed bandit environments. Not meant to be instantiated.
    """

    def __init__(self, seed=None):
        """
        Args:
            seed: Optional. If passed with an integer, it sets the seed for the random number generator of the arm.
        """
        self.log = BanditRecord()
        self._rng = np.random.default_rng(seed=seed)
        self._arms = None

    @property
    def arms(self) -> list[BaseArm]:
        """
        Get a list of arms of this environment.
        """
        if self._arms is None:
            raise ValueError("Arms have not been defined yet")
        return self._arms

    @arms.setter
    def arms(self, arms: list[BaseArm]):
        self._arms = arms

    @abstractmethod
    def act(self):
        """
        Act for one round.
        """
        pass

    @property
    @abstractmethod
    def regret(self) -> float:
        """
        Compute the regret at the moment.
        """
        pass
