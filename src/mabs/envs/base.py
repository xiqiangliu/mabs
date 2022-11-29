from abc import ABC, abstractmethod

import numpy as np

from ..arms import BaseArm
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
        self._log = None
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

    @property
    def log(self) -> BanditRecord:
        if self._log is not None:
            return self._log
        raise RuntimeError("The arms have not been set yet!")

    @log.setter
    def log(self, log: BanditRecord):
        self._log = log

    @arms.setter
    def arms(self, arms: list[BaseArm]):
        self._arms = arms

        # Trick to initialize logger
        self.log = BanditRecord()
        for arm in self._arms:
            self.log.actions[arm]

    @abstractmethod
    def act(self) -> float:
        """
        Act for one round. Returns the reward for this round.
        """
        pass

    @property
    @abstractmethod
    def regret(self) -> float:
        """
        Compute the regret at the moment.
        """
        pass
