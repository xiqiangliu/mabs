from collections import defaultdict
from typing import Union

from ..arms import BaseArm


class BanditRecord:
    """
    Log keeper for a multi-armed bandit environment.
    """

    def __init__(self):
        self.total_rounds = 0
        self.total_rewards = 0
        self._actions = defaultdict(lambda: dict(actions=0, rewards=0))

    def record(self, arm: BaseArm, reward: Union[int, float]):
        """
        Record one pull of an arm.

        Args:
            arm: A multi-armed bandit arm instance.
            reward: The reward given by the arm.

        Returns:
            The reward given by the arm.
        """
        if not isinstance(arm, BaseArm):
            raise ValueError(f"The arm {arm} being recorded is not a bandit arm.")

        self.total_rounds += 1
        self.total_rewards += reward
        self._actions[arm]["actions"] += 1
        self._actions[arm]["rewards"] += reward

        return reward

    def __getitem__(self, arm: BaseArm) -> dict:
        if not isinstance(arm, BaseArm):
            raise ValueError(f"Trying to retrieve {arm} that is not a bandit arm.")
        return self._actions[arm]

    @property
    def actions(self):
        """The log dict for all arms in the environment."""

        return self._actions

    @property
    def empirical_mu(self):
        """The empirical expected reward of each arm in the environment."""
        return [
            self.actions[arm]["rewards"] / self.actions[arm]["actions"]
            if self.actions[arm]["actions"] > 0
            else 0
            for arm in self.actions
        ]
