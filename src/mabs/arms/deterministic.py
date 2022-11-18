from typing import Union

from .base import BaseArm


class DeterministicArm(BaseArm):
    """
    Arm with determinstic reward.
    """

    def __init__(self, reward: Union[float, int], **kwargs):
        """
        Args:
            reward: The deterministic reward to be returned.
        """
        super().__init__(**kwargs)
        self.reward = reward

    def pull(self):
        return self.reward
