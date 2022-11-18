from .base import BaseArm


class NormalArm(BaseArm):
    """Arm with normal distribution reward."""

    def __init__(self, mu: float, sigma: float, **kwargs):
        """
        Args:
            mu: Mean of normal distribution.
            sigma: Standard deviation of normal distribution.
        """
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def pull(self):
        return self._rng.normal(loc=self.mu, scale=self.sigma)
