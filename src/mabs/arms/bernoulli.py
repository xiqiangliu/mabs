from .base import BaseArm


class BernoulliArm(BaseArm):
    """
    Arm with bernoulli distribution reward.
    """

    def __init__(self, p: float, **kwargs):
        """
        Args:
            p: Probability of the arm returning a +1 reward. Otherwise, returns 0 reward.
        """
        super().__init__(**kwargs)

        if not 0 <= p <= 1:
            raise ValueError(
                f"Bernoulli arm's probability {p} not in the interval [0, 1]"
            )

        self.p = p

    def pull(self):
        return self._rng.binomial(n=1, p=self.p)
