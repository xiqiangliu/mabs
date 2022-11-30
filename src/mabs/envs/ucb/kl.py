import numpy as np
from numpy.typing import ArrayLike
from scipy.special import rel_entr

from .base import UCB


def kl_bernoulli(p: ArrayLike, q: ArrayLike):
    assert p.shape == q.shape
    return rel_entr(p, q) + rel_entr(1 - p, 1 - q)


class KLUCB(UCB):
    """
    Implements the KL-UCB policy.
    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            kind: Arms' distribution in str. Only supports 'bernoulli' now.
        """

        super().__init__(*args, **kwargs)
        kind = "bernoulli" if "kind" not in kwargs else "bernoulli"
        if kind == "bernoulli":
            self.kl_div = kl_bernoulli

    @property
    def ucb_value(self):
        action_counts = np.array([self.log[arm]["actions"] for arm in self.arms])
        if (action_counts == 0).any():
            return np.full_like(action_counts, np.inf)

        bound = np.log(1 + self.t * np.log(np.log(self.t))) / action_counts
        p = np.array(self.log.empirical_mu)

        l = p
        r = np.ones_like(l)

        for _ in range(10):
            q = (l + r) / 2
            if np.allclose(l, r):
                break

            kl = self.kl_div(p, q)

            mask = kl < bound

            l[mask] = q[mask]
            r[~mask] = q[~mask]

        return q
