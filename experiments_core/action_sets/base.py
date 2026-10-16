from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualBounds:
    """Upper bounds on the residual action vector x_hat - x*.

    Used by config.py to compute the gradient norm bound g_bound, which in
    turn determines theoretical hyperparameters (lambda, epsilon, OGD step).

    Attributes
    ----------
    residual_l2_max : float
        Maximum L2 norm of (x_hat - x*) over all pairs of feasible actions.
        For the m-of-n action set this equals sqrt(2 * min(m, n-m)).
    """

    residual_l2_max: float


class ActionSetPlugin(ABC):
    """Abstract plugin defining a feasible action set.

    The runner calls :meth:`select` and :meth:`expert` each round; config.py
    calls :meth:`residual_bounds` once at initialisation to derive hyperparams.

    Distinction between select and expert
    --------------------------------------
    - :meth:`select` is called with the *learner's* predicted scores and
      returns the learner's action x_hat_t (used for regret computation).
    - :meth:`expert` is called with the *true* scores and returns the optimal
      action x*_t (the oracle/expert action revealed after prediction).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def select(self, scores: np.ndarray) -> np.ndarray:
        """Return the learner's action given its predicted score vector."""
        raise NotImplementedError

    @abstractmethod
    def expert(self, true_scores: np.ndarray) -> np.ndarray:
        """Return the optimal (expert) action given the true score vector."""
        raise NotImplementedError

    @abstractmethod
    def residual_bounds(self) -> ResidualBounds:
        """Return upper bounds on ||x_hat - x*|| for this action set."""
        raise NotImplementedError
