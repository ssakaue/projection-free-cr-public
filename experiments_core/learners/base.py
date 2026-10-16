from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseLearner(ABC):
    """Abstract base class for all online learners.

    Subclasses must implement :meth:`predict_scores` and :meth:`update`.

    Attributes
    ----------
    name : str
        Human-readable algorithm name, used as a label in output DataFrames.
    hyperparams : dict
        Method metadata for inspection, not used for computation. Experiment
        settings are saved separately in resolved_config.yaml.
    projection_events : int
        Cumulative number of rounds in which a nontrivial Mahalanobis projection was triggered.
        Subclasses increment this counter inside :meth:`update` whenever such a
        projection step is executed.  Projection-free learners (e.g. CoRectron)
        never increment it, so it remains 0 throughout the run.
    """

    def __init__(self, name: str, hyperparams: dict[str, Any]):
        self.name = name
        self.hyperparams = hyperparams
        # Counts rounds where domain projection fired.  Stays 0 for
        # projection-free learners (CoRectron, KernelCoRectron).
        self.projection_events = 0

    @abstractmethod
    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        """Return a score vector of shape (action_dim,) for the given context.

        The caller selects the top-m actions by argsort; higher score = more
        preferred action.
        """
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        context: np.ndarray,
        action_pred: np.ndarray,
        action_expert: np.ndarray,
        round_id: int,
    ) -> int:
        """Update internal state after observing the expert's action.

        Parameters
        ----------
        context : np.ndarray, shape (context_dim,)
        action_pred : np.ndarray, shape (action_dim,)
            The learner's own prediction from this round (scaled indicator vector).
        action_expert : np.ndarray, shape (action_dim,)
            The expert's action revealed at the end of the round (scaled indicator vector).
        round_id : int
            1-indexed round number (used by some learners for step-size schedules).

        Returns
        -------
        int
            Always 0. Projection counts are tracked through the cumulative
            ``self.projection_events`` attribute, which excludes OGD's
            Euclidean projections.
        """
        raise NotImplementedError
