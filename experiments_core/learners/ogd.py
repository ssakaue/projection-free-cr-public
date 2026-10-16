from __future__ import annotations

import math

import numpy as np

from ..utils import euclidean_ball_projection
from .base import BaseLearner
from ._validation import (
    positive_float, positive_int, finite_vector, round_inputs, require_finite,
)


class OGDLearner(BaseLearner):
    """Online Gradient Descent for contextual recommendation (baseline).

    At each round t the weight is updated as:

        w_{t+1} = Pi_R( w_t - eta_t * g_t )

    where  g_t = outer(x_hat_t - x*_t, z_t).reshape(-1)  is the subgradient,
    eta_t is the step size given by ``schedule``, and  Pi_R  is Euclidean
    projection onto the ball of radius R.  Note that OGD projects every round;
    ``projection_events`` is not tracked (it is always 0) because the
    Euclidean projection is O(d) and not the focus of the paper's comparison.

    Parameters
    ----------
    action_dim : int
    context_dim : int
    radius : float
        Radius R of the Euclidean ball domain.
    step0 : float
        Base step size.  Meaning depends on ``schedule``.
    schedule : {"constant", "inverse_sqrt"}
        Step-size schedule.
        - ``"constant"``:      eta_t = step0  (used for the linear setting)
        - ``"inverse_sqrt"``:  eta_t = step0 / sqrt(t)
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        radius: float,
        step0: float,
        schedule: str,
    ):
        self.action_dim = positive_int(action_dim, "action_dim")
        self.context_dim = positive_int(context_dim, "context_dim")
        self.dim = self.action_dim * self.context_dim
        self.radius = positive_float(radius, "radius")
        self.step0 = positive_float(step0, "step0")
        self.schedule = str(schedule)
        if self.schedule not in {"constant", "inverse_sqrt"}:
            raise ValueError("schedule must be constant or inverse_sqrt")

        super().__init__(
            name="ogd",
            hyperparams={
                "action_dim": self.action_dim,
                "context_dim": self.context_dim,
                "dim": self.dim,
                "radius": self.radius,
                "step0": self.step0,
                "schedule": self.schedule,
            },
        )

        self.w = np.zeros(self.dim)

    def _matrix_view(self) -> np.ndarray:
        return self.w.reshape(self.action_dim, self.context_dim)

    def _lr(self, t: int) -> float:
        if self.schedule == "constant":
            return self.step0
        if self.schedule == "inverse_sqrt":
            return self.step0 / math.sqrt(max(1, t))
        raise ValueError(f"Unknown OGD schedule: {self.schedule}")

    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        context = finite_vector(context, "context", self.context_dim)
        return self._matrix_view() @ context

    def update(
        self,
        context: np.ndarray,
        action_pred: np.ndarray,
        action_expert: np.ndarray,
        round_id: int,
    ) -> int:
        context, action_pred, action_expert = round_inputs(
            context, action_pred, action_expert, self.action_dim, self.context_dim)
        residual = action_pred - action_expert
        g = np.outer(residual, context).reshape(-1)
        round_id = positive_int(round_id, "round_id")
        lr = self._lr(round_id)
        candidate = self.w - lr * g
        require_finite(candidate, f"OGD update is nonfinite at round {round_id}")
        require_finite(np.linalg.norm(candidate), f"OGD norm overflowed at round {round_id}")
        self.w = euclidean_ball_projection(candidate, self.radius)
        return 0
