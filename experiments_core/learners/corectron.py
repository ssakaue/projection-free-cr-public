"""CoRectron: projection-free second-order learner for contextual recommendation.

Algorithm
---------
At each round t the learner maintains weight w_t ∈ R^d  (d = action_dim *
context_dim).  Prediction: score(z) = reshape(w_t) @ z.

After observing the expert action x*_t the gradient is

    g_t = outer(x_hat_t - x*_t, z_t).reshape(-1)

and the accumulated curvature matrix is

    A_t = lambda * I + sum_{s=1}^{t} g_s g_s^T.

The next weight is  w_{t+1} = -A_t^{-1} zeta_t  where
zeta_t = sum_{s=1}^{t} g_s.

O(d^2) per-round implementation via Sherman-Morrison
-----------------------------------------------------
Instead of solving the d×d system from scratch (O(d^3)), we maintain
A_t^{-1} directly and update it via the Sherman-Morrison rank-one formula:

    A_t^{-1} = A_{t-1}^{-1} - h h^T / (1 + mu)

where  h = A_{t-1}^{-1} g  and  mu = g^T h.

The weight update uses the same h without a second solve:

    w_{t+1} = w_t - h * (1 + nu) / (1 + mu)

where  nu = g^T w_t.

Derivation of the weight update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We want w_{t+1} = -A_t^{-1} zeta_t where zeta_t = zeta_{t-1} + g.

    -A_t^{-1} zeta_t
        = -(A_{t-1}^{-1} - h h^T/(1+mu)) (zeta_{t-1} + g)
        = -A_{t-1}^{-1} zeta_{t-1}        # = w_t
          + h h^T zeta_{t-1}/(1+mu)       # via SMW
          - A_{t-1}^{-1} g                # = -h
          + h h^T g/(1+mu)               # = h * mu/(1+mu)

    = w_t  + h * (h^T zeta_{t-1}) / (1+mu)  -  h  +  h * mu/(1+mu)

Now substitute h^T zeta_{t-1} = -nu (since zeta_{t-1} = -A_{t-1} w_t and h = A_{t-1}^{-1} g):

    = w_t  + h * (-nu) / (1+mu)  -  h  +  h * mu/(1+mu)
    = w_t  - h * [nu/(1+mu) + 1 - mu/(1+mu)]
    = w_t  - h * [(nu + 1 + mu - mu) / (1+mu)]
    = w_t  - h * (1 + nu) / (1 + mu)

Final update formula:

    w_{t+1} = w_t - h * (1 + nu) / (1 + mu)

where  h = A_{t-1}^{-1} g,  mu = g^T h,  nu = g^T w_t.

State
-----
a_inv : np.ndarray, shape (d, d)
    Maintains A_t^{-1} = (lambda*I + sum_{s<=t} g_s g_s^T)^{-1}.
    Updated via SMW at the END of each update(), AFTER computing h.
w : np.ndarray, shape (d,)
    Pre-computed NEXT-round weight w_{t+1} = -A_t^{-1} zeta_t.
    Initialised to w_1 = 0.  Updated at the END of each update() call
    so that predict_scores() at round t+1 is a pure O(d) matvec.
"""
from __future__ import annotations

import numpy as np

from .base import BaseLearner
from ._validation import (
    positive_float, positive_int, finite_vector, round_inputs, require_finite,
)


class CoRectronLearner(BaseLearner):
    """CoRectron: projection-free second-order learner.

    Per-round cost: O(d^2)  via Sherman-Morrison on A_t^{-1}.

    Parameters
    ----------
    action_dim : int
    context_dim : int
    lambda_reg : float
        Regularisation for A_0 = lambda * I.
    """

    def __init__(self, action_dim: int, context_dim: int, lambda_reg: float):
        self.action_dim = positive_int(action_dim, "action_dim")
        self.context_dim = positive_int(context_dim, "context_dim")
        self.dim = self.action_dim * self.context_dim
        self.lambda_reg = positive_float(lambda_reg, "lambda_reg")

        super().__init__(
            name="corectron",
            hyperparams={
                "action_dim": self.action_dim,
                "context_dim": self.context_dim,
                "dim": self.dim,
                "lambda_reg": self.lambda_reg,
            },
        )

        # A_t^{-1} maintained explicitly: A_0^{-1} = (1/lambda) * I.
        self.a_inv: np.ndarray = (1.0 / self.lambda_reg) * np.eye(
            self.dim, dtype=np.float64
        )
        require_finite(self.a_inv, "CoRectron regularization is too small for a finite inverse")
        # w_1 = -A_0^{-1} * 0 = 0.
        self.w: np.ndarray = np.zeros(self.dim, dtype=np.float64)

    def _matrix_view(self) -> np.ndarray:
        return self.w.reshape(self.action_dim, self.context_dim)

    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        # self.w holds w_t = -A_{t-1}^{-1} zeta_{t-1} (pre-computed last round).
        context = finite_vector(context, "context", self.context_dim)
        return self._matrix_view() @ context

    def update(
        self,
        context: np.ndarray,
        action_pred: np.ndarray,
        action_expert: np.ndarray,
        round_id: int,
    ) -> int:
        """Perform one CoRectron update step.

        CoRectron is projection-free: no domain projection is ever executed,
        so ``self.projection_events`` stays 0 throughout the run.

        Returns 0 (no projection events this round).
        """
        context, action_pred, action_expert = round_inputs(
            context, action_pred, action_expert, self.action_dim, self.context_dim)
        residual = action_pred - action_expert
        g = np.outer(residual, context).reshape(-1)

        # h = A_{t-1}^{-1} g  (O(d^2) matvec with the stored inverse)
        h = self.a_inv @ g
        mu = float(np.dot(g, h))     # g^T A_{t-1}^{-1} g  >= 0
        nu = float(np.dot(g, self.w))  # g^T w_t

        tolerance = 64 * np.finfo(float).eps * np.linalg.norm(g) * np.linalg.norm(h)
        denom = 1.0 + mu
        if not np.isfinite(denom) or mu < -tolerance or denom <= 0.0:
            raise FloatingPointError(
                f"CoRectron lost positive definiteness at round {round_id}: "
                f"g^T A_inv g={mu}, denominator={denom}")
        next_w = self.w - h * ((1.0 + nu) / denom)
        next_inverse = self.a_inv - np.outer(h, h) / denom
        require_finite(next_w, f"CoRectron weight is nonfinite at round {round_id}")
        require_finite(next_inverse, f"CoRectron inverse is nonfinite at round {round_id}")
        self.w = next_w
        self.a_inv = next_inverse

        return 0
