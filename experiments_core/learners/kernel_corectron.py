"""KernelCoRectron: kernelized projection-free second-order learner.

The RKHS weight is  f = -sum_{s<=t} alpha_s phi_s  where phi_s = K_{z_s} r_s.
The Gram matrix is  G_t[i,j] = k(z_i, z_j) <r_i, r_j>.

Coefficient update via bordered-inverse + differential formula (O(t^2))
-----------------------------------------------------------------------
We want alpha_t = (G_t + lambda*I)^{-1} 1_t.  Instead of re-solving the
full t×t system (O(t^3)), use the Schur complement of G_{t-1}+lambda*I:

    k_prev = G_t[:-1, -1]     (cross-column, known after appending context)
    rho    = G_t[-1, -1]      (diagonal entry)
    s      = (G_{t-1}+lambda*I)^{-1} k_prev    [O(t^2) via KernelCholeskyCache]
    sigma  = rho + lambda - k_prev^T s          [Schur complement]

    alpha_t[-1]  = (1 - k_prev^T alpha_{t-1}) / sigma
    alpha_t[:-1] = alpha_{t-1} - alpha_t[-1] * s

Critical ordering: s must be computed with the OLD (t-1)×(t-1) cache,
BEFORE calling gram_cache.extend().
"""
from __future__ import annotations

import numpy as np

from ..utils import kernel_vector
from .base import BaseLearner
from ._validation import (
    positive_float, positive_int, finite_vector, round_inputs, kernel_parameters, require_finite,
)
from .kernel_cholesky_cache import KernelCholeskyCache


class KernelCoRectronLearner(BaseLearner):
    """Kernelized CoRectron for vector-valued RKHS with k(z,z') I_n.

    Per-round cost: O(t^2) via bordered-inverse coefficient update.
    """

    def __init__(
        self,
        action_dim: int,
        lambda_reg: float,
        kernel_type: str,
        kernel_sigma: float,
        kernel_degree: int = 2,
        kernel_coef0: float = 0.0,
        kernel_alpha: float = 1.0,
    ):
        self.action_dim = positive_int(action_dim, "action_dim")
        self.lambda_reg = positive_float(lambda_reg, "lambda_reg")
        (self.kernel_type, self.kernel_sigma, self.kernel_degree,
         self.kernel_coef0, self.kernel_alpha) = kernel_parameters(
            kernel_type, kernel_sigma, kernel_degree, kernel_coef0, kernel_alpha)

        super().__init__(
            name="corectron_rbf",
            hyperparams={
                "action_dim": self.action_dim,
                "lambda_reg": self.lambda_reg,
                "kernel_type": self.kernel_type,
                "kernel_sigma": self.kernel_sigma,
                "kernel_degree": self.kernel_degree,
                "kernel_coef0": self.kernel_coef0,
                "kernel_alpha": self.kernel_alpha,
            },
        )

        self.context_hist = np.zeros((0, 0), dtype=np.float64)
        self.residual_hist = np.zeros((0, self.action_dim), dtype=np.float64)
        self.gram_cache = KernelCholeskyCache(lambda_reg=self.lambda_reg)
        # alpha_t: coefficients satisfying (G_t + lambda*I) alpha = 1_t.
        self.coeffs: np.ndarray = np.zeros(0, dtype=np.float64)

    # ------------------------------------------------------------------
    # Kernel utilities
    # ------------------------------------------------------------------

    def _kvec(self, x_hist: np.ndarray, x: np.ndarray) -> np.ndarray:
        return kernel_vector(
            x_hist=x_hist, x=x,
            kernel_type=self.kernel_type,
            kernel_sigma=self.kernel_sigma,
            kernel_degree=self.kernel_degree,
            kernel_coef0=self.kernel_coef0,
            kernel_alpha=self.kernel_alpha,
        )

    def _kernel_self(self, context: np.ndarray) -> float:
        """Return k(context, context).

        For the RBF kernel k(z, z) = exp(0) = 1 always, so no computation is
        needed.  For the polynomial kernel k(z, z) = (alpha * ||z||^2 + coef0)^degree
        which can differ from 1 when ||z|| < 1, so we must evaluate it explicitly.
        """
        if self.kernel_type == "rbf":
            return 1.0
        return float(self._kvec(context[None, :], context)[0])

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        context = finite_vector(context, "context",
                                self.context_hist.shape[1] if self.coeffs.size else None)
        if self.coeffs.size == 0:
            return np.zeros(self.action_dim, dtype=np.float64)
        k_vec = self._kvec(self.context_hist, context)
        # score(z) = -sum_i alpha_i k(z_i, z) r_i
        return -((self.coeffs * k_vec) @ self.residual_hist)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(
        self,
        context: np.ndarray,
        action_pred: np.ndarray,
        action_expert: np.ndarray,
        round_id: int,
    ) -> int:
        """Perform one KernelCoRectron update step.

        KernelCoRectron is projection-free: no domain projection is ever
        executed, so ``self.projection_events`` stays 0 throughout the run.

        Returns 0 (no projection events this round).
        """
        context, action_pred, action_expert = round_inputs(
            context, action_pred, action_expert, self.action_dim,
            self.context_hist.shape[1] if self.coeffs.size else None)
        residual = action_pred - action_expert

        k_self = self._kernel_self(context)
        rho = k_self * float(residual @ residual)  # G_t[-1,-1]
        t_prev = self.coeffs.size

        if t_prev == 0:
            # Base case: 1×1 system  (lambda + rho) * alpha = 1.
            self.gram_cache.extend(np.empty(0, dtype=np.float64), rho)
            self.context_hist = context[None, :].copy()
            self.residual_hist = residual[None, :].copy()
            sigma = rho + self.lambda_reg
            self.coeffs = np.array([1.0 / sigma], dtype=np.float64)
            require_finite(self.coeffs, f"KernelCoRectron coefficients are nonfinite at round {round_id}")
            return 0

        # Cross-column: k_prev[i] = k(z_i, z_t) * <r_i, r_t>
        k_ctx = self._kvec(self.context_hist, context)
        k_prev = k_ctx * (self.residual_hist @ residual)  # shape (t_prev,)

        # --- Schur complement (uses OLD t_prev-dimensional cache) ---
        # IMPORTANT: must call gram_cache.solve BEFORE gram_cache.extend.
        s = self.gram_cache.solve(k_prev)                       # O(t^2)
        sigma = rho + self.lambda_reg - float(np.dot(k_prev, s))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise FloatingPointError(
                f"KernelCoRectron lost positive definiteness at round {round_id}: Schur complement={sigma}")

        # Bordered-inverse update of alpha.
        y_new = (1.0 - float(np.dot(k_prev, self.coeffs))) / sigma
        x_old = self.coeffs - y_new * s

        new_coeffs = np.empty(t_prev + 1, dtype=np.float64)
        new_coeffs[:t_prev] = x_old
        new_coeffs[t_prev] = y_new
        require_finite(new_coeffs, f"KernelCoRectron coefficients are nonfinite at round {round_id}")

        # --- Extend Cholesky factor: H_{t-1} -> H_t  (O(t^2)) ---
        self.gram_cache.extend(k_prev, rho)

        # Append history.
        self.context_hist = np.vstack([self.context_hist, context[None, :]])
        self.residual_hist = np.vstack([self.residual_hist, residual[None, :]])
        self.coeffs = new_coeffs

        return 0
