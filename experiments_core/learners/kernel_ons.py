from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from ..utils import kernel_vector
from .base import BaseLearner
from ._validation import (
    positive_float, positive_int, finite_vector, round_inputs, kernel_parameters, require_finite, nonnegative_quadratic,
)


class KernelONSLearner(BaseLearner):
    """Kernel ONS with projection in the RKHS curvature metric.

    With feature map F and Gram matrix G = F*F, the RKHS curvature is
    A = epsilon I + eta^2 F F*.  The coefficient solve uses
    B = epsilon I + eta^2 G, since A F = F B.

    The Newton direction is eta B^{-1} e_t, computed by Cholesky solves.
    Projection minimizes (c-y)^T G B (c-y) subject to c^T G c <= R^2.
    One representative of the minimizing RKHS element satisfies
    (B + nu I)c = B y. This remains valid when G is singular: coefficient
    vectors differing by an element of ker(G) represent the same function.

    Each projection tridiagonalizes G once in O(t^3), then uses O(t) solves
    for bisection. The stopping criterion bounds the RKHS distance between
    the two bracket endpoints, rather than only their boundary norm error.
    """

    def __init__(
        self,
        action_dim: int,
        radius: float,
        eta_fixed: float,
        gamma_theory: float,
        kons_lr: float,
        epsilon: float,
        kernel_type: str,
        kernel_sigma: float,
        kernel_degree: int,
        kernel_coef0: float,
        kernel_alpha: float,
        kappa: float,
        proj_tol: float = 1e-10,
    ):
        self.action_dim = positive_int(action_dim, "action_dim")
        self.radius = positive_float(radius, "radius")
        self.eta = positive_float(eta_fixed, "eta_fixed")
        # gamma_theory: theoretical gamma derived from problem parameters by
        # config.py.  Stored for logging only; not used in any computation
        # below.  Using gamma_theory in place of gamma_used would also be a
        # valid choice; here we simply use 1/kons_lr.
        self.gamma_theory = positive_float(gamma_theory, "gamma_theory")
        self.kons_lr = positive_float(kons_lr, "kons_lr")
        self.gamma_used = 1.0 / self.kons_lr  # gamma actually used in config epsilon_theory
        self.epsilon = positive_float(epsilon, "epsilon")
        (self.kernel_type, self.kernel_sigma, self.kernel_degree,
         self.kernel_coef0, self.kernel_alpha) = kernel_parameters(
            kernel_type, kernel_sigma, kernel_degree, kernel_coef0, kernel_alpha)
        self.kappa = positive_float(kappa, "kappa")
        self.proj_tol = positive_float(proj_tol, "proj_tol")

        super().__init__(
            name="kons",
            hyperparams={
                "action_dim": self.action_dim,
                "radius": self.radius,
                "eta_fixed": self.eta,
                "gamma_theory": self.gamma_theory,
                "kons_lr": self.kons_lr,
                "gamma_used": self.gamma_used,
                "epsilon": self.epsilon,
                "kernel_type": self.kernel_type,
                "kernel_sigma": self.kernel_sigma,
                "kernel_degree": self.kernel_degree,
                "kernel_coef0": self.kernel_coef0,
                "kernel_alpha": self.kernel_alpha,
                "kappa": self.kappa,
                "proj_tol": self.proj_tol,
            },
        )

        self.context_hist = np.zeros((0, 0), dtype=np.float64)
        self.residual_hist = np.zeros((0, self.action_dim), dtype=np.float64)
        # Gram matrix G_t (growing t*t)
        self._G = np.zeros((0, 0), dtype=np.float64)
        # Cholesky factor of curvature matrix B_t = eta^2 * G_t + eps * I_t
        # L_B satisfies L_B @ L_B.T = B_t
        self._L_B = np.zeros((0, 0), dtype=np.float64)
        # Running sum of diagonal entries for eigenvalue bound on G
        self._sum_diag = 0.0
        # RKHS weight: f = sum_i coeffs[i] phi_i.
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

    def _rkhs_norm_sq(self, c: np.ndarray) -> float:
        """Return ||f_c||_H^2 = c^T G_t c  in O(t^2)."""
        if c.size == 0:
            return 0.0
        product = self._G @ c
        return nonnegative_quadratic(float(c @ product), float(np.abs(c) @ (np.abs(self._G) @ np.abs(c))),
                                     "KONS squared RKHS norm")

    def _rkhs_norm(self, c: np.ndarray) -> float:
        """Return ||f_c||_H = sqrt(c^T G_t c)  in O(t^2)."""
        return math.sqrt(max(0.0, self._rkhs_norm_sq(c)))

    # ------------------------------------------------------------------
    # Cholesky extension of B_t
    # ------------------------------------------------------------------

    def _extend_matrices(self, k_prev: np.ndarray, rho: float) -> None:
        """Extend G_t and the Cholesky factor of B_t by one row/column.  O(t^2)."""
        t = self._G.shape[0]
        k_prev = finite_vector(k_prev, "k_prev", t)
        rho = float(rho)
        if not np.isfinite(rho) or rho < 0.0:
            raise ValueError("rho must be finite and nonnegative")

        # --- Extend Gram matrix ---
        new_G = np.zeros((t + 1, t + 1), dtype=np.float64)
        new_G[:t, :t] = self._G
        new_G[:t, t] = k_prev
        new_G[t, :t] = k_prev
        new_G[t, t] = rho

        # --- Extend Cholesky factor of B_t = eta^2 * G_t + eps * I_t ---
        # New column/row of B corresponds to:
        #   b = eta^2 * k_prev  (length-t vector, off-diagonal block)
        #   beta = eta^2 * rho + epsilon  (diagonal entry)
        b = self.eta ** 2 * k_prev    # length-t vector
        beta = self.eta ** 2 * rho + self.epsilon

        if t == 0:
            if not np.isfinite(beta) or beta <= 0.0:
                raise FloatingPointError("KONS initial curvature is not positive and finite")
            next_L = np.array([[np.sqrt(beta)]], dtype=np.float64)
        else:
            # l satisfies L_B @ l = b (forward substitution)
            l = scipy.linalg.solve_triangular(self._L_B, b, lower=True)
            s = beta - float(l @ l)
            if not np.isfinite(s) or s <= 0.0:
                raise FloatingPointError(
                    f"KONS Cholesky extension lost positive definiteness at size {t + 1}: {s}")
            new_L = np.zeros((t + 1, t + 1), dtype=np.float64)
            new_L[:t, :t] = self._L_B
            new_L[t, :t] = l
            new_L[t, t] = np.sqrt(s)
            next_L = new_L

        require_finite(new_G, "KONS Gram matrix is nonfinite")
        require_finite(next_L, "KONS Cholesky factor is nonfinite")
        self._G = new_G
        self._L_B = next_L
        # Track eigenvalue upper bound for G
        self._sum_diag += rho

    # ------------------------------------------------------------------
    # Gram extension + history management
    # ------------------------------------------------------------------

    def _append_gram(self, context: np.ndarray, residual: np.ndarray) -> None:
        """Extend the Gram matrix by one row/column, updating all caches."""
        k_self = self._kernel_self(context)
        rho = k_self * float(residual @ residual)
        t_prev = self.coeffs.size

        if t_prev == 0:
            self.context_hist = context[None, :].copy()
            self.residual_hist = residual[None, :].copy()
            self._extend_matrices(np.empty(0, dtype=np.float64), rho)
            return

        k_ctx = self._kvec(self.context_hist, context)
        k_prev = k_ctx * (self.residual_hist @ residual)

        self.context_hist = np.vstack([self.context_hist, context[None, :]])
        self.residual_hist = np.vstack([self.residual_hist, residual[None, :]])
        self._extend_matrices(k_prev, rho)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        context = finite_vector(context, "context",
                                self.context_hist.shape[1] if self.coeffs.size else None)
        if self.coeffs.size == 0:
            return np.zeros(self.action_dim, dtype=np.float64)
        k_vec = self._kvec(self.context_hist, context)
        # score(z) = sum_i c_i k(z_i,z) r_i
        return (self.coeffs * k_vec) @ self.residual_hist

    # ------------------------------------------------------------------
    # Projection: FastProj Choice 2 for RKHS ball
    # ------------------------------------------------------------------

    def _fastproj_kernel(self, y_coeff: np.ndarray) -> tuple[np.ndarray, bool]:
        """Project the represented function onto the RKHS ball in the A metric.

        Solve (eta^2 G + (epsilon + nu) I)c = B y.  For the scalar
        multiplier nu, bisection brackets the unique boundary solution.
        In the Gram eigenbasis each represented coordinate varies
        monotonically with nu, so the RKHS distance between endpoint
        functions bounds the distance from either endpoint to the solution.
        """
        y_coeff = finite_vector(y_coeff, "y_coeff", self._G.shape[0])
        norm_y = self._rkhs_norm(y_coeff)
        if norm_y <= self.radius:
            return y_coeff.copy(), False

        t = y_coeff.size
        T, Q = scipy.linalg.hessenberg(self._G, calc_q=True)
        diag = np.diag(T).copy()
        upper = np.diag(T, 1).copy()
        lower = np.diag(T, -1).copy()
        eta_sq = self.eta ** 2
        rhs = self.epsilon * y_coeff + eta_sq * (self._G @ y_coeff)
        q = Q.T @ rhs
        require_finite(q, "KONS projection right-hand side is nonfinite")

        def squared_norm(z):
            product = diag * z
            product[:-1] += upper * z[1:]
            product[1:] += lower * z[:-1]
            absolute_product = np.abs(diag) * np.abs(z)
            absolute_product[:-1] += np.abs(upper) * np.abs(z[1:])
            absolute_product[1:] += np.abs(lower) * np.abs(z[:-1])
            return nonnegative_quadratic(float(z @ product),
                                         float(np.abs(z) @ absolute_product),
                                         "KONS projected squared RKHS norm")

        def solve(nu):
            bands = np.zeros((3, t))
            bands[0, 1:] = eta_sq * upper
            bands[1] = eta_sq * diag + self.epsilon + nu
            bands[2, :-1] = eta_sq * lower
            z = scipy.linalg.solve_banded((1, 1), bands, q)
            require_finite(z, "KONS projected coefficients are nonfinite")
            return z

        low = 0.0
        high = (norm_y / self.radius - 1.0) * (self.epsilon + eta_sq * self._sum_diag)
        if not np.isfinite(high) or high <= 0.0:
            raise FloatingPointError("KONS projection multiplier bracket is invalid")
        z_low = Q.T @ y_coeff
        z_high = solve(high)
        # The trace bound makes high feasible in exact arithmetic. Extend it
        # only to accommodate rounding in that bound or in the linear solve.
        for _ in range(64):
            if squared_norm(z_high) <= self.radius ** 2:
                break
            high *= 2.0
            if not np.isfinite(high):
                raise FloatingPointError("KONS projection bracket overflowed")
            z_high = solve(high)
        else:
            raise FloatingPointError("KONS could not bracket the RKHS projection")

        for _ in range(200):
            if math.sqrt(squared_norm(z_low - z_high)) <= self.proj_tol:
                alpha = Q @ z_high
                norm_alpha = self._rkhs_norm(alpha)
                # Undo only an outward floating-point reconstruction error.
                if norm_alpha > self.radius:
                    alpha *= self.radius / norm_alpha
                return alpha, True
            mid = low + 0.5 * (high - low)
            if mid == low or mid == high:
                break
            z_mid = solve(mid)
            if squared_norm(z_mid) > self.radius ** 2:
                low, z_low = mid, z_mid
            else:
                high, z_high = mid, z_mid
        raise FloatingPointError("KONS projection did not reach its RKHS error tolerance")

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
        context, action_pred, action_expert = round_inputs(
            context, action_pred, action_expert, self.action_dim,
            self.context_hist.shape[1] if self.coeffs.size else None)
        residual = action_pred - action_expert

        t_prev = self.coeffs.size
        self._append_gram(context, residual)
        t_now = t_prev + 1

        # Extend coefficient vector (new entry initialised to 0).
        c_ext = np.zeros(t_now, dtype=np.float64)
        if t_prev > 0:
            c_ext[:t_prev] = self.coeffs

        # Gradient direction: q = B_t^{-1} e_t  (O(t^2) via two triangular solves)
        e_t = np.zeros(t_now, dtype=np.float64)
        e_t[-1] = 1.0
        z = scipy.linalg.solve_triangular(self._L_B, e_t, lower=True)
        direction_raw = scipy.linalg.solve_triangular(self._L_B.T, z, lower=False)
        direction_coeff = self.eta * direction_raw
        y_coeff = c_ext - self.kons_lr * direction_coeff

        require_finite(y_coeff, f"KONS Newton step is nonfinite at round {round_id}")
        coeff_next, did_project = self._fastproj_kernel(y_coeff)
        self.coeffs = coeff_next

        if did_project:
            self.projection_events += 1
        return 0
