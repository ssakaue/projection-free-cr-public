from __future__ import annotations

import numpy as np
import scipy.linalg

from .base import BaseLearner
from ._validation import (
    positive_float, positive_int, finite_vector, round_inputs, require_finite,
)


def _cholupdate(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Rank-1 update: L s.t. LL^T += xx^T. L is lower-triangular, modified in-place.

    Uses Givens rotations: for each k, apply a rotation G_k in the (k, k+1..n-1)
    plane that annihilates x[k] and updates L[k,k] <- sqrt(L[k,k]^2 + x[k]^2).
    """
    n = len(x)
    x = x.copy()
    for k in range(n):
        r = np.hypot(L[k, k], x[k])
        c = L[k, k] / r   # cos(theta)
        s = x[k] / r       # sin(theta)
        L[k, k] = r
        if k < n - 1:
            tmp = c * L[k + 1:, k] + s * x[k + 1:]
            x[k + 1:] = -s * L[k + 1:, k] + c * x[k + 1:]
            L[k + 1:, k] = tmp
    return L


class ONSLearner(BaseLearner):
    """Online Newton Step with Cholesky factor updates and FastProj.

    The curvature matrix is

        A_t = epsilon * I + sum_{s=1}^{t} grad_s grad_s^T,  grad_s = eta * g_s

    Its Cholesky factor L_t (s.t. L_t L_t^T = A_t) is maintained via rank-1
    Cholesky updates so that each update costs O(d^2).  The Newton step
    A_t^{-1} grad is computed via two triangular solves (also O(d^2)).

    Projection step (FastProj Choice 2)
    ------------------------------------
    When  y = w - lr * A_t^{-1} grad  lies outside the Euclidean ball of
    radius R, we project y onto {w : ||w||_2 <= R} w.r.t. the ||.||_A norm
    using FastProj (Choice 2) from arXiv:2512.23190.

    The algorithm Householder-tridiagonalizes A once per projection, then
    runs bisection with O(d) tridiagonal solves per iteration.
    Runtime: O(d^3 + d * n_bisect) per projection event.
    """

    def __init__(
        self,
        action_dim: int,
        context_dim: int,
        radius: float,
        eta_fixed: float,
        gamma_theory: float,
        ons_lr: float,
        epsilon: float,
        proj_tol: float = 1e-10,
    ):
        self.action_dim = positive_int(action_dim, "action_dim")
        self.context_dim = positive_int(context_dim, "context_dim")
        self.dim = self.action_dim * self.context_dim
        self.radius = positive_float(radius, "radius")
        self.eta = positive_float(eta_fixed, "eta_fixed")
        # gamma_theory: theoretical gamma derived from problem parameters
        # (eta, h_bound) by config.py.  Stored for logging only; not used in
        # any computation below.  Using gamma_theory in place of gamma_used
        # would also be a valid choice; here we simply use 1/ons_lr.
        self.gamma_theory = positive_float(gamma_theory, "gamma_theory")
        self.ons_lr = positive_float(ons_lr, "ons_lr")
        self.gamma_used = 1.0 / self.ons_lr  # gamma actually used in epsilon_theory
        self.epsilon = positive_float(epsilon, "epsilon")
        self.proj_tol = positive_float(proj_tol, "proj_tol")

        w_diameter = 2.0 * self.radius
        # epsilon_theory: reference value for logging; actual epsilon used is
        # self.epsilon (set by config.py, which may differ from this formula).
        self.epsilon_theory = self.dim / (
            w_diameter * w_diameter * self.gamma_used * self.gamma_used
        )

        super().__init__(
            name="ons",
            hyperparams={
                "action_dim": self.action_dim,
                "context_dim": self.context_dim,
                "dim": self.dim,
                "radius": self.radius,
                "eta_fixed": self.eta,
                "gamma_theory": self.gamma_theory,
                "ons_lr": self.ons_lr,
                "gamma_used": self.gamma_used,
                "epsilon_theory": self.epsilon_theory,
                "epsilon": self.epsilon,
                "proj_tol": self.proj_tol,
            },
        )

        # Curvature matrix and its Cholesky factor (L L^T = A)
        self._A = self.epsilon * np.eye(self.dim, dtype=np.float64)
        self._L = np.sqrt(self.epsilon) * np.eye(self.dim, dtype=np.float64)
        self._sum_grad_sq = 0.0  # running sum of ||g||^2 for eigenvalue upper bound

        self.w = np.zeros(self.dim, dtype=np.float64)

    def _matrix_view(self) -> np.ndarray:
        return self.w.reshape(self.action_dim, self.context_dim)

    def predict_scores(self, context: np.ndarray) -> np.ndarray:
        context = finite_vector(context, "context", self.context_dim)
        return self._matrix_view() @ context

    def _fastproj(self, u_vec: np.ndarray) -> np.ndarray:
        """Mahalanobis projection using tridiagonal solves and bisection.

        The Euclidean distance between the feasible and infeasible endpoint
        iterates bounds their distance to the exact projection. Failure to
        reach proj_tol raises an error instead of returning a different
        projection or an infeasible point.
        """
        u_vec = finite_vector(u_vec, "u_vec", self.dim)
        u_norm = float(np.linalg.norm(u_vec))
        if u_norm <= self.radius:
            return u_vec.copy()
        C, Q = scipy.linalg.hessenberg(self._A, calc_q=True)
        q = C @ (Q.T @ u_vec)
        diag = np.diag(C).copy()
        upper = np.diag(C, 1).copy()
        lower = np.diag(C, -1).copy()

        def solve(mu):
            bands = np.zeros((3, self.dim))
            bands[0, 1:] = upper
            bands[1] = diag + mu
            bands[2, :-1] = lower
            x = scipy.linalg.solve_banded((1, 1), bands, q)
            require_finite(x, "ONS projected point is nonfinite")
            return x

        low = 0.0
        high = (u_norm / self.radius - 1.0) * (self.epsilon + self._sum_grad_sq)
        if not np.isfinite(high) or high <= 0.0:
            raise FloatingPointError("ONS projection multiplier bracket is invalid")
        x_low = Q.T @ u_vec
        x_high = solve(high)
        for _ in range(64):
            if np.linalg.norm(x_high) <= self.radius:
                break
            high *= 2.0
            if not np.isfinite(high):
                raise FloatingPointError("ONS projection bracket overflowed")
            x_high = solve(high)
        else:
            raise FloatingPointError("ONS could not bracket the projection")

        for _ in range(200):
            if np.linalg.norm(x_low - x_high) <= self.proj_tol:
                result = Q @ x_high
                result_norm = np.linalg.norm(result)
                if result_norm > self.radius:
                    result *= self.radius / result_norm
                return result
            mid = low + 0.5 * (high - low)
            if mid == low or mid == high:
                break
            x_mid = solve(mid)
            if np.linalg.norm(x_mid) > self.radius:
                low, x_low = mid, x_mid
            else:
                high, x_high = mid, x_mid
        raise FloatingPointError("ONS projection did not reach its Euclidean error tolerance")

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
        grad = self.eta * g

        # Rank-1 update of A: O(d^2)
        self._A += np.outer(grad, grad)
        self._sum_grad_sq += float(np.dot(grad, grad))

        # Rank-1 Cholesky update: L L^T += grad grad^T, O(d^2)
        _cholupdate(self._L, grad)

        # Newton step via two triangular solves: O(d^2)
        z = scipy.linalg.solve_triangular(self._L, grad, lower=True)
        delta = scipy.linalg.solve_triangular(self._L.T, z, lower=False)
        w_tilde = self.w - self.ons_lr * delta
        require_finite(w_tilde, f"ONS Newton step is nonfinite at round {round_id}")

        # FastProj Mahalanobis projection onto Euclidean ball
        w_next = self._fastproj(w_tilde)

        if np.linalg.norm(w_tilde) > self.radius:
            self.projection_events += 1

        self.w = w_next
        return 0
