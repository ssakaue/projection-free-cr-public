"""KernelCholeskyCache: O(t^2) Cholesky maintenance for regularised Gram matrices.

The regularised Gram matrix  H_t = G_t + lambda * I  grows by one row/column
each round.  This module maintains its Cholesky factorisation

    L_t L_t^T = H_t = G_t + lambda * I

so that the linear solve  (G_t + lambda * I)^{-1} b  costs O(t^2).

Bordered extension (O(t^2) update)
------------------------------------
At round t the new regularised Gram matrix has the bordered form

    H_t = [ H_{t-1}       k_prev         ]
          [ k_prev^T      rho + lambda   ]

where  k_prev ∈ R^{t-1}  and  rho ∈ R  is the new diagonal kernel entry.

Cholesky update (pure O(t^2)):
  1. Solve for y:  L_{t-1} y = k_prev   (forward substitution, O(t^2))
  2. Schur complement:  beta^2 = rho + lambda - ||y||^2   (positive if H_t > 0)
  3. New factor:
         L_t = [ L_{t-1}   0    ]
               [ y^T        beta ]

Linear solve (O(t^2)):
    Forward substitution:   L_t u = b
    Backward substitution:  L_t^T x = u

Note: lambda is baked into L_t; the solve method requires no shift argument.
This is the right choice for KernelCoRectron where lambda is fixed, but means
KernelCholeskyCache does not directly handle the varying diagonal shift in
KernelONS projection. That solver uses a tridiagonal reduction instead.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


from ._validation import finite_vector, require_finite


class KernelCholeskyCache:
    """Maintains the Cholesky factor of G_t + lambda * I.

    Parameters
    ----------
    lambda_reg : float
        Regularisation constant baked into the Cholesky factor.
        Must be non-negative.
    """

    def __init__(self, lambda_reg: float = 0.0) -> None:
        self.lambda_reg = float(lambda_reg)
        if not np.isfinite(self.lambda_reg) or self.lambda_reg < 0.0:
            raise ValueError("lambda_reg must be finite and nonnegative")
        self.t: int = 0
        # L: lower-triangular Cholesky factor, shape (t, t).
        # Satisfies  L @ L.T = G_t + lambda_reg * I.
        self.L: np.ndarray = np.empty((0, 0), dtype=np.float64)

    # ------------------------------------------------------------------
    # Bordered Cholesky extension
    # ------------------------------------------------------------------

    def extend(self, k_prev: np.ndarray, rho: float) -> None:
        """Extend the Cholesky factor from H_{t-1} to H_t.

        Parameters
        ----------
        k_prev : np.ndarray, shape (t-1,)
            Cross-column of the new Gram matrix: G_t[:-1, -1].
            Pass an empty array (shape (0,)) for the first call (t: 0 -> 1).
        rho : float
            New diagonal kernel entry: G_t[-1, -1].
        """
        t_prev = self.t
        t_now = t_prev + 1
        k_prev = finite_vector(k_prev, "k_prev", t_prev)
        rho = float(rho)
        if not np.isfinite(rho) or rho < 0.0:
            raise ValueError("rho must be finite and nonnegative")

        if t_prev == 0:
            # Base case: H_1 = [[rho + lambda]],  L_1 = [[beta]].
            val = rho + self.lambda_reg
            if not np.isfinite(val) or val <= 0.0:
                raise np.linalg.LinAlgError("Initial regularized Gram matrix is not positive definite")
            self.L = np.array([[np.sqrt(val)]], dtype=np.float64)
            self.t = t_now
            return

        # --- Step 1: forward substitution  L_{t-1} y = k_prev  (O(t^2)) ---
        y = solve_triangular(self.L, k_prev, lower=True)

        # --- Step 2: Schur complement  beta^2 = rho + lambda - ||y||^2 ---
        schur = rho + self.lambda_reg - float(np.dot(y, y))
        if not np.isfinite(schur) or schur <= 0.0:
            raise np.linalg.LinAlgError(
                f"Gram Cholesky extension lost positive definiteness at size {t_now}: {schur}")
        beta = float(np.sqrt(schur))

        # --- Step 3: assemble new L_t (O(t^2) copy) ---
        new_L = np.zeros((t_now, t_now), dtype=np.float64)
        new_L[:t_prev, :t_prev] = self.L
        new_L[t_prev, :t_prev] = y
        new_L[t_prev, t_prev] = beta
        require_finite(new_L, "Gram Cholesky factor is nonfinite")
        self.L = new_L
        self.t = t_now

    # ------------------------------------------------------------------
    # O(t^2) linear solve
    # ------------------------------------------------------------------

    def solve(self, b: np.ndarray) -> np.ndarray:
        """Return (G_t + lambda * I)^{-1} b via two triangular solves.

        Cost: O(t^2).

        Parameters
        ----------
        b : np.ndarray, shape (t,)
        """
        b = finite_vector(b, "b", self.t)
        # Forward substitution:  L u = b
        u = solve_triangular(self.L, b, lower=True)
        # Backward substitution: L^T x = u
        result = solve_triangular(self.L, u, lower=True, trans="T")
        require_finite(result, "Gram linear solve returned nonfinite values")
        return result
