"""Unit tests for KernelCholeskyCache.

Tests verify:
  1. Correctness of the Cholesky factor: L @ L.T == G_t + lambda * I
  2. Correctness of the solve: (G_t + lambda * I)^{-1} b
  3. Edge cases: t=1 base case, lambda=0 (exact Cholesky of G_t)
  4. Sequential bordered extensions simulating KernelCoRectron
  5. Agreement of Schur-complement coefficient update with direct solve
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.learners.kernel_cholesky_cache import KernelCholeskyCache

RNG = np.random.default_rng(7)
ATOL = 1e-10


def _make_gram(t: int, rng: np.random.Generator) -> np.ndarray:
    """Random positive-definite Gram matrix via G = A A^T / t."""
    A = rng.standard_normal((t, t))
    return (A @ A.T) / t + np.eye(t) * 0.1  # ensure PD


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestKernelCholeskyCache(unittest.TestCase):

    def test_base_case_t1(self) -> None:
        """t=1: L = [[sqrt(rho + lambda)]]."""
        lam = 2.0
        rho = 5.0
        cache = KernelCholeskyCache(lambda_reg=lam)
        cache.extend(np.empty(0), rho)
        expected_L = np.array([[np.sqrt(rho + lam)]])
        np.testing.assert_allclose(cache.L, expected_L, atol=1e-15)

    def test_cholesky_invariant_small(self) -> None:
        """L L^T == G_t + lambda * I for 20 random G of size t=10."""
        lam = 1.5
        for _ in range(20):
            G = _make_gram(10, RNG)
            cache = KernelCholeskyCache(lambda_reg=lam)
            for t in range(10):
                k_prev = G[:t, t]
                rho = G[t, t]
                cache.extend(k_prev, rho)

            expected = G + lam * np.eye(10)
            np.testing.assert_allclose(
                cache.L @ cache.L.T, expected, atol=ATOL,
                err_msg="L L^T != G + lambda*I",
            )

    def test_solve_correctness_small(self) -> None:
        """(G_t + lambda*I)^{-1} b matches numpy.linalg.solve."""
        lam = 2.0
        for _ in range(20):
            t = 8
            G = _make_gram(t, RNG)
            cache = KernelCholeskyCache(lambda_reg=lam)
            for s in range(t):
                cache.extend(G[:s, s], G[s, s])

            b = RNG.standard_normal(t)
            x_chol = cache.solve(b)
            x_ref = np.linalg.solve(G + lam * np.eye(t), b)
            np.testing.assert_allclose(x_chol, x_ref, atol=ATOL,
                                       err_msg="solve disagrees with linalg.solve")

    def test_solve_correctness_t100(self) -> None:
        """Correctness at KernelCoRectron production size t=100."""
        lam = 1.0
        for _ in range(5):
            t = 100
            G = _make_gram(t, RNG)
            cache = KernelCholeskyCache(lambda_reg=lam)
            for s in range(t):
                cache.extend(G[:s, s], G[s, s])

            b = RNG.standard_normal(t)
            x_chol = cache.solve(b)
            x_ref = np.linalg.solve(G + lam * np.eye(t), b)
            np.testing.assert_allclose(x_chol, x_ref, atol=1e-8,
                                       err_msg="solve disagrees at t=100")

    def test_lambda_zero(self) -> None:
        """lambda=0: L L^T == G_t exactly (positive definite G required)."""
        for _ in range(10):
            t = 6
            G = _make_gram(t, RNG)
            cache = KernelCholeskyCache(lambda_reg=0.0)
            for s in range(t):
                cache.extend(G[:s, s], G[s, s])
            np.testing.assert_allclose(
                cache.L @ cache.L.T, G, atol=ATOL,
                err_msg="L L^T != G with lambda=0",
            )

    def test_sequential_extends_rbf_kernel(self) -> None:
        """Simulate KernelCoRectron: sequential extends with RBF kernel values."""
        T = 50
        lam = 3.0
        rng = np.random.default_rng(99)
        xs = rng.standard_normal((T, 4))

        # Build full Gram matrix directly.
        G = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                G[i, j] = float(np.exp(-0.5 * np.sum((xs[i] - xs[j]) ** 2)))

        cache = KernelCholeskyCache(lambda_reg=lam)
        for t in range(T):
            k_prev = G[:t, t]
            rho = G[t, t]
            cache.extend(k_prev, rho)

        # L L^T should equal G + lam*I
        np.testing.assert_allclose(
            cache.L @ cache.L.T, G + lam * np.eye(T), atol=1e-8,
            err_msg="Sequential extend: L L^T != G + lam*I",
        )

        # Random solve check
        b = rng.standard_normal(T)
        x_chol = cache.solve(b)
        x_ref = np.linalg.solve(G + lam * np.eye(T), b)
        np.testing.assert_allclose(x_chol, x_ref, atol=1e-8,
                                   err_msg="Sequential extend: solve error")

    def test_schur_complement_update(self) -> None:
        """Verify that the Schur-complement coefficient update matches direct solve.

        This mirrors exactly what KernelCoRectron does on each round:
          s      = (G_{t-1} + lam*I)^{-1} k_prev
          sigma  = rho + lam - k_prev^T s
          y_new  = (1 - k_prev^T alpha_{t-1}) / sigma
          alpha_t = [alpha_{t-1} - y_new * s;  y_new]
        should satisfy (G_t + lam*I) alpha_t = 1_t.
        """
        T = 20
        lam = 2.0
        rng = np.random.default_rng(55)
        xs = rng.standard_normal((T, 3))

        G = np.zeros((T, T))
        for i in range(T):
            for j in range(T):
                G[i, j] = float(np.exp(-0.5 * np.sum((xs[i] - xs[j]) ** 2)))

        cache = KernelCholeskyCache(lambda_reg=lam)
        alpha = np.empty(0)

        for t in range(T):
            k_prev = G[:t, t]
            rho = G[t, t]

            if t == 0:
                cache.extend(np.empty(0), rho)
                sigma = rho + lam
                alpha = np.array([1.0 / sigma])
            else:
                # Schur-complement update (must use OLD cache)
                s = cache.solve(k_prev)
                sigma = rho + lam - float(np.dot(k_prev, s))
                if abs(sigma) < 1e-300:
                    sigma = 1e-300
                y_new = (1.0 - float(np.dot(k_prev, alpha))) / sigma
                new_alpha = np.empty(t + 1)
                new_alpha[:t] = alpha - y_new * s
                new_alpha[t] = y_new
                alpha = new_alpha

                # Extend after computing s
                cache.extend(k_prev, rho)

            # Check: (G_{t+1} + lam*I) alpha = 1_{t+1}
            H = G[:t + 1, :t + 1] + lam * np.eye(t + 1)
            residual = H @ alpha - np.ones(t + 1)
            np.testing.assert_allclose(
                residual, np.zeros(t + 1), atol=1e-9,
                err_msg=f"Schur-complement update failed at t={t}",
            )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
