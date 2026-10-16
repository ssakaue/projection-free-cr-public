"""Regression tests for FastProj (Choice 2) in ONS and KernelONS.

Tests cover:
  1. ONS: Cholesky factor accuracy (L @ L.T ≈ A)
  2. ONS: FastProj projection onto Euclidean ball (inside / outside cases)
  3. ONS: Projected point satisfies Mahalanobis optimality (KKT)
  4. ONS: Descent direction sign (regression for the +/- sign bug)
  5. KONS: Cholesky factor accuracy (L_B @ L_B.T ≈ B)
  6. KONS: FastProj projection onto RKHS ball
  7. Integration: ONS cumulative regret stays bounded on synthetic data
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.learners.ons import ONSLearner
from experiments_core.learners.kernel_ons import KernelONSLearner


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ons(dim: int = 6, radius: float = 1.0, epsilon: float = 1.0,
              eta: float = 0.1, ons_lr: float = 2.0) -> ONSLearner:
    """Create an ONSLearner with action_dim * context_dim = dim."""
    # Pick factors so action_dim * context_dim = dim
    return ONSLearner(
        action_dim=dim,
        context_dim=1,
        radius=radius,
        eta_fixed=eta,
        gamma_theory=1.0 / ons_lr,
        ons_lr=ons_lr,
        epsilon=epsilon,
    )


def _feed_random_updates(learner: ONSLearner, n: int, rng: np.random.Generator):
    """Feed n random gradient updates to the learner, building up A."""
    dim = learner.dim
    for i in range(n):
        context = rng.standard_normal(learner.context_dim)
        action_pred = rng.standard_normal(learner.action_dim)
        action_expert = rng.standard_normal(learner.action_dim)
        learner.update(context, action_pred, action_expert, i + 1)


def _make_kons(action_dim: int = 3, radius: float = 1.0, epsilon: float = 1.0,
               eta: float = 0.1, kons_lr: float = 2.0) -> KernelONSLearner:
    return KernelONSLearner(
        action_dim=action_dim,
        radius=radius,
        eta_fixed=eta,
        gamma_theory=1.0 / kons_lr,
        kons_lr=kons_lr,
        epsilon=epsilon,
        kernel_type="rbf",
        kernel_sigma=1.0,
        kernel_degree=2,
        kernel_coef0=0.0,
        kernel_alpha=1.0,
        kappa=1.0,
    )


# ===========================================================================
# ONS Tests
# ===========================================================================

class TestONSShermanMorrison(unittest.TestCase):
    """Verify that the Cholesky factor L satisfies L @ L.T ≈ A."""

    def test_inverse_matches_direct(self):
        rng = np.random.default_rng(42)
        learner = _make_ons(dim=6, epsilon=2.0)
        _feed_random_updates(learner, 10, rng)

        # Verify Cholesky factor: L @ L.T should equal A
        np.testing.assert_allclose(
            learner._L @ learner._L.T, learner._A, atol=1e-10,
            err_msg="Cholesky factor drifted: L @ L.T != A"
        )

    def test_A_is_symmetric(self):
        rng = np.random.default_rng(123)
        learner = _make_ons(dim=8)
        _feed_random_updates(learner, 15, rng)
        np.testing.assert_allclose(
            learner._A, learner._A.T, atol=1e-14,
            err_msg="Curvature matrix A must remain symmetric"
        )


class TestONSFastProj(unittest.TestCase):
    """Verify FastProj (Choice 2) Mahalanobis projection properties."""

    def test_inside_ball_unchanged(self):
        """A point inside the ball should not be modified."""
        learner = _make_ons(dim=4, radius=10.0, epsilon=1.0)
        u = np.array([1.0, 2.0, 0.5, -0.3])
        self.assertLess(np.linalg.norm(u), 10.0)

        projected = learner._fastproj(u)
        np.testing.assert_allclose(projected, u, atol=1e-14)

    def test_outside_ball_projected_to_boundary(self):
        """A point outside the ball should land on the boundary."""
        rng = np.random.default_rng(7)
        learner = _make_ons(dim=6, radius=1.0, epsilon=1.0)
        _feed_random_updates(learner, 5, rng)

        # Create a point well outside the ball
        u = rng.standard_normal(6) * 5.0
        self.assertGreater(np.linalg.norm(u), 1.0)

        projected = learner._fastproj(u)
        self.assertAlmostEqual(
            np.linalg.norm(projected), 1.0, places=6,
            msg="FastProj must project onto the Euclidean ball boundary"
        )

    def test_projection_reduces_a_norm_distance(self):
        """The projected point should be closer (A-norm) to u than any other
        feasible point on the boundary."""
        rng = np.random.default_rng(99)
        learner = _make_ons(dim=4, radius=1.0, epsilon=1.0)
        _feed_random_updates(learner, 8, rng)

        u = rng.standard_normal(4) * 3.0
        projected = learner._fastproj(u)

        # A-norm distance: (p - u)^T A (p - u)
        diff = projected - u
        dist_proj = float(diff @ learner._A @ diff)

        # Compare with several random feasible points on the boundary
        for _ in range(50):
            v = rng.standard_normal(4)
            v = (1.0 / np.linalg.norm(v)) * v  # on boundary
            diff_v = v - u
            dist_v = float(diff_v @ learner._A @ diff_v)
            self.assertLessEqual(
                dist_proj, dist_v + 1e-6,
                msg="FastProj result should minimize A-norm distance to u"
            )

    def test_identity_A_reduces_to_euclidean_proj(self):
        """When A = eps*I (no updates), FastProj = Euclidean projection."""
        learner = _make_ons(dim=5, radius=2.0, epsilon=3.0)
        # No updates => A = 3*I, Mahalanobis w.r.t. scalar*I = Euclidean

        u = np.array([3.0, 4.0, 0.0, 0.0, 0.0])  # norm = 5
        projected = learner._fastproj(u)

        expected = (2.0 / 5.0) * u  # simple Euclidean projection
        np.testing.assert_allclose(projected, expected, atol=1e-8)


class TestONSDescentDirection(unittest.TestCase):
    """Regression test: ONS must descend (not ascend) in the Newton direction."""

    def test_regret_bounded_on_constant_expert(self):
        """On a trivial problem, ONS cumulative regret should not blow up.

        If the sign were flipped, regret would grow linearly/diverge.
        """
        rng = np.random.default_rng(0)
        action_dim = 3
        context_dim = 2
        learner = ONSLearner(
            action_dim=action_dim,
            context_dim=context_dim,
            radius=1.0,
            eta_fixed=0.1,
            gamma_theory=0.5,
            ons_lr=2.0,
            epsilon=10.0,
        )

        horizon = 100
        true_W = rng.standard_normal((action_dim, context_dim)) * 0.1
        true_W /= max(np.linalg.norm(true_W, "fro"), 1.0)  # inside ball

        cumulative_regret = 0.0
        from experiments_core.utils import top_m_binary

        for t in range(1, horizon + 1):
            context = rng.standard_normal(context_dim)
            context /= np.linalg.norm(context)

            true_scores = true_W @ context
            action_expert = top_m_binary(true_scores, 1)

            scores = learner.predict_scores(context)
            action_pred = top_m_binary(scores, 1)

            inst_regret = float(true_scores @ (action_expert - action_pred))
            cumulative_regret += inst_regret

            learner.update(context, action_pred, action_expert, t)

        # With correct sign, regret should be sublinear and modest.
        # With flipped sign, it was ~63 on similar setups.
        self.assertLess(
            cumulative_regret, 20.0,
            msg=f"ONS regret {cumulative_regret:.2f} is too large — "
                f"possible descent direction sign bug"
        )


# ===========================================================================
# KernelONS Tests
# ===========================================================================

class TestKONSBorderedInverse(unittest.TestCase):
    """Verify that the Cholesky factor L_B satisfies L_B @ L_B.T ≈ B_t."""

    def test_inverse_matches_direct(self):
        rng = np.random.default_rng(42)
        learner = _make_kons(action_dim=3, epsilon=2.0)

        # Feed a few updates to grow the matrices
        for i in range(8):
            context = rng.standard_normal(3)
            context /= np.linalg.norm(context)
            action_pred = rng.standard_normal(3)
            action_expert = rng.standard_normal(3)
            learner.update(context, action_pred, action_expert, i + 1)

        t = learner._G.shape[0]
        self.assertGreater(t, 0)

        # B = eta^2 * G + eps * I
        B_direct = learner.eta ** 2 * learner._G + learner.epsilon * np.eye(t)

        # Verify Cholesky factor: L_B @ L_B.T should equal B
        np.testing.assert_allclose(
            learner._L_B @ learner._L_B.T, B_direct, atol=1e-8,
            err_msg="Cholesky factor drifted: L_B @ L_B.T != B"
        )


class TestKONSFastProj(unittest.TestCase):
    """Verify FastProj (Choice 2) for RKHS ball projection."""

    def test_inside_rkhs_ball_unchanged(self):
        """Coefficients whose RKHS norm is within R should not be modified."""
        rng = np.random.default_rng(55)
        learner = _make_kons(action_dim=2, radius=100.0, epsilon=1.0)

        # Feed updates to build G
        for i in range(5):
            context = rng.standard_normal(2)
            context /= np.linalg.norm(context)
            action_pred = rng.standard_normal(2)
            action_expert = rng.standard_normal(2)
            learner.update(context, action_pred, action_expert, i + 1)

        # Create small coefficients that are inside the ball
        t = learner._G.shape[0]
        y = np.ones(t) * 0.01
        rkhs_sq = float(y @ learner._G @ y)
        self.assertLess(math.sqrt(rkhs_sq), 100.0)

        proj, did_project = learner._fastproj_kernel(y)
        self.assertFalse(did_project)
        np.testing.assert_allclose(proj, y, atol=1e-14)

    def test_outside_rkhs_ball_projected(self):
        """Coefficients outside the RKHS ball should be projected onto boundary."""
        rng = np.random.default_rng(77)
        learner = _make_kons(action_dim=2, radius=0.5, epsilon=1.0)

        # Feed updates
        for i in range(6):
            context = rng.standard_normal(2)
            context /= np.linalg.norm(context)
            action_pred = rng.standard_normal(2)
            action_expert = rng.standard_normal(2)
            learner.update(context, action_pred, action_expert, i + 1)

        t = learner._G.shape[0]
        y = rng.standard_normal(t) * 10.0  # large coefficients
        rkhs_sq = float(y @ learner._G @ y)
        self.assertGreater(math.sqrt(max(0, rkhs_sq)), 0.5)

        proj, did_project = learner._fastproj_kernel(y)
        self.assertTrue(did_project)

        rkhs_sq_proj = float(proj @ learner._G @ proj)
        self.assertAlmostEqual(
            math.sqrt(max(0, rkhs_sq_proj)), 0.5, places=4,
            msg="Projected coefficients should lie on RKHS ball boundary"
        )


class TestKONSIntegration(unittest.TestCase):
    """End-to-end regression test for KernelONS."""

    def test_kons_runs_without_error(self):
        """KernelONS should run for a number of rounds without crashing."""
        rng = np.random.default_rng(0)
        learner = _make_kons(action_dim=3, radius=1.0, epsilon=5.0, kons_lr=1.0)

        for t in range(1, 30):
            context = rng.standard_normal(3)
            context /= np.linalg.norm(context)
            scores = learner.predict_scores(context)
            action_pred = np.zeros(3)
            action_pred[np.argmax(scores)] = 1.0
            action_expert = np.zeros(3)
            action_expert[rng.integers(3)] = 1.0
            learner.update(context, action_pred, action_expert, t)

        # Should have run without exception
        self.assertEqual(learner.coeffs.size, 29)


# ===========================================================================
# n_bisect overflow guard
# ===========================================================================

class TestNBisectOverflow(unittest.TestCase):
    """Regression test: n_bisect must not overflow when A = alpha*I."""

    def test_no_update_fastproj(self):
        """FastProj on a fresh learner (A = eps*I) should not raise."""
        learner = _make_ons(dim=4, radius=0.5, epsilon=1.0)
        u = np.array([1.0, 1.0, 1.0, 1.0])  # norm = 2, outside ball

        # This would previously raise OverflowError if n_bisect guard was missing
        projected = learner._fastproj(u)
        self.assertAlmostEqual(np.linalg.norm(projected), 0.5, places=8)


if __name__ == "__main__":
    unittest.main()
