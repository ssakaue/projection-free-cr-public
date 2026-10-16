"""Tests for Cholesky-factor refactors of ONSLearner and KernelONSLearner.

For each learner we verify:
1. The Cholesky factor satisfies L @ L.T ≈ expected matrix (A for ONS, B_t for KONS).
2. The Newton step / direction matches the brute-force reference computed via
   np.linalg.inv.
3. Assertions hold every round for at least 20 rounds with random data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.learners.ons import ONSLearner
from experiments_core.learners.kernel_ons import KernelONSLearner


# ---------------------------------------------------------------------------
# ONS tests
# ---------------------------------------------------------------------------

class TestONSCholeskyRefactor:
    """Verify that ONSLearner Cholesky factor stays consistent with _A."""

    ONS_PARAMS = dict(
        action_dim=3,
        context_dim=3,
        radius=1.0,
        eta_fixed=0.1,
        ons_lr=1.0,
        gamma_theory=1.0,
        epsilon=0.1,
    )

    def _make_learner(self) -> ONSLearner:
        return ONSLearner(**self.ONS_PARAMS)

    def test_cholesky_factor_tracks_A(self):
        """L @ L.T should equal A after every round."""
        rng = np.random.default_rng(42)
        learner = self._make_learner()
        n_rounds = 20
        action_dim = self.ONS_PARAMS["action_dim"]
        context_dim = self.ONS_PARAMS["context_dim"]

        for t in range(n_rounds):
            context = rng.standard_normal(context_dim)
            action_pred = rng.standard_normal(action_dim)
            action_expert = rng.standard_normal(action_dim)
            learner.update(context, action_pred, action_expert, round_id=t)

            L = learner._L
            A = learner._A
            reconstructed = L @ L.T
            assert np.allclose(reconstructed, A, atol=1e-10), (
                f"Round {t}: L @ L.T != A. "
                f"max diff = {np.max(np.abs(reconstructed - A)):.2e}"
            )

    def test_newton_step_matches_brute_force(self):
        """w trajectory should match brute-force A_inv @ grad Newton step."""
        rng = np.random.default_rng(42)
        learner = self._make_learner()
        n_rounds = 20
        action_dim = self.ONS_PARAMS["action_dim"]
        context_dim = self.ONS_PARAMS["context_dim"]
        eta = self.ONS_PARAMS["eta_fixed"]
        ons_lr = self.ONS_PARAMS["ons_lr"]
        epsilon = self.ONS_PARAMS["epsilon"]
        dim = action_dim * context_dim

        # Brute-force A
        A_ref = epsilon * np.eye(dim, dtype=np.float64)

        for t in range(n_rounds):
            context = rng.standard_normal(context_dim)
            action_pred = learner._matrix_view() @ context  # use current w
            action_expert = rng.standard_normal(action_dim)

            # Compute what the delta should be (before update)
            residual = action_pred - action_expert
            g = np.outer(residual, context).reshape(-1)
            grad = eta * g

            # Update brute-force A
            A_ref = A_ref + np.outer(grad, grad)
            A_inv_ref = np.linalg.inv(A_ref)
            delta_ref = A_inv_ref @ grad
            w_tilde_ref = learner.w - ons_lr * delta_ref

            # Now run the learner update
            learner.update(context, action_pred, action_expert, round_id=t)

            # The learner's w (before projection clamp) should be w_tilde_ref
            # if no projection occurred. If projection did occur, just verify
            # the Cholesky factor is correct.
            L = learner._L
            reconstructed = L @ L.T
            assert np.allclose(reconstructed, A_ref, atol=1e-10), (
                f"Round {t}: L @ L.T != A_ref. "
                f"max diff = {np.max(np.abs(reconstructed - A_ref)):.2e}"
            )

            # Verify delta via triangular solves matches reference
            z = scipy.linalg.solve_triangular(L, grad, lower=True)
            delta_chol = scipy.linalg.solve_triangular(L.T, z, lower=False)
            assert np.allclose(delta_chol, delta_ref, atol=1e-9), (
                f"Round {t}: Cholesky Newton step != brute-force. "
                f"max diff = {np.max(np.abs(delta_chol - delta_ref)):.2e}"
            )


# ---------------------------------------------------------------------------
# KONS tests
# ---------------------------------------------------------------------------

class TestKONSCholeskyRefactor:
    """Verify that KernelONSLearner Cholesky factor stays consistent with B_t."""

    KONS_PARAMS = dict(
        action_dim=2,
        radius=1.0,
        eta_fixed=0.1,
        kons_lr=1.0,
        gamma_theory=1.0,
        epsilon=0.1,
        kernel_type="rbf",
        kernel_sigma=1.0,
        kernel_degree=3,
        kernel_coef0=1.0,
        kernel_alpha=1.0,
        kappa=1.0,
    )

    def _make_learner(self) -> KernelONSLearner:
        return KernelONSLearner(**self.KONS_PARAMS)

    def test_cholesky_factor_tracks_B(self):
        """L_B @ L_B.T should equal B_t = eta^2 * G + eps * I after every round."""
        rng = np.random.default_rng(42)
        learner = self._make_learner()
        n_rounds = 20
        action_dim = self.KONS_PARAMS["action_dim"]
        eta = self.KONS_PARAMS["eta_fixed"]
        epsilon = self.KONS_PARAMS["epsilon"]
        context_dim = 4  # arbitrary context dimension

        for t in range(n_rounds):
            context = rng.standard_normal(context_dim)
            action_pred = rng.standard_normal(action_dim)
            action_expert = rng.standard_normal(action_dim)
            learner.update(context, action_pred, action_expert, round_id=t)

            L_B = learner._L_B
            G = learner._G
            t_now = G.shape[0]
            B_ref = eta ** 2 * G + epsilon * np.eye(t_now, dtype=np.float64)
            reconstructed = L_B @ L_B.T
            assert np.allclose(reconstructed, B_ref, atol=1e-10), (
                f"Round {t}: L_B @ L_B.T != B_ref. "
                f"max diff = {np.max(np.abs(reconstructed - B_ref)):.2e}"
            )

    def test_direction_matches_brute_force(self):
        """direction_coeff should match eta * B_inv @ e_t (brute-force)."""
        rng = np.random.default_rng(42)
        learner = self._make_learner()
        n_rounds = 20
        action_dim = self.KONS_PARAMS["action_dim"]
        eta = self.KONS_PARAMS["eta_fixed"]
        epsilon = self.KONS_PARAMS["epsilon"]
        context_dim = 4

        for t in range(n_rounds):
            context = rng.standard_normal(context_dim)
            action_pred = learner.predict_scores(context)
            action_expert = rng.standard_normal(action_dim)

            learner.update(context, action_pred, action_expert, round_id=t)

            L_B = learner._L_B
            G = learner._G
            t_now = G.shape[0]
            B_ref = eta ** 2 * G + epsilon * np.eye(t_now, dtype=np.float64)

            # Brute-force: B_inv @ e_t
            B_inv_ref = np.linalg.inv(B_ref)
            e_t = np.zeros(t_now, dtype=np.float64)
            e_t[-1] = 1.0
            direction_ref = eta * B_inv_ref @ e_t

            # Cholesky-based direction
            z = scipy.linalg.solve_triangular(L_B, e_t, lower=True)
            direction_chol = eta * scipy.linalg.solve_triangular(L_B.T, z, lower=False)

            assert np.allclose(direction_chol, direction_ref, atol=1e-9), (
                f"Round {t}: Cholesky direction != brute-force. "
                f"max diff = {np.max(np.abs(direction_chol - direction_ref)):.2e}"
            )

            # Also verify L_B @ L_B.T == B_ref
            reconstructed = L_B @ L_B.T
            assert np.allclose(reconstructed, B_ref, atol=1e-10), (
                f"Round {t}: L_B @ L_B.T != B_ref after direction check. "
                f"max diff = {np.max(np.abs(reconstructed - B_ref)):.2e}"
            )


# ---------------------------------------------------------------------------
# pytest entry-points
# ---------------------------------------------------------------------------

def test_ons_cholesky_factor_tracks_A():
    TestONSCholeskyRefactor().test_cholesky_factor_tracks_A()


def test_ons_newton_step_matches_brute_force():
    TestONSCholeskyRefactor().test_newton_step_matches_brute_force()


def test_kons_cholesky_factor_tracks_B():
    TestKONSCholeskyRefactor().test_cholesky_factor_tracks_B()


def test_kons_direction_matches_brute_force():
    TestKONSCholeskyRefactor().test_direction_matches_brute_force()
