"""Independent finite-feature checks for learner updates and RKHS projection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments_core.learners import (
    CoRectronLearner, KernelCoRectronLearner, KernelCholeskyCache,
    ONSLearner, KernelONSLearner, OGDLearner,
)


def make_kons(radius=0.8, epsilon=0.07, eta=0.4):
    return KernelONSLearner(3, radius, eta, 0.5, 2.0, epsilon,
                            "poly", 1.0, 1, 0.0, 1.0, 1.0)


def primal_projection(y, curvature, radius):
    """Independent dense spectral solution in the finite feature space."""
    eigvals, eigvecs = np.linalg.eigh(curvature)
    coordinates = eigvecs.T @ y
    if np.linalg.norm(y) <= radius:
        return y.copy()
    def norm_error(multiplier):
        return np.linalg.norm(eigvals * coordinates / (eigvals + multiplier)) - radius
    high = max(eigvals) * (np.linalg.norm(y) / radius - 1)
    multiplier = brentq(norm_error, 0, high * 1.001 + 1e-12, xtol=1e-14)
    return eigvecs @ (eigvals * coordinates / (eigvals + multiplier))


@pytest.mark.parametrize("feature_shape", [(4, 3), (4, 7), (1, 1)])
def test_kons_projection_matches_primal_problem(feature_shape):
    rng = np.random.default_rng(384)
    features = rng.normal(size=feature_shape)
    gram = features.T @ features
    learner = make_kons()
    learner._G = gram
    learner._sum_diag = float(np.trace(gram))
    coefficients = rng.normal(size=feature_shape[1]) * 4
    result, projected = learner._fastproj_kernel(coefficients)
    curvature = learner.epsilon * np.eye(feature_shape[0]) + learner.eta**2 * features @ features.T
    expected = primal_projection(features @ coefficients, curvature, learner.radius)
    actual = features @ result
    assert projected
    assert np.linalg.norm(actual) <= learner.radius + 1e-12
    np.testing.assert_allclose(actual, expected, atol=5e-10, rtol=0)


@pytest.mark.parametrize("radius", [1e6, 0.4])
def test_linear_kernel_ons_equals_finite_dimensional_ons(radius):
    rng = np.random.default_rng(52)
    n, p, epsilon, eta = 3, 2, 0.03, 0.2
    linear = ONSLearner(n, p, radius, eta, 0.5, 2.0, epsilon)
    kernel = make_kons(radius, epsilon, eta)
    for t in range(1, 21):
        context = rng.normal(size=p)
        context /= np.linalg.norm(context)
        residual = rng.normal(size=n)
        linear.update(context, residual, np.zeros(n), t)
        kernel.update(context, residual, np.zeros(n), t)
        features = np.array([np.outer(r, z).ravel()
                             for r, z in zip(kernel.residual_hist, kernel.context_hist)]).T
        np.testing.assert_allclose(linear.w, features @ kernel.coeffs, atol=2e-9, rtol=0)
    assert (linear.projection_events > 0) == (radius < 1)
    assert kernel.projection_events == linear.projection_events


def test_corectron_linear_and_kernel_match_direct_regularized_solve():
    rng = np.random.default_rng(53)
    n, p, regularization = 3, 2, 0.03
    linear = CoRectronLearner(n, p, regularization)
    kernel = KernelCoRectronLearner(n, regularization, "poly", 1.0, 1, 0.0, 1.0)
    curvature = regularization * np.eye(n * p)
    cumulative_gradient = np.zeros(n * p)
    for t in range(1, 51):
        context = rng.normal(size=p)
        context /= np.linalg.norm(context)
        residual = rng.normal(size=n)
        linear.update(context, residual, np.zeros(n), t)
        kernel.update(context, residual, np.zeros(n), t)
        gradient = np.outer(residual, context).ravel()
        curvature += np.outer(gradient, gradient)
        cumulative_gradient += gradient
        expected = -np.linalg.solve(curvature, cumulative_gradient)
        features = np.array([np.outer(r, z).ravel()
                             for r, z in zip(kernel.residual_hist, kernel.context_hist)]).T
        np.testing.assert_allclose(linear.w, expected, atol=2e-10, rtol=0)
        np.testing.assert_allclose(-features @ kernel.coeffs, expected, atol=2e-10, rtol=0)


def test_singular_cache_fails_instead_of_manufacturing_factor():
    cache = KernelCholeskyCache(0)
    with pytest.raises(np.linalg.LinAlgError):
        cache.extend(np.zeros(0), 0.0)
    assert cache.t == 0
    assert cache.L.shape == (0, 0)
    cache.extend(np.zeros(0), 1.0)
    before = cache.L.copy()
    with pytest.raises(np.linalg.LinAlgError):
        cache.extend(np.ones(1), 1.0)
    assert cache.t == 1
    np.testing.assert_array_equal(cache.L, before)


def test_small_positive_regularization_is_not_silently_changed():
    learner = make_kons(epsilon=1e-14)
    learner._extend_matrices(np.zeros(0), 0.0)
    learner._extend_matrices(np.zeros(1), 0.0)
    np.testing.assert_allclose(learner._L_B @ learner._L_B.T, 1e-14 * np.eye(2), rtol=1e-14, atol=0)


def test_corectron_invalid_inverse_fails_without_silent_skip():
    learner = CoRectronLearner(2, 1, 1.0)
    learner.a_inv = -2 * np.eye(2)
    before = learner.a_inv.copy()
    with pytest.raises(FloatingPointError, match="round 7"):
        learner.update(np.ones(1), np.array([1.0, 0.0]), np.zeros(2), 7)
    np.testing.assert_array_equal(learner.a_inv, before)


@pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
def test_invalid_regularization_rejected(value):
    with pytest.raises(ValueError):
        CoRectronLearner(2, 1, value)
    with pytest.raises(ValueError):
        make_kons(epsilon=value)


def test_nan_input_rejected_before_update():
    learner = CoRectronLearner(2, 1, 1.0)
    with pytest.raises(ValueError):
        learner.update(np.array([np.nan]), np.ones(2), np.zeros(2), 1)
    np.testing.assert_array_equal(learner.a_inv, np.eye(2))


def test_ogd_matches_projected_gradient():
    learner = OGDLearner(2, 1, 0.5, 0.3, "constant")
    residual = np.array([3.0, -4.0])
    learner.update(np.ones(1), residual, np.zeros(2), 1)
    np.testing.assert_allclose(learner.w, -0.5 * residual / np.linalg.norm(residual))
