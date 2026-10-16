"""Numerical range checks for kernel evaluation and Euclidean projection."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments_core.utils import (
    clip_vector_to_norm,
    euclidean_ball_projection,
    poly_kernel_vector,
    rbf_kernel_vector,
)


def test_polynomial_kernel_converts_before_integer_dot_product():
    result = poly_kernel_vector(np.array([[2**32]], dtype=np.int64),
                                np.array([2**32], dtype=np.int64), 1, 0.0)
    np.testing.assert_array_equal(result, np.array([float(2**64)]))
    assert result.dtype == np.float64


def test_rbf_kernel_converts_before_integer_subtraction():
    result = rbf_kernel_vector(np.array([[2**63 - 1]], dtype=np.int64),
                               np.array([-2**63 + 1], dtype=np.int64), float(2**63))
    np.testing.assert_allclose(result, [math.exp(-2.0)], rtol=1e-15, atol=0)


def test_rbf_kernel_rejects_unrepresentable_difference():
    with pytest.raises(FloatingPointError):
        rbf_kernel_vector(np.array([[1e308]]), np.array([-1e308]), 1e308)


def test_rbf_kernel_allows_large_scaled_distance_and_tiny_bandwidth():
    result = rbf_kernel_vector(np.array([[0.0], [1.0]]), np.array([0.0]), 1e-300)
    np.testing.assert_array_equal(result, [1.0, 0.0])


def test_polynomial_kernel_rejects_intermediate_overflow():
    with pytest.raises(FloatingPointError):
        poly_kernel_vector(np.array([[1e308]]), np.array([1e308]), 1, 0.0)


@pytest.mark.parametrize("dtype", [bool, complex, object, str])
@pytest.mark.parametrize("kernel", [rbf_kernel_vector, poly_kernel_vector])
def test_kernel_rejects_nonreal_or_nonnumeric_arrays(dtype, kernel):
    arguments = (1.0,) if kernel is rbf_kernel_vector else (1, 0.0)
    with pytest.raises(ValueError, match="real numeric"):
        kernel(np.array([[1]], dtype=dtype), np.array([1], dtype=dtype), *arguments)


def test_ordinary_float64_kernel_values_are_unchanged():
    history = np.random.default_rng(52).normal(size=(20, 10))
    point = np.random.default_rng(53).normal(size=10)
    for sigma in (0.5, 1.0, 2.0):
        expected = np.exp(-0.5 * np.sum(((history - point[None, :]) / sigma)**2, axis=1))
        np.testing.assert_array_equal(rbf_kernel_vector(history, point, sigma), expected)
    expected = np.power(0.5 * (history @ point) + 1.0, 4)
    np.testing.assert_array_equal(poly_kernel_vector(history, point, 4, 1.0, 0.5), expected)


@pytest.mark.parametrize("project", [clip_vector_to_norm, euclidean_ball_projection])
@pytest.mark.parametrize("vector,radius,expected", [
    ([1e-200], 1e-250, [1e-250]),
    ([4e-162], 3.9e-162, [3.9e-162]),
    ([3e-200, -4e-200], 1e-250, [0.6e-250, -0.8e-250]),
    ([3e200, -4e200], 1.0, [0.6, -0.8]),
])
def test_projection_handles_underflow_and_overflow(project, vector, radius, expected):
    result = project(np.array(vector), radius)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=0)
    assert math.hypot(*result) <= radius * (1.0 + 2e-15)


@pytest.mark.parametrize("project", [clip_vector_to_norm, euclidean_ball_projection])
@pytest.mark.parametrize("vector,radius", [
    ([], 1.0), ([0.0, 0.0], 1e-250), ([3e-200, 4e-200], 1e-199),
    ([0.3, -0.4], 0.5), ([3.0, -4.0], 6.0),
])
def test_projection_preserves_points_inside_ball(project, vector, radius):
    value = np.array(vector)
    assert project(value, radius) is value


@pytest.mark.parametrize("project", [clip_vector_to_norm, euclidean_ball_projection])
def test_projection_preserves_ordinary_arithmetic(project):
    value = np.random.default_rng(54).normal(size=100)
    expected = (1.0 / float(np.linalg.norm(value))) * value
    np.testing.assert_array_equal(project(value, 1.0), expected)
