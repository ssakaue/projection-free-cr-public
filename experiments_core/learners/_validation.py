"""Validation shared by the experiment learners."""
from __future__ import annotations

import numpy as np

from ..validation import finite_float, integer


def positive_float(value: float, name: str) -> float:
    return finite_float(value, name, positive=True)


def positive_int(value: int, name: str) -> int:
    return integer(value, name, minimum=1)


def finite_vector(value: np.ndarray, name: str, size: int | None = None) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.ndim != 1 or (size is not None and value.size != size):
        raise ValueError(f"{name} has an invalid shape")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def round_inputs(context, action_pred, action_expert, action_dim, context_dim=None):
    context = finite_vector(context, "context", context_dim)
    if context.size == 0:
        raise ValueError("context must be nonempty")
    action_pred = finite_vector(action_pred, "action_pred", action_dim)
    action_expert = finite_vector(action_expert, "action_expert", action_dim)
    return context, action_pred, action_expert


def kernel_parameters(kernel_type, sigma, degree, coef0, alpha):
    kernel_type = str(kernel_type).lower()
    if kernel_type not in {"rbf", "poly"}:
        raise ValueError("kernel_type must be rbf or poly")
    sigma = positive_float(sigma, "kernel_sigma")
    degree = positive_int(degree, "kernel_degree")
    alpha = positive_float(alpha, "kernel_alpha")
    coef0 = float(coef0)
    if not np.isfinite(coef0) or coef0 < 0:
        raise ValueError("kernel_coef0 must be finite and nonnegative")
    return kernel_type, sigma, degree, coef0, alpha


def require_finite(value, message):
    if not np.all(np.isfinite(value)):
        raise FloatingPointError(message)


def nonnegative_quadratic(value, scale, name):
    """Accept roundoff at zero, but reject a materially negative squared norm."""
    if not np.isfinite(value) or not np.isfinite(scale):
        raise FloatingPointError(f"{name} is not finite")
    tolerance = 64.0 * np.finfo(float).eps * max(float(scale), np.finfo(float).tiny)
    if value < -tolerance:
        raise FloatingPointError(f"{name} is negative ({value}); positive semidefiniteness was lost")
    return max(0.0, float(value))
