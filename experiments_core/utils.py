from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .validation import finite_float, integer


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True)


def _finite_vector(v: np.ndarray, name: str) -> None:
    if not isinstance(v, np.ndarray) or v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite one-dimensional array")


def clip_vector_to_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    _finite_vector(v, "v")
    max_norm = finite_float(max_norm, "max_norm", positive=True)
    norm = float(np.linalg.norm(v))
    if norm <= max_norm or norm == 0.0:
        return v
    if not np.isfinite(norm):
        scale = float(np.max(np.abs(v)))
        unit = v / scale
        return (max_norm / np.linalg.norm(unit)) * unit
    return (max_norm / norm) * v


def top_m_binary(scores: np.ndarray, m: int) -> np.ndarray:
    _finite_vector(scores, "scores")
    m = integer(m, "m", minimum=1)
    n = scores.shape[0]
    if m > n:
        raise ValueError("m cannot exceed score dimension")
    # Stable ascending sort followed by the last m indices makes ties explicit:
    # larger indices are selected first. The cost is O(n log n).
    idx = np.argsort(scores, kind="stable")[-m:]
    out = np.zeros(n, dtype=np.float64)
    out[idx] = 1.0
    return out


def euclidean_ball_projection(v: np.ndarray, radius: float) -> np.ndarray:
    _finite_vector(v, "v")
    radius = finite_float(radius, "radius", positive=True)
    norm = float(np.linalg.norm(v))
    if norm <= radius or norm == 0.0:
        return v
    if not np.isfinite(norm):
        scale = float(np.max(np.abs(v)))
        unit = v / scale
        return (radius / np.linalg.norm(unit)) * unit
    return (radius / norm) * v


def _validate_kernel_input(x_hist: np.ndarray, x: np.ndarray) -> None:
    if not isinstance(x_hist, np.ndarray) or not isinstance(x, np.ndarray) or x_hist.ndim != 2 or x.ndim != 1:
        raise ValueError("x_hist must be 2D and x must be 1D.")
    if not np.isfinite(x_hist).all() or not np.isfinite(x).all():
        raise ValueError("Kernel inputs must be finite")
    if x_hist.shape[1] != x.shape[0]:
        raise ValueError("dimension mismatch between x_hist and x.")


def rbf_kernel_vector(x_hist: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Return k(x_i, x) for each row x_i in x_hist with RBF kernel."""
    _validate_kernel_input(x_hist, x)
    sigma = finite_float(sigma, "sigma", positive=True)
    # Dividing before squaring also handles tiny positive bandwidths at x_i=x.
    # Overflow in a nonzero squared distance correctly gives a zero kernel value.
    with np.errstate(over="ignore", under="ignore"):
        diff = (x_hist - x[None, :]) / sigma
        sq = np.sum(diff * diff, axis=1)
        return np.exp(-0.5 * sq)


def poly_kernel_vector(
    x_hist: np.ndarray,
    x: np.ndarray,
    degree: int,
    coef0: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return k(x_i, x) for each row x_i in x_hist with polynomial kernel."""
    _validate_kernel_input(x_hist, x)
    degree = integer(degree, "degree", minimum=1)
    alpha = finite_float(alpha, "alpha", positive=True)
    coef0 = finite_float(coef0, "coef0", nonnegative=True)
    with np.errstate(over="raise", invalid="raise"):
        return np.power(alpha * (x_hist @ x) + coef0, degree)


def kernel_vector(
    x_hist: np.ndarray,
    x: np.ndarray,
    kernel_type: str,
    kernel_sigma: float = 1.0,
    kernel_degree: int = 2,
    kernel_coef0: float = 0.0,
    kernel_alpha: float = 1.0,
) -> np.ndarray:
    """Return k(x_i, x) for each row x_i in x_hist with selected kernel."""
    ktype = str(kernel_type).lower()
    if ktype == "rbf":
        return rbf_kernel_vector(x_hist, x, kernel_sigma)
    if ktype == "poly":
        return poly_kernel_vector(
            x_hist=x_hist,
            x=x,
            degree=kernel_degree,
            coef0=float(kernel_coef0),
            alpha=float(kernel_alpha),
        )
    raise ValueError(f"Unsupported kernel_type: {kernel_type}")
