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
    return _project_to_ball(v, max_norm)


def _project_to_ball(v: np.ndarray, radius: float) -> np.ndarray:
    norm_value = np.linalg.norm(v)
    norm = float(norm_value)
    # Squaring tiny components can underflow even when the computed norm is
    # nonzero. Keep the usual arithmetic outside that range.
    if np.isfinite(norm) and norm >= np.sqrt(np.finfo(norm_value.dtype).tiny):
        if norm <= radius:
            return v
        return (radius / norm) * v

    scale = float(np.max(np.abs(v), initial=0.0))
    if scale == 0.0:
        return v
    unit = v / scale
    unit_norm = float(np.linalg.norm(unit))
    if scale <= radius / unit_norm:
        return v
    return (radius / unit_norm) * unit


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
    return _project_to_ball(v, radius)


def _validate_kernel_input(x_hist: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(x_hist, np.ndarray) or not isinstance(x, np.ndarray) or x_hist.ndim != 2 or x.ndim != 1:
        raise ValueError("x_hist must be 2D and x must be 1D.")
    if x_hist.shape[1] != x.shape[0]:
        raise ValueError("dimension mismatch between x_hist and x.")
    if x_hist.dtype.kind not in "iuf" or x.dtype.kind not in "iuf":
        raise ValueError("Kernel inputs must be real numeric arrays")
    # Convert before subtraction or matrix multiplication to avoid integer
    # overflow. Float64 inputs retain their original values and storage.
    with np.errstate(over="raise", invalid="raise"):
        x_hist = np.asarray(x_hist, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x_hist).all() or not np.isfinite(x).all():
        raise ValueError("Kernel inputs must be finite in float64")
    return x_hist, x


def rbf_kernel_vector(x_hist: np.ndarray, x: np.ndarray, sigma: float) -> np.ndarray:
    """Return the RBF kernel in float64 for finite real numeric arrays.

    Raise FloatingPointError if a coordinate difference exceeds float64 range.
    """
    x_hist, x = _validate_kernel_input(x_hist, x)
    sigma = finite_float(sigma, "sigma", positive=True)
    with np.errstate(over="raise", invalid="raise"):
        diff = x_hist - x[None, :]
    # Dividing before squaring also handles tiny positive bandwidths at x_i=x.
    # Overflow in a nonzero squared distance correctly gives a zero kernel value.
    with np.errstate(over="ignore", under="ignore"):
        diff = diff / sigma
        sq = np.sum(diff * diff, axis=1)
        return np.exp(-0.5 * sq)


def poly_kernel_vector(
    x_hist: np.ndarray,
    x: np.ndarray,
    degree: int,
    coef0: float,
    alpha: float = 1.0,
) -> np.ndarray:
    """Return the polynomial kernel in float64 for finite real numeric arrays.

    Raise FloatingPointError when an intermediate value exceeds float64 range.
    """
    x_hist, x = _validate_kernel_input(x_hist, x)
    degree = integer(degree, "degree", minimum=1)
    alpha = finite_float(alpha, "alpha", positive=True)
    coef0 = finite_float(coef0, "coef0", nonnegative=True)
    with np.errstate(over="raise", invalid="raise"):
        dot = x_hist @ x
        # BLAS matrix products do not always honor NumPy's error-state flags.
        if not np.isfinite(dot).all():
            raise FloatingPointError("Polynomial kernel dot product exceeds float64 range")
        return np.power(alpha * dot + coef0, degree)


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
