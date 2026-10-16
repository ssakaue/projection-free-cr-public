from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import clip_vector_to_norm, kernel_vector
from .validation import finite_float, integer


@dataclass
class RoundData:
    context: np.ndarray
    true_scores: np.ndarray


class SyntheticGenerator:
    """Synthetic contextual recommendation generator.

    Linear contextual model:
      H = R^{n x p}, true score at round t is U_* z_t.

    Kernel contextual model:
      h_*(z) = sum_j k(z, c_j) * a_j, where a_j in R^n.
    """

    def __init__(self, cfg: dict[str, Any], seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(integer(seed, "seed"))

        dcfg = cfg["dataset"]
        self.model_type = str(dcfg.get("model_type", "linear_contextual"))
        self.action_dim = integer(dcfg["action_dim"], "dataset.action_dim", minimum=1)
        self.context_dim = integer(dcfg["context_dim"], "dataset.context_dim", minimum=1)
        # Two-step normalization: contexts are first clipped to the raw bound
        # (context_norm_raw), then multiplied by context_scale which is set by
        # config.py's normalization policy (target_context_l2 / context_norm_raw).
        # The final per-round norm bound is context_norm = context_norm_raw * context_scale.
        # Action-vector normalization is handled separately by the action set.
        self.context_norm_raw = finite_float(dcfg.get("context_norm_raw", dcfg.get("context_norm", 1.0)), "dataset.context_norm_raw", positive=True)
        self.context_scale = finite_float(dcfg.get("context_scale", 1.0), "dataset.context_scale", positive=True)
        self.context_norm = self.context_norm_raw * self.context_scale
        if self.context_norm_raw <= 0.0:
            raise ValueError("dataset.context_norm_raw must be positive")
        if self.context_scale <= 0.0:
            raise ValueError("dataset.context_scale must be positive")

        if self.model_type == "linear_contextual":
            lcfg = dcfg.get("linear", {})
            norm_u = finite_float(lcfg.get("true_weight_norm", 1.0), "dataset.linear.true_weight_norm", positive=True)
            u = self.rng.normal(size=(self.action_dim, self.context_dim))
            u_norm = float(np.linalg.norm(u, ord="fro"))
            if u_norm == 0.0:
                raise ValueError("sampled zero true weight unexpectedly")
            self.true_weight = (norm_u / u_norm) * u
            self.kernel_sigma_true = None
            self.kernel_type_true = None
            self.kernel_degree_true = None
            self.kernel_coef0_true = None
            self.kernel_alpha_true = None
            self.centers = None
            self.coeffs = None
        elif self.model_type == "kernel_contextual":
            kcfg = dcfg.get("kernel", {})
            n_centers = integer(kcfg.get("num_centers", 16), "dataset.kernel.num_centers", minimum=1)
            coeff_std = finite_float(kcfg.get("coeff_std", 1.0), "dataset.kernel.coeff_std", positive=True)
            true_weight_norm = finite_float(kcfg.get("true_weight_norm", 1.0), "dataset.kernel.true_weight_norm", positive=True)
            self.kernel_type_true = str(kcfg.get("kernel_type", "rbf")).lower()
            self.kernel_sigma_true = float(kcfg.get("sigma_true", 1.0))
            self.kernel_degree_true = integer(kcfg.get("degree", 2), "dataset.kernel.degree", minimum=1)
            self.kernel_coef0_true = float(kcfg.get("coef0", 0.0))
            self.kernel_alpha_true = float(kcfg.get("alpha", 1.0))

            if self.kernel_type_true not in {"rbf", "poly"}:
                raise ValueError(f"Unsupported kernel.kernel_type: {self.kernel_type_true}")
            if self.kernel_type_true == "rbf" and self.kernel_sigma_true <= 0.0:
                raise ValueError("kernel.sigma_true must be positive for RBF")
            if self.kernel_type_true == "poly" and self.kernel_degree_true <= 0:
                raise ValueError("kernel.degree must be positive for polynomial kernel")
            if coeff_std <= 0.0:
                raise ValueError("kernel.coeff_std must be positive")
            if true_weight_norm <= 0.0:
                raise ValueError("kernel.true_weight_norm must be positive")

            centers = self.rng.normal(size=(n_centers, self.context_dim))
            clipped = np.vstack([clip_vector_to_norm(c, self.context_norm_raw) for c in centers])
            self.centers = self.context_scale * clipped
            coeffs = None
            rkhs_norm_raw = 0.0
            # Retry up to 8 times in case the sampled coefficients are
            # degenerate (RKHS norm ≈ 0), which can happen if the kernel
            # matrix is nearly singular.
            for _ in range(8):
                sampled = self.rng.normal(scale=coeff_std, size=(n_centers, self.action_dim))
                rkhs_norm_raw = self._rkhs_norm(sampled)
                if rkhs_norm_raw > 1e-12:
                    coeffs = (true_weight_norm / rkhs_norm_raw) * sampled
                    break
            if coeffs is None:
                raise ValueError("Failed to sample non-degenerate kernel coefficients for RKHS normalization")
            self.coeffs = coeffs
            self.kernel_true_weight_norm_target = true_weight_norm
            self.kernel_true_weight_norm_raw = rkhs_norm_raw
            self.kernel_true_weight_norm_realized = self._rkhs_norm(self.coeffs)
            self.true_weight = None
        else:
            raise ValueError(f"Unknown dataset.model_type: {self.model_type}")

    def _sample_context(self) -> np.ndarray:
        z = self.rng.normal(size=self.context_dim)
        z_raw = clip_vector_to_norm(z, self.context_norm_raw)
        return self.context_scale * z_raw

    def _kernel(self, z: np.ndarray) -> np.ndarray:
        assert self.centers is not None
        assert self.kernel_type_true is not None
        return kernel_vector(
            x_hist=self.centers,
            x=z,
            kernel_type=self.kernel_type_true,
            kernel_sigma=float(self.kernel_sigma_true),
            kernel_degree=int(self.kernel_degree_true),
            kernel_coef0=float(self.kernel_coef0_true),
            kernel_alpha=float(self.kernel_alpha_true),
        )

    def _kernel_matrix(self) -> np.ndarray:
        assert self.centers is not None
        cols = [self._kernel(self.centers[j]) for j in range(self.centers.shape[0])]
        mat = np.column_stack(cols)
        return 0.5 * (mat + mat.T)

    def _rkhs_norm(self, coeffs: np.ndarray) -> float:
        # ||h||²_H = tr(coeffs^T K coeffs), where K[i,j] = k(c_i, c_j).
        # Derivation: h(z) = sum_j k(z, c_j) a_j, so
        #   <h, h>_H = sum_{i,j} k(c_i,c_j) <a_i, a_j> = tr(A^T K A).
        k = self._kernel_matrix()
        if not np.isfinite(coeffs).all() or not np.isfinite(k).all():
            raise FloatingPointError("Nonfinite coefficients or Gram matrix in RKHS normalization")
        norm_sq = float(np.trace(coeffs.T @ k @ coeffs))
        roundoff_scale = float(np.sum(np.abs(coeffs) * (np.abs(k) @ np.abs(coeffs))))
        tolerance = 64 * np.finfo(float).eps * k.shape[0] * roundoff_scale
        if not np.isfinite(norm_sq) or not np.isfinite(roundoff_scale) or norm_sq < -tolerance:
            raise FloatingPointError(f"Invalid squared RKHS norm: {norm_sq}")
        return float(np.sqrt(max(0.0, norm_sq)))

    def _true_scores(self, z: np.ndarray) -> np.ndarray:
        if self.model_type == "linear_contextual":
            assert self.true_weight is not None
            return self.true_weight @ z

        assert self.coeffs is not None
        basis = self._kernel(z)  # shape: num_centers
        # Weighted sum of vector coefficients -> shape: action_dim
        return basis @ self.coeffs

    def sample_round(self) -> RoundData:
        context = self._sample_context()
        true_scores = self._true_scores(context)
        if not np.isfinite(true_scores).all():
            raise FloatingPointError("Nonfinite true utility scores")
        return RoundData(context=context, true_scores=true_scores)
