from __future__ import annotations

import math

import numpy as np

from ..utils import top_m_binary
from ..validation import finite_float, integer
from .base import ActionSetPlugin, ResidualBounds


class MOutOfNActionSet(ActionSetPlugin):
    def __init__(self, action_dim: int, m_select: int, scale: float = 1.0):
        """
        Parameters
        ----------
        action_dim : int
            Total number of items n.
        m_select : int
            Number of items to select m (1 <= m <= n).
        scale : float
            Normalisation factor applied to all action vectors.  config.py
            sets this to ``action_scale = target_residual_l2 / residual_l2_raw``
            so that the scaled residual L2 bound equals the target (default 1.0).
        """
        self.action_dim = integer(action_dim, "action_dim", minimum=1)
        self.m_select = integer(m_select, "m_select", minimum=1)
        self.scale = finite_float(scale, "scale", positive=True)
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.m_select <= 0:
            raise ValueError("m_select must be positive")
        if self.m_select > self.action_dim:
            raise ValueError("m_select cannot exceed action_dim")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")

    @property
    def name(self) -> str:
        return "m_out_of_n"

    def _check_scores(self, scores: np.ndarray) -> None:
        if not isinstance(scores, np.ndarray) or scores.shape != (self.action_dim,) or not np.isfinite(scores).all():
            raise ValueError(f"scores must be a finite array of shape ({self.action_dim},)")

    def select(self, scores: np.ndarray) -> np.ndarray:
        self._check_scores(scores)
        return self.scale * top_m_binary(scores, self.m_select)

    def expert(self, true_scores: np.ndarray) -> np.ndarray:
        self._check_scores(true_scores)
        return self.scale * top_m_binary(true_scores, self.m_select)

    def user_action(
        self,
        true_scores: np.ndarray,
        rng: np.random.Generator,
        subopt_prob: float = 0.0,
        subopt_mode: str = "uniform",
        subopt_noise_std: float = 0.0,
    ) -> np.ndarray:
        """Possibly suboptimal user action.

        With probability subopt_prob, draws a possibly suboptimal feasible action
        determined by subopt_mode.  Otherwise returns the optimal action,
        identical to expert().

        Parameters
        ----------
        true_scores : ndarray of shape (n,)
            True utility vector u^base_t.
        rng : np.random.Generator
            RNG used for the random draw.  Must be independent from the data
            generator's RNG so that subopt_prob=0 reproduces the optimal run.
        subopt_prob : float in [0, 1]
            Probability of deviating from the optimal action.
            subopt_prob=0 always returns expert().
        subopt_mode : {"uniform", "one_swap", "score_perturb", "score_perturb_relative"}
            How the suboptimal action is generated when the deviation occurs.

            "uniform"  — uniformly random m-subset of {0,...,n-1}.
                         delta_t can be anywhere in [0, B].

            "one_swap" — replace the m-th ranked item (last item in the
                         optimal set) with the (m+1)-th ranked item (first
                         item outside the optimal set).
                         delta_t = scale * (u_{(m)} - u_{(m+1)}) >= 0 (small gap).
                         Falls back to optimal when m_select == action_dim.

            "score_perturb" — add i.i.d. Gaussian noise N(0, subopt_noise_std^2)
                         to true_scores and take argmax of the perturbed vector.
                         delta_t = <u, x*> - <u, x_perturb> >= 0.
                         Noise std is controlled by subopt_noise_std.

            "score_perturb_relative" — as above, with standard deviation
                         subopt_noise_std * ||true_scores||_2 in each coordinate.
        subopt_noise_std : float >= 0
            Standard deviation of the score perturbation noise.
            For score_perturb_relative, this is the dimensionless multiplier.
        """
        self._check_scores(true_scores)
        subopt_prob = finite_float(subopt_prob, "subopt_prob", nonnegative=True)
        subopt_noise_std = finite_float(subopt_noise_std, "subopt_noise_std", nonnegative=True)
        if subopt_prob > 1.0:
            raise ValueError("subopt_prob must be in [0, 1]")
        if subopt_mode not in {"uniform", "one_swap", "score_perturb", "score_perturb_relative"}:
            raise ValueError(f"Unknown subopt_mode: {subopt_mode!r}")
        if subopt_prob > 0.0 and rng.random() < subopt_prob:
            if subopt_mode == "uniform":
                perm = rng.permutation(self.action_dim)
                out = np.zeros(self.action_dim, dtype=np.float64)
                out[perm[: self.m_select]] = 1.0
                return self.scale * out
            elif subopt_mode == "one_swap":
                if self.m_select < self.action_dim:
                    # Sort items by descending true utility (stable for reproducibility).
                    idx = np.argsort(true_scores, kind="stable")[::-1]
                    out = np.zeros(self.action_dim, dtype=np.float64)
                    # Keep top (m-1) items, replace rank-m with rank-(m+1).
                    out[idx[: self.m_select - 1]] = 1.0
                    out[idx[self.m_select]] = 1.0
                    return self.scale * out
                # Fall through to optimal when swap is impossible.
            elif subopt_mode == "score_perturb":
                noise = rng.normal(scale=subopt_noise_std, size=true_scores.shape)
                return self.scale * top_m_binary(true_scores + noise, self.m_select)
            elif subopt_mode == "score_perturb_relative":
                scale = subopt_noise_std * float(np.linalg.norm(true_scores))
                noise = rng.normal(scale=scale if scale > 0.0 else 0.0, size=true_scores.shape)
                return self.scale * top_m_binary(true_scores + noise, self.m_select)
            else:
                raise ValueError(f"Unknown subopt_mode: {subopt_mode!r}")
        return self.scale * top_m_binary(true_scores, self.m_select)

    def residual_bounds(self) -> ResidualBounds:
        # For x_hat, x_star in {0,1}^n with exactly m ones:
        # ||x_hat - x_star||_2 <= sqrt(2 * min(m, n-m)).
        l2 = self.scale * math.sqrt(2.0 * min(self.m_select, self.action_dim - self.m_select))
        return ResidualBounds(residual_l2_max=l2)
