from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.config import resolve_config
from experiments_core.data import SyntheticGenerator
from experiments_core.utils import kernel_vector


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _rkhs_norm(generator: SyntheticGenerator) -> float:
    centers = generator.centers
    coeffs = generator.coeffs
    assert centers is not None
    assert coeffs is not None
    cols = [
        kernel_vector(
            x_hist=centers,
            x=centers[j],
            kernel_type=str(generator.kernel_type_true),
            kernel_sigma=float(generator.kernel_sigma_true),
            kernel_degree=int(generator.kernel_degree_true),
            kernel_coef0=float(generator.kernel_coef0_true),
            kernel_alpha=float(generator.kernel_alpha_true),
        )
        for j in range(centers.shape[0])
    ]
    k = np.column_stack(cols)
    k = 0.5 * (k + k.T)
    norm_sq = float(np.trace(coeffs.T @ k @ coeffs))
    return math.sqrt(max(0.0, norm_sq))


class KernelTrueWeightNormTests(unittest.TestCase):
    def test_resolved_config_records_kernel_scale_roles(self) -> None:
        raw = _load_yaml(REPO_ROOT / "configs/main_kernel_t1000_s10.yaml")
        cfg = resolve_config(raw=raw, cli_ons_lr=2.0, cli_horizon=None, cli_seeds=None)
        kcfg = cfg["dataset"]["kernel"]
        self.assertEqual(kcfg["coeff_std_role"], "pre_norm_shape_only")
        self.assertEqual(kcfg["true_weight_norm_role"], "rkhs_norm_target")

    def test_true_weight_norm_enforced_and_score_scales_linearly(self) -> None:
        raw_base = _load_yaml(REPO_ROOT / "configs/main_kernel_t1000_s10.yaml")
        raw_small = copy.deepcopy(raw_base)
        raw_large = copy.deepcopy(raw_base)
        raw_small["dataset"]["kernel"]["true_weight_norm"] = 0.5
        raw_large["dataset"]["kernel"]["true_weight_norm"] = 2.0

        cfg_small = resolve_config(raw=raw_small, cli_ons_lr=2.0, cli_horizon=None, cli_seeds=None)
        cfg_large = resolve_config(raw=raw_large, cli_ons_lr=2.0, cli_horizon=None, cli_seeds=None)

        seed = 7
        gen_small = SyntheticGenerator(cfg_small, seed=seed)
        gen_large = SyntheticGenerator(cfg_large, seed=seed)

        norm_small = _rkhs_norm(gen_small)
        norm_large = _rkhs_norm(gen_large)
        self.assertAlmostEqual(norm_small, 0.5, places=10)
        self.assertAlmostEqual(norm_large, 2.0, places=10)

        ratio = 4.0
        for _ in range(20):
            rd_small = gen_small.sample_round()
            rd_large = gen_large.sample_round()
            self.assertTrue(np.allclose(rd_small.context, rd_large.context, atol=1e-12, rtol=0.0))
            self.assertTrue(np.allclose(rd_large.true_scores, ratio * rd_small.true_scores, atol=1e-10, rtol=1e-10))


if __name__ == "__main__":
    unittest.main()
