from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.config import resolve_config


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class TheoryHyperparamTests(unittest.TestCase):
    def test_linear_main_theory_values(self) -> None:
        raw = _load_yaml(REPO_ROOT / "configs/main_linear_t10000_s10.yaml")
        cfg = resolve_config(raw=raw, cli_ons_lr=2.0, cli_horizon=None, cli_seeds=None)

        d_ref = int(cfg["dataset"]["d_ref"])
        self.assertEqual(d_ref, 100)
        self.assertAlmostEqual(float(cfg["dataset"]["g_bound_linear"]), 1.0, places=12)

        core = cfg["learners"]["corectron"]
        self.assertAlmostEqual(float(core["lambda_reg_theory"]), 100.0, places=12)

        ons = cfg["learners"]["ons"]
        self.assertAlmostEqual(float(ons["eta_fixed"]), 0.1, places=12)
        self.assertAlmostEqual(float(ons["ons_lr_theory"]), 1.96, places=12)
        self.assertAlmostEqual(float(ons["epsilon_theory"]), 100.0, places=12)
        self.assertAlmostEqual(float(ons["epsilon"]), 100.0, places=12)

        ogd = cfg["learners"]["ogd"]
        self.assertAlmostEqual(float(ogd["ogd_lr_theory"]), 2.0 / math.sqrt(10000.0), places=12)
        self.assertAlmostEqual(float(ogd["step0"]), 0.02, places=12)

    def test_kernel_main_theory_values(self) -> None:
        raw = _load_yaml(REPO_ROOT / "configs/main_kernel_t1000_s10.yaml")
        cfg = resolve_config(raw=raw, cli_ons_lr=2.0, cli_horizon=None, cli_seeds=None, cli_kons_lr=2.0)

        d_ref = int(cfg["dataset"]["d_ref"])
        self.assertEqual(d_ref, 100)

        ker = cfg["learners"]["corectron_rbf"]
        self.assertAlmostEqual(float(ker["kappa"]), 1.0, places=12)
        self.assertAlmostEqual(float(ker["lambda_reg_theory"]), 100.0, places=12)
        self.assertAlmostEqual(float(ker["lambda_reg"]), 100.0, places=12)

        kons = cfg["learners"]["kons"]
        self.assertAlmostEqual(float(kons["eta_fixed"]), 0.1, places=12)
        self.assertAlmostEqual(float(kons["kons_lr_theory"]), 1.96, places=12)
        self.assertEqual(int(kons["d_cap_linear_proxy"]), 100)
        self.assertAlmostEqual(float(kons["epsilon_theory"]), 100.0, places=12)
        self.assertAlmostEqual(float(kons["epsilon"]), 100.0, places=12)


if __name__ == "__main__":
    unittest.main()
