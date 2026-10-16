from __future__ import annotations

import copy
import itertools
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments_core.action_sets.m_out_of_n import MOutOfNActionSet
from experiments_core.config import resolve_config
from experiments_core.data import SyntheticGenerator
from experiments_core.learners.corectron import CoRectronLearner
from experiments_core.learners.kernel_ons import KernelONSLearner
from experiments_core.learners.kernel_corectron import KernelCoRectronLearner
from experiments_core.learners.ogd import OGDLearner
from experiments_core.learners.ons import ONSLearner
from experiments_core.runner import run_experiment
from experiments_core.utils import clip_vector_to_norm, top_m_binary


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolved_cfg(
    rel_path: str,
    *,
    horizon: int | None = None,
    seeds: list[int] | None = None,
    ons_lr: float = 2.0,
    kons_lr: float | None = None,
) -> dict:
    raw = _load_yaml(REPO_ROOT / rel_path)
    cfg = resolve_config(
        raw=copy.deepcopy(raw),
        cli_ons_lr=ons_lr,
        cli_horizon=horizon,
        cli_seeds=",".join(map(str, seeds)) if seeds is not None else None,
        cli_kons_lr=kons_lr,
    )
    return cfg


class ActionSetInvariantTests(unittest.TestCase):
    def test_m_out_of_n_residual_bounds_are_valid_and_tight(self) -> None:
        action_set = MOutOfNActionSet(action_dim=6, m_select=2, scale=1.0)
        bounds = action_set.residual_bounds()

        actions = []
        for idx in itertools.combinations(range(6), 2):
            a = np.zeros(6)
            a[list(idx)] = 1.0
            actions.append(a)

        l2_max = 0.0
        for a in actions:
            for b in actions:
                diff = a - b
                l2_max = max(l2_max, float(np.linalg.norm(diff, ord=2)))
                self.assertLessEqual(float(np.linalg.norm(diff, ord=2)), bounds.residual_l2_max + 1e-12)

        self.assertAlmostEqual(l2_max, bounds.residual_l2_max, places=12)


class SyntheticGeneratorInvariantTests(unittest.TestCase):
    def test_contexts_respect_normalized_bound_for_linear_and_kernel(self) -> None:
        cfg_linear = _resolved_cfg("configs/main_linear_t10000_s10.yaml", horizon=10, seeds=[0])
        cfg_kernel = _resolved_cfg(
            "configs/main_kernel_t1000_s10.yaml",
            horizon=10,
            seeds=[0],
            kons_lr=2.0,
        )

        for cfg in [cfg_linear, cfg_kernel]:
            gen = SyntheticGenerator(cfg, seed=123)
            target = float(cfg["dataset"]["context_norm"])
            for _ in range(100):
                rd = gen.sample_round()
                self.assertLessEqual(float(np.linalg.norm(rd.context)), target + 1e-12)
                self.assertTrue(np.all(np.isfinite(rd.true_scores)))

    def test_linear_true_weight_matches_requested_norm(self) -> None:
        cfg = _resolved_cfg("configs/main_linear_t10000_s10.yaml", horizon=10, seeds=[0])
        gen = SyntheticGenerator(cfg, seed=7)
        self.assertIsNotNone(gen.true_weight)
        self.assertAlmostEqual(
            float(np.linalg.norm(gen.true_weight, ord="fro")),
            float(cfg["dataset"]["linear"]["true_weight_norm"]),
            places=12,
        )

    def test_kernel_realized_norm_matches_requested_norm_for_poly(self) -> None:
        raw = _load_yaml(REPO_ROOT / "configs/main_kernel_t1000_s10.yaml")
        raw["dataset"]["kernel"]["kernel_type"] = "poly"
        raw["dataset"]["kernel"]["degree"] = 4
        raw["dataset"]["kernel"]["coef0"] = 0.0
        raw["dataset"]["kernel"]["alpha"] = 1.0
        cfg = resolve_config(
            raw=raw,
            cli_ons_lr=2.0,
            cli_horizon=20,
            cli_seeds="0",
            cli_kons_lr=2.0,
            cli_kernel_type="poly",
            cli_kernel_degree=4,
            cli_kernel_coef0=0.0,
            cli_kernel_alpha=1.0,
        )
        gen = SyntheticGenerator(cfg, seed=11)
        self.assertAlmostEqual(
            float(gen.kernel_true_weight_norm_realized),
            float(cfg["dataset"]["kernel"]["true_weight_norm"]),
            places=10,
        )


class LearnerDomainInvariantTests(unittest.TestCase):
    def test_ons_iterates_stay_in_ball(self) -> None:
        rng = np.random.default_rng(0)
        learner = ONSLearner(
            action_dim=5,
            context_dim=3,
            radius=1.0,
            eta_fixed=0.1,
            gamma_theory=0.5,
            ons_lr=2.0,
            epsilon=10.0,
            proj_tol=1e-10,
        )

        for t in range(1, 41):
            context = clip_vector_to_norm(rng.standard_normal(3), 1.0)
            pred = top_m_binary(rng.standard_normal(5), 2)
            expert = top_m_binary(rng.standard_normal(5), 2)
            learner.update(context, pred, expert, t)

            self.assertTrue(np.all(np.isfinite(learner.w)))
            self.assertLessEqual(float(np.linalg.norm(learner.w)), learner.radius + 1e-8)

    def test_kons_iterates_stay_in_rkhs_ball(self) -> None:
        rng = np.random.default_rng(1)
        learner = KernelONSLearner(
            action_dim=5,
            radius=1.0,
            eta_fixed=0.1,
            gamma_theory=0.5,
            kons_lr=2.0,
            epsilon=10.0,
            kernel_type="rbf",
            kernel_sigma=1.0,
            kernel_degree=2,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
            kappa=1.0,
            proj_tol=1e-10,
        )

        for t in range(1, 31):
            context = clip_vector_to_norm(rng.standard_normal(4), 1.0)
            pred = top_m_binary(rng.standard_normal(5), 2)
            expert = top_m_binary(rng.standard_normal(5), 2)
            learner.update(context, pred, expert, t)

            self.assertTrue(np.all(np.isfinite(learner.coeffs)))
            self.assertTrue(np.all(np.isfinite(learner._G)))
            self.assertTrue(np.all(np.isfinite(learner._L_B)))
            self.assertLessEqual(learner._rkhs_norm_sq(learner.coeffs), learner.radius ** 2 + 1e-7)

    def test_poly_kernel_learners_stay_finite(self) -> None:
        rng = np.random.default_rng(2)
        kons = KernelONSLearner(
            action_dim=5,
            radius=1.0,
            eta_fixed=0.1,
            gamma_theory=0.5,
            kons_lr=2.0,
            epsilon=10.0,
            kernel_type="poly",
            kernel_sigma=1.0,
            kernel_degree=4,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
            kappa=1.0,
            proj_tol=1e-10,
        )
        core = KernelCoRectronLearner(
            action_dim=5,
            lambda_reg=10.0,
            kernel_type="poly",
            kernel_sigma=1.0,
            kernel_degree=4,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
        )

        for t in range(1, 21):
            context = clip_vector_to_norm(rng.standard_normal(4), 1.0)
            pred = top_m_binary(rng.standard_normal(5), 2)
            expert = top_m_binary(rng.standard_normal(5), 2)

            kons.update(context, pred, expert, t)
            core.update(context, pred, expert, t)

            self.assertTrue(np.all(np.isfinite(kons.coeffs)))
            self.assertTrue(np.all(np.isfinite(core.coeffs)))
            self.assertLessEqual(kons._rkhs_norm_sq(kons.coeffs), kons.radius ** 2 + 1e-7)


class RunnerBehaviorTests(unittest.TestCase):
    def _assert_deterministic(self, cfg: dict) -> None:
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            round_df_1, seed_df_1, agg_df_1 = run_experiment(cfg, Path(tmp1))
            round_df_2, seed_df_2, agg_df_2 = run_experiment(cfg, Path(tmp2))

            round_cols = [
                "seed",
                "algorithm",
                "round_id",
                "instant_regret",
                "cumulative_regret",
            ]
            seed_cols = ["seed", "algorithm", "final_regret", "projection_events"]
            agg_cols = [
                "algorithm",
                "num_seeds",
                "final_regret_mean",
                "final_regret_std",
                "projection_events_mean",
                "projection_events_std",
                "final_regret_ci95",
                "projection_events_ci95",
            ]

            pd.testing.assert_frame_equal(round_df_1[round_cols], round_df_2[round_cols], check_exact=True)
            pd.testing.assert_frame_equal(seed_df_1[seed_cols], seed_df_2[seed_cols], check_exact=True)
            pd.testing.assert_frame_equal(agg_df_1[agg_cols], agg_df_2[agg_cols], check_exact=True)

    def test_linear_run_is_deterministic_ignoring_timing(self) -> None:
        cfg = _resolved_cfg("configs/smoke_linear.yaml", horizon=30, seeds=[0, 1])
        self._assert_deterministic(cfg)

    def test_kernel_run_is_deterministic_ignoring_timing(self) -> None:
        cfg = _resolved_cfg(
            "configs/main_kernel_t1000_s10.yaml",
            horizon=30,
            seeds=[0],
            kons_lr=2.0,
        )
        self._assert_deterministic(cfg)

    def test_output_schema_uses_projection_events(self) -> None:
        cfg = _resolved_cfg("configs/smoke_linear.yaml", horizon=20, seeds=[0])
        with tempfile.TemporaryDirectory() as tmp:
            round_df, seed_df, agg_df = run_experiment(cfg, Path(tmp))

        self.assertIn("projection_events", seed_df.columns)
        self.assertNotIn("total_linear_solves", seed_df.columns)
        self.assertIn("projection_events_mean", agg_df.columns)
        self.assertNotIn("total_linear_solves_mean", agg_df.columns)
        self.assertNotIn("linear_solves", round_df.columns)


class ProjectionEventTests(unittest.TestCase):
    """Verify that projection_events is correctly tracked for all learners.

    Key contracts:
    - ONS/KONS: projection_events is incremented each round the iterate lands
      outside the domain ball (FastProj is triggered).  When the radius is
      very small, nearly every round triggers a projection.
    - CoRectron / KernelCoRectron: projection-free by design — projection_events
      must remain 0 regardless of inputs.
    - OGD: projection is O(d) clipping, but projection_events is NOT tracked
      (stays 0 by BaseLearner default).
    - All learners: update() return value must be 0.
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_round(self, rng, action_dim, context_dim):
        context = clip_vector_to_norm(rng.standard_normal(context_dim), 1.0)
        pred = top_m_binary(rng.standard_normal(action_dim), action_dim // 2)
        expert = top_m_binary(rng.standard_normal(action_dim), action_dim // 2)
        return context, pred, expert

    # ------------------------------------------------------------------
    # ONS: projection fires when radius is tiny
    # ------------------------------------------------------------------

    def test_ons_projection_events_incremented_when_radius_tiny(self):
        """ONS with radius=0.01 must trigger projection on nearly every round."""
        rng = np.random.default_rng(0)
        learner = ONSLearner(
            action_dim=4,
            context_dim=3,
            radius=0.01,   # tiny: iterates almost always violate the ball
            eta_fixed=1.0,
            gamma_theory=0.5,
            ons_lr=0.5,
            epsilon=0.01,
        )
        n_rounds = 30
        for t in range(1, n_rounds + 1):
            ctx, pred, expert = self._random_round(rng, 4, 3)
            ret = learner.update(ctx, pred, expert, t)
            self.assertEqual(ret, 0, "update() must return 0")

        self.assertGreater(
            learner.projection_events, 0,
            "ONS with tiny radius must have triggered at least one projection"
        )

    def test_ons_projection_events_zero_when_radius_large(self):
        """ONS with very large radius: iterates stay in the ball, no projections."""
        rng = np.random.default_rng(1)
        learner = ONSLearner(
            action_dim=4,
            context_dim=3,
            radius=1e6,
            eta_fixed=0.01,
            gamma_theory=0.5,
            ons_lr=2.0,
            epsilon=1.0,
        )
        for t in range(1, 51):
            ctx, pred, expert = self._random_round(rng, 4, 3)
            learner.update(ctx, pred, expert, t)

        self.assertEqual(
            learner.projection_events, 0,
            "ONS with huge radius should never project"
        )

    # ------------------------------------------------------------------
    # KONS: projection fires when radius is tiny
    # ------------------------------------------------------------------

    def test_kons_projection_events_incremented_when_radius_tiny(self):
        """KONS with radius=0.001 must trigger at least one RKHS projection."""
        rng = np.random.default_rng(2)
        learner = KernelONSLearner(
            action_dim=3,
            radius=0.001,
            eta_fixed=1.0,
            gamma_theory=0.5,
            kons_lr=0.5,
            epsilon=0.001,
            kernel_type="rbf",
            kernel_sigma=1.0,
            kernel_degree=2,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
            kappa=1.0,
        )
        n_rounds = 30
        for t in range(1, n_rounds + 1):
            ctx, pred, expert = self._random_round(rng, 3, 3)
            ret = learner.update(ctx, pred, expert, t)
            self.assertEqual(ret, 0, "update() must return 0")

        self.assertGreater(
            learner.projection_events, 0,
            "KONS with tiny radius must have triggered at least one projection"
        )

    def test_kons_projection_events_zero_when_radius_large(self):
        """KONS with very large radius should never project."""
        rng = np.random.default_rng(3)
        learner = KernelONSLearner(
            action_dim=3,
            radius=1e6,
            eta_fixed=0.01,
            gamma_theory=0.5,
            kons_lr=2.0,
            epsilon=1.0,
            kernel_type="rbf",
            kernel_sigma=1.0,
            kernel_degree=2,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
            kappa=1.0,
        )
        for t in range(1, 31):
            ctx, pred, expert = self._random_round(rng, 3, 3)
            learner.update(ctx, pred, expert, t)

        self.assertEqual(
            learner.projection_events, 0,
            "KONS with huge radius should never project"
        )

    # ------------------------------------------------------------------
    # CoRectron: always projection-free
    # ------------------------------------------------------------------

    def test_corectron_projection_events_always_zero(self):
        """CoRectron is projection-free: projection_events must remain 0."""
        rng = np.random.default_rng(4)
        learner = CoRectronLearner(action_dim=5, context_dim=4, lambda_reg=0.01)
        for t in range(1, 51):
            ctx, pred, expert = self._random_round(rng, 5, 4)
            ret = learner.update(ctx, pred, expert, t)
            self.assertEqual(ret, 0, "update() must return 0")
            self.assertEqual(
                learner.projection_events, 0,
                f"CoRectron must never increment projection_events (round {t})"
            )

    # ------------------------------------------------------------------
    # KernelCoRectron: always projection-free
    # ------------------------------------------------------------------

    def test_kernel_corectron_projection_events_always_zero(self):
        """KernelCoRectron is projection-free: projection_events must remain 0."""
        rng = np.random.default_rng(5)
        learner = KernelCoRectronLearner(
            action_dim=4,
            lambda_reg=0.01,
            kernel_type="rbf",
            kernel_sigma=1.0,
            kernel_degree=2,
            kernel_coef0=0.0,
            kernel_alpha=1.0,
        )
        for t in range(1, 41):
            ctx, pred, expert = self._random_round(rng, 4, 3)
            ret = learner.update(ctx, pred, expert, t)
            self.assertEqual(ret, 0, "update() must return 0")
            self.assertEqual(
                learner.projection_events, 0,
                f"KernelCoRectron must never increment projection_events (round {t})"
            )

    # ------------------------------------------------------------------
    # OGD: update() return value
    # ------------------------------------------------------------------

    def test_ogd_update_return_value(self):
        """OGD update() must return 0."""
        rng = np.random.default_rng(6)
        learner = OGDLearner(
            action_dim=4,
            context_dim=3,
            radius=1.0,
            step0=0.1,
            schedule="constant",
        )
        for t in range(1, 21):
            ctx, pred, expert = self._random_round(rng, 4, 3)
            ret = learner.update(ctx, pred, expert, t)
            self.assertEqual(ret, 0, "update() must return 0")

    # ------------------------------------------------------------------
    # projection_events consistency with seed_df in runner output
    # ------------------------------------------------------------------

    def test_runner_records_projection_events_nonzero_for_ons_tiny_radius(self):
        """When ONS is run with tiny radius, seed_df must record projection_events > 0."""
        import copy
        import tempfile
        raw = _load_yaml(REPO_ROOT / "configs/smoke_linear.yaml")

        # Force ONS to project by setting a tiny radius via direct config manipulation.
        # We override domain_radius in the ONS learner section after resolution.
        cfg = resolve_config(
            copy.deepcopy(raw),
            cli_ons_lr=0.5,
            cli_ons_epsilon=0.01,
            cli_horizon=50,
            cli_seeds="0",
        )
        # Set domain_radius tiny enough to force projections
        cfg["learners"]["ons"]["domain_radius"] = 0.001
        cfg["learners"]["ons"]["epsilon"] = 0.01
        # Disable corectron and ogd to isolate
        cfg["learners"]["corectron"]["enabled"] = False
        cfg["learners"]["ogd"]["enabled"] = False

        with tempfile.TemporaryDirectory() as tmp:
            _, seed_df, _ = run_experiment(cfg, Path(tmp))

        ons_rows = seed_df[seed_df["algorithm"] == "ons"]
        self.assertFalse(ons_rows.empty, "ONS rows must be present in seed_df")
        self.assertTrue(
            (ons_rows["projection_events"] > 0).all(),
            f"All ONS seeds must have projection_events > 0 with tiny radius; got {ons_rows['projection_events'].tolist()}"
        )


class SuboptimalUserActionTests(unittest.TestCase):
    """Tests for one_swap and score_perturb_relative user-action modes."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.action_set = MOutOfNActionSet(action_dim=10, m_select=5, scale=1.0)

    def _true_scores(self):
        return self.rng.standard_normal(10)

    # --- feasibility: all modes must return a valid m-of-n action ---

    def _assert_feasible(self, action):
        self.assertEqual(action.shape, (10,))
        self.assertTrue(np.all((action == 0.0) | (action == 1.0)), "action must be binary")
        self.assertEqual(int(action.sum()), 5, "action must select exactly m=5 items")

    def test_one_swap_is_feasible(self):
        for _ in range(50):
            u = self._true_scores()
            x = self.action_set.user_action(u, self.rng, subopt_prob=1.0, subopt_mode="one_swap")
            self._assert_feasible(x)

    def test_score_perturb_relative_is_feasible(self):
        for _ in range(50):
            u = self._true_scores()
            x = self.action_set.user_action(
                u, self.rng, subopt_prob=1.0,
                subopt_mode="score_perturb_relative", subopt_noise_std=1.0,
            )
            self._assert_feasible(x)

    # --- delta_t >= 0: suboptimal action never beats the optimum ---

    def _delta_t(self, u, x):
        x_opt = self.action_set.expert(u)
        return float(u @ (x_opt - x))

    def test_one_swap_delta_nonneg(self):
        for _ in range(100):
            u = self._true_scores()
            x = self.action_set.user_action(u, self.rng, subopt_prob=1.0, subopt_mode="one_swap")
            self.assertGreaterEqual(self._delta_t(u, x), -1e-12)

    def test_score_perturb_relative_delta_nonneg(self):
        for _ in range(100):
            u = self._true_scores()
            x = self.action_set.user_action(
                u, self.rng, subopt_prob=1.0,
                subopt_mode="score_perturb_relative", subopt_noise_std=0.5,
            )
            self.assertGreaterEqual(self._delta_t(u, x), -1e-12)

    # --- prob=0 always returns optimal regardless of mode ---

    def test_prob_zero_returns_optimal_one_swap(self):
        for _ in range(20):
            u = self._true_scores()
            x = self.action_set.user_action(u, self.rng, subopt_prob=0.0, subopt_mode="one_swap")
            x_opt = self.action_set.expert(u)
            np.testing.assert_array_equal(x, x_opt)

    def test_prob_zero_returns_optimal_score_perturb_relative(self):
        for _ in range(20):
            u = self._true_scores()
            x = self.action_set.user_action(
                u, self.rng, subopt_prob=0.0,
                subopt_mode="score_perturb_relative", subopt_noise_std=2.0,
            )
            x_opt = self.action_set.expert(u)
            np.testing.assert_array_equal(x, x_opt)

    # --- cumulative_suboptimality >= 0 in runner output ---

    def test_runner_cumulative_suboptimality_nonneg_one_swap(self):
        raw = _load_yaml(REPO_ROOT / "configs/smoke_linear.yaml")
        cfg = resolve_config(
            raw, cli_ons_lr=2.0, cli_horizon=50, cli_seeds="0",
            cli_subopt_prob=0.5, cli_subopt_mode="one_swap",
        )
        cfg["learners"]["ons"]["enabled"] = False
        cfg["learners"]["ogd"]["enabled"] = False
        with tempfile.TemporaryDirectory() as tmp:
            _, seed_df, _ = run_experiment(cfg, Path(tmp))
        self.assertTrue(
            (seed_df["cumulative_suboptimality"] >= -1e-12).all(),
            f"cumulative_suboptimality must be >= 0, got {seed_df['cumulative_suboptimality'].tolist()}",
        )

    def test_runner_cumulative_suboptimality_nonneg_score_perturb_relative(self):
        raw = _load_yaml(REPO_ROOT / "configs/smoke_linear.yaml")
        cfg = resolve_config(
            raw, cli_ons_lr=2.0, cli_horizon=50, cli_seeds="0",
            cli_subopt_prob=1.0, cli_subopt_mode="score_perturb_relative",
            cli_subopt_noise_std=0.5,
        )
        cfg["learners"]["ons"]["enabled"] = False
        cfg["learners"]["ogd"]["enabled"] = False
        with tempfile.TemporaryDirectory() as tmp:
            _, seed_df, _ = run_experiment(cfg, Path(tmp))
        self.assertTrue(
            (seed_df["cumulative_suboptimality"] >= -1e-12).all(),
            f"cumulative_suboptimality must be >= 0, got {seed_df['cumulative_suboptimality'].tolist()}",
        )

    # --- determinism: same seed produces same cumulative_suboptimality ---

    def test_subopt_run_is_deterministic(self):
        raw = _load_yaml(REPO_ROOT / "configs/smoke_linear.yaml")
        cfg = resolve_config(
            raw, cli_ons_lr=2.0, cli_horizon=30, cli_seeds="0,1",
            cli_subopt_prob=1.0, cli_subopt_mode="one_swap",
        )
        cfg["learners"]["ons"]["enabled"] = False
        cfg["learners"]["ogd"]["enabled"] = False
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            _, seed_df_1, _ = run_experiment(cfg, Path(tmp1))
            _, seed_df_2, _ = run_experiment(cfg, Path(tmp2))
        pd.testing.assert_frame_equal(
            seed_df_1[["seed", "algorithm", "final_regret", "cumulative_suboptimality"]].reset_index(drop=True),
            seed_df_2[["seed", "algorithm", "final_regret", "cumulative_suboptimality"]].reset_index(drop=True),
            check_exact=True,
        )


if __name__ == "__main__":
    unittest.main()
