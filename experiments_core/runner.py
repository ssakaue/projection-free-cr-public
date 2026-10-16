from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .action_sets import build_action_set
from .data import RoundData, SyntheticGenerator
from .learners import (
    CoRectronLearner,
    KernelCoRectronLearner,
    KernelONSLearner,
    OGDLearner,
    ONSLearner,
)
from .metrics import RoundMetric, SeedSummary
from .utils import ensure_dir


def build_learners(cfg: dict[str, Any]) -> list:
    dcfg = cfg["dataset"]
    lcfg = cfg["learners"]

    action_dim = int(dcfg["action_dim"])
    context_dim = int(dcfg["context_dim"])

    learners = []

    if bool(lcfg["corectron"]["enabled"]):
        learners.append(
            CoRectronLearner(
                action_dim=action_dim,
                context_dim=context_dim,
                lambda_reg=float(lcfg["corectron"]["lambda_reg"]),
            )
        )

    if bool(lcfg["ons"]["enabled"]):
        learners.append(
            ONSLearner(
                action_dim=action_dim,
                context_dim=context_dim,
                radius=float(lcfg["ons"]["domain_radius"]),
                eta_fixed=float(lcfg["ons"]["eta_fixed"]),
                gamma_theory=float(lcfg["ons"]["gamma_theory"]),
                ons_lr=float(lcfg["ons"]["ons_lr"]),
                epsilon=float(lcfg["ons"]["epsilon"]),
                proj_tol=float(lcfg["ons"].get("proj_tol", 1e-10)),
            )
        )

    if bool(lcfg["ogd"]["enabled"]):
        learners.append(
            OGDLearner(
                action_dim=action_dim,
                context_dim=context_dim,
                radius=float(lcfg["ogd"].get("domain_radius", 1.0)),
                step0=float(lcfg["ogd"]["step0"]),
                schedule=str(lcfg["ogd"].get("schedule", "constant")),
            )
        )

    if bool(lcfg.get("corectron_rbf", {}).get("enabled", False)):
        learners.append(
            KernelCoRectronLearner(
                action_dim=action_dim,
                lambda_reg=float(lcfg["corectron_rbf"]["lambda_reg"]),
                kernel_type=str(lcfg["corectron_rbf"].get("kernel_type", "rbf")),
                kernel_sigma=float(lcfg["corectron_rbf"]["kernel_sigma"]),
                kernel_degree=int(lcfg["corectron_rbf"].get("kernel_degree", 2)),
                kernel_coef0=float(lcfg["corectron_rbf"].get("kernel_coef0", 0.0)),
                kernel_alpha=float(lcfg["corectron_rbf"].get("kernel_alpha", 1.0)),
            )
        )

    if bool(lcfg.get("kons", {}).get("enabled", False)):
        learners.append(
            KernelONSLearner(
                action_dim=action_dim,
                radius=float(lcfg["kons"]["domain_radius"]),
                eta_fixed=float(lcfg["kons"]["eta_fixed"]),
                gamma_theory=float(lcfg["kons"]["gamma_theory"]),
                kons_lr=float(lcfg["kons"]["kons_lr"]),
                epsilon=float(lcfg["kons"]["epsilon"]),
                kernel_type=str(lcfg["kons"].get("kernel_type", "rbf")),
                kernel_sigma=float(lcfg["kons"]["kernel_sigma"]),
                kernel_degree=int(lcfg["kons"].get("kernel_degree", 2)),
                kernel_coef0=float(lcfg["kons"].get("kernel_coef0", 0.0)),
                kernel_alpha=float(lcfg["kons"].get("kernel_alpha", 1.0)),
                kappa=float(lcfg["kons"]["kappa"]),
                proj_tol=float(lcfg["kons"].get("proj_tol", 1e-10)),
            )
        )

    return learners


def _run_one(
    seed: int,
    learner,
    round_data: list[RoundData],
    action_set,
    progress_every: int,
    save_round_metrics: bool,
    subopt_prob: float = 0.0,
    subopt_mode: str = "uniform",
    subopt_noise_std: float = 0.0,
    user_rng: np.random.Generator | None = None,
) -> tuple[list[RoundMetric], SeedSummary]:
    horizon = len(round_data)

    cumulative_regret = 0.0
    total_runtime = 0.0
    total_cpu_time = 0.0
    cumulative_suboptimality = 0.0
    rows: list[RoundMetric] = []

    for t, data in enumerate(round_data, start=1):
        # Compute expert action outside the timing block: it is shared oracle
        # infrastructure, not part of the learner's per-round computational cost.
        action_optimal = action_set.expert(data.true_scores)
        if subopt_prob > 0.0 and user_rng is not None:
            action_expert = action_set.user_action(
                data.true_scores, user_rng, subopt_prob, subopt_mode, subopt_noise_std
            )
        else:
            action_expert = action_optimal
        # delta_t = max_x <u, x> - <u, x_t> >= 0  (per-round suboptimality)
        cumulative_suboptimality += float(data.true_scores @ (action_optimal - action_expert))
        tic = time.perf_counter()
        tic_cpu = time.process_time()
        scores = learner.predict_scores(data.context)
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"Nonfinite prediction: seed={seed}, algorithm={learner.name}, round={t}")
        action_pred = action_set.select(scores)
        learner.update(data.context, action_pred, action_expert, t)
        elapsed = time.perf_counter() - tic
        elapsed_cpu = time.process_time() - tic_cpu

        inst_regret = float(data.true_scores @ (action_expert - action_pred))
        if not np.isfinite(inst_regret) or not np.isfinite(cumulative_suboptimality):
            raise FloatingPointError(f"Nonfinite regret: seed={seed}, algorithm={learner.name}, round={t}")
        cumulative_regret += inst_regret
        total_runtime += elapsed
        total_cpu_time += elapsed_cpu

        if save_round_metrics:
            rows.append(
                RoundMetric(
                    seed=seed,
                    algorithm=learner.name,
                    round_id=t,
                    instant_regret=inst_regret,
                    cumulative_regret=cumulative_regret,
                    runtime_sec=elapsed,
                    cpu_time_sec=elapsed_cpu,
                )
            )

        if progress_every > 0 and t % progress_every == 0:
            print(
                f"[progress] seed={seed} algo={learner.name} round={t}/{horizon} "
                f"cum_regret={cumulative_regret:.4f}"
            )

    summary = SeedSummary(
        seed=seed,
        algorithm=learner.name,
        final_regret=cumulative_regret,
        total_runtime_sec=total_runtime,
        mean_runtime_sec=total_runtime / horizon,
        total_cpu_time_sec=total_cpu_time,
        mean_cpu_time_sec=total_cpu_time / horizon,
        projection_events=int(getattr(learner, "projection_events", 0)),
        cumulative_suboptimality=cumulative_suboptimality,
    )
    return rows, summary


def _to_df_round(rows: list[RoundMetric]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows], columns=RoundMetric.__dataclass_fields__)


def _to_df_summary(rows: list[SeedSummary]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in rows], columns=SeedSummary.__dataclass_fields__)


def _aggregate_seed_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    g = seed_df.groupby("algorithm", as_index=False)
    agg = g.agg(
        num_seeds=("seed", "count"),
        final_regret_mean=("final_regret", "mean"),
        final_regret_std=("final_regret", "std"),
        total_runtime_mean=("total_runtime_sec", "mean"),
        total_runtime_std=("total_runtime_sec", "std"),
        mean_runtime_per_round_mean=("mean_runtime_sec", "mean"),
        mean_runtime_per_round_std=("mean_runtime_sec", "std"),
        total_cpu_time_mean=("total_cpu_time_sec", "mean"),
        total_cpu_time_std=("total_cpu_time_sec", "std"),
        mean_cpu_time_per_round_mean=("mean_cpu_time_sec", "mean"),
        mean_cpu_time_per_round_std=("mean_cpu_time_sec", "std"),
        projection_events_mean=("projection_events", "mean"),
        projection_events_std=("projection_events", "std"),
        cumulative_suboptimality_mean=("cumulative_suboptimality", "mean"),
        cumulative_suboptimality_std=("cumulative_suboptimality", "std"),
    )

    n = agg["num_seeds"].astype(float)
    # Use t-distribution critical value (two-sided 95%) instead of z=1.96,
    # which is more appropriate for small sample sizes (e.g., n=10).
    t_crit = n.apply(lambda ni: stats.t.ppf(0.975, ni - 1))
    agg["final_regret_ci95"] = t_crit * agg["final_regret_std"] / np.sqrt(n)
    agg["total_runtime_ci95"] = t_crit * agg["total_runtime_std"] / np.sqrt(n)
    agg["total_cpu_time_ci95"] = t_crit * agg["total_cpu_time_std"] / np.sqrt(n)
    agg["projection_events_ci95"] = t_crit * agg["projection_events_std"] / np.sqrt(n)
    return agg


def run_experiment(cfg: dict[str, Any], output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dir(output_dir)

    ecfg = cfg["experiment"]
    dcfg = cfg["dataset"]
    horizon = int(ecfg["horizon"])
    progress_every = int(ecfg.get("progress_every", 0))
    save_round_metrics = bool(cfg.get("runtime", {}).get("save_round_metrics", True))
    seeds = [int(s) for s in ecfg["seeds"]]

    subopt_prob = float(cfg.get("user_policy", {}).get("subopt_prob", 0.0))
    if not (0.0 <= subopt_prob <= 1.0):
        raise ValueError(f"user_policy.subopt_prob must be in [0, 1], got {subopt_prob}")
    subopt_mode = str(cfg.get("user_policy", {}).get("subopt_mode", "uniform"))
    subopt_noise_std = float(cfg.get("user_policy", {}).get("subopt_noise_std", 0.0))

    action_set = build_action_set(
        action_dim=int(dcfg["action_dim"]),
        m_select=int(dcfg["m_select"]),
        scale=float(dcfg.get("action_scale", 1.0)),
    )

    round_rows: list[RoundMetric] = []
    seed_rows: list[SeedSummary] = []

    for seed in seeds:
        generator = SyntheticGenerator(cfg, seed)
        # Pre-generate all rounds before running any learner: this ensures
        # every algorithm sees exactly the same context/score sequence for a
        # given seed, making per-algorithm comparisons fair.
        round_data = [generator.sample_round() for _ in range(horizon)]

        learners = build_learners(cfg)
        for learner in learners:
            # Reset the independent feedback stream for every learner. This
            # preserves each separate-target run and removes order dependence.
            user_rng = np.random.default_rng(seed + 100003) if subopt_prob > 0.0 else None
            print(f"[run] seed={seed} algo={learner.name}")
            rr, ss = _run_one(
                seed=seed,
                learner=learner,
                round_data=round_data,
                action_set=action_set,
                progress_every=progress_every,
                save_round_metrics=save_round_metrics,
                subopt_prob=subopt_prob,
                subopt_mode=subopt_mode,
                subopt_noise_std=subopt_noise_std,
                user_rng=user_rng,
            )
            round_rows.extend(rr)
            seed_rows.append(ss)

    round_df = _to_df_round(round_rows)
    seed_df = _to_df_summary(seed_rows)
    agg_df = _aggregate_seed_summary(seed_df)

    if save_round_metrics:
        round_df.to_csv(output_dir / "round_metrics.csv", index=False)
    seed_df.to_csv(output_dir / "seed_summary.csv", index=False)
    agg_df.to_csv(output_dir / "aggregate_summary.csv", index=False)

    return round_df, seed_df, agg_df
