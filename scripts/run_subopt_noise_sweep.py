#!/usr/bin/env python3
"""Score-perturbation sweep for CoRectron robustness experiment.

For each value of subopt_noise_std sigma in --subopt-noise-std-grid, runs the
experiment with:
  - subopt_mode = "score_perturb_relative"
  - subopt_prob = 1.0  (perturb every round)
  - fixed best-c regularisation coefficients

and collects (R_T, Delta_T) per seed.

Produces:
  <output-root>/subopt_noise_summary.csv   -- per-seed rows
  <figure-root>/regret_vs_delta.png        -- R_T vs Delta_T
  <figure-root>/regret_vs_noise_std.png    -- R_T mean vs sigma
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = SCRIPT_ROOT / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = SCRIPT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_core.config import load_config, resolve_config
from experiments_core.reproducibility import run_or_resume
from experiments_core.validation import finite_float, parse_grid, value_tag, target_only
from experiments_core.utils import ensure_dir

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.titlesize": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5,
    "legend.fontsize": 13.5,
    "figure.titlesize": 16,
})

_DEFAULT_BEST_COEFFS = {
    "corectron": 0.001,
    "ons": 0.1,
    "ogd": 1.0,
}

ALL_TARGETS = ("corectron", "ons", "ogd")

DISPLAY_ALGO = {
    "corectron": "CoRectron-L",
    "ons": "ONS",
    "ogd": "OGD",
}

ALGO_COLOR = {
    "corectron": "#ff7f0e",
    "ons": "#1f77b4",
    "ogd": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Relative score-noise sweep (fixed best-c).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--subopt-noise-std-grid",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated list of subopt_noise_std values (default: 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0).",
    )
    p.add_argument(
        "--subopt-prob",
        type=float,
        default=1.0,
        help="Probability of applying score perturbation each round (default: 1.0).",
    )
    p.add_argument(
        "--targets",
        type=str,
        default="corectron,ons,ogd",
        help="Comma-separated algorithms to evaluate (default: corectron,ons,ogd).",
    )
    p.add_argument(
        "--best-coeff-corectron", type=float, default=_DEFAULT_BEST_COEFFS["corectron"],
        help=f"Best c for corectron (default: {_DEFAULT_BEST_COEFFS['corectron']}).",
    )
    p.add_argument(
        "--best-coeff-ons", type=float, default=_DEFAULT_BEST_COEFFS["ons"],
        help=f"Best c for ons (default: {_DEFAULT_BEST_COEFFS['ons']}).",
    )
    p.add_argument(
        "--best-coeff-ogd", type=float, default=_DEFAULT_BEST_COEFFS["ogd"],
        help=f"Best c for ogd (default: {_DEFAULT_BEST_COEFFS['ogd']}).",
    )
    p.add_argument("--ons-lr", type=float, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--figure-root", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _parse_targets(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--targets must not be empty")
    bad = [v for v in vals if v not in ALL_TARGETS]
    if bad:
        raise ValueError(f"Unknown targets: {bad}; allowed={list(ALL_TARGETS)}")
    seen: set[str] = set()
    out: list[str] = []
    for v in vals:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _parse_noise_stds(raw: str) -> list[float]:
    return parse_grid(raw, "--subopt-noise-std-grid")

def _target_only(raw_cfg: dict[str, Any], target: str) -> dict[str, Any]:
    return target_only(raw_cfg, target)

def _best_coeffs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "corectron": finite_float(args.best_coeff_corectron, "--best-coeff-corectron", positive=True),
        "ons": finite_float(args.best_coeff_ons, "--best-coeff-ons", positive=True),
        "ogd": finite_float(args.best_coeff_ogd, "--best-coeff-ogd", positive=True),
    }


def _resolve_with_best_c(
    raw_cfg: dict[str, Any],
    target: str,
    best_c: float,
    subopt_prob: float,
    noise_std: float,
    cli_ons_lr: float | None,
) -> dict[str, Any]:
    base = resolve_config(
        raw=raw_cfg,
        cli_ons_lr=cli_ons_lr,
        cli_horizon=None,
        cli_seeds=None,
    )
    eps_ons_base = float(base["learners"].get("ons", {}).get("epsilon_theory", 0.0))
    lam_core_base = float(base["learners"].get("corectron", {}).get("lambda_reg_theory", 0.0))
    step_base = float(base["learners"].get("ogd", {}).get("ogd_lr_theory", 0.0))

    cli_eps_ons = best_c * eps_ons_base if target == "ons" else None
    cli_lam_core = best_c * lam_core_base if target == "corectron" else None
    cli_step0 = step_base / best_c if target == "ogd" else None

    run_cfg = _target_only(raw_cfg, target)
    return resolve_config(
        raw=run_cfg,
        cli_ons_lr=cli_ons_lr,
        cli_horizon=None,
        cli_seeds=None,
        cli_ons_epsilon=cli_eps_ons,
        cli_corectron_lambda=cli_lam_core,
        cli_ogd_step0=cli_step0,
        cli_subopt_prob=subopt_prob,
        cli_subopt_mode="score_perturb_relative",
        cli_subopt_noise_std=noise_std,
    )


def _noise_tag(sigma: float) -> str:
    return value_tag(sigma)


def _plot_regret_vs_delta(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))

    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        ax.scatter(
            sub["cumulative_suboptimality"],
            sub["final_regret"],
            label=DISPLAY_ALGO.get(algo, algo),
            color=ALGO_COLOR.get(algo, None),
            alpha=0.75,
            s=40,
        )

    ax.set_xlabel(r"Cumulative suboptimality $\Delta_T$")
    ax.set_ylabel(r"Final regret $R_T$")
    ax.set_title(r"$R_T$ vs $\Delta_T$ — score perturbation (each point = one seed)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _plot_regret_vs_noise_std(agg: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    for algo in agg["algorithm"].unique():
        sub = agg[agg["algorithm"] == algo].sort_values("noise_std")
        x = sub["noise_std"].to_numpy()
        y = sub["final_regret_mean"].to_numpy()
        ci = sub["final_regret_ci95"].to_numpy()
        ax.plot(x, y, marker="o", label=DISPLAY_ALGO.get(algo, algo),
                color=ALGO_COLOR.get(algo, None))
        ax.fill_between(x, y - ci, y + ci, alpha=0.2, color=ALGO_COLOR.get(algo, None))

    ax.set_xlabel(r"Relative noise level $\xi$")
    ax.set_ylabel(r"Final regret $R_T$ (mean $\pm$ 95% CI)")
    ax.set_title(r"$R_T$ vs relative noise level")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    targets = _parse_targets(args.targets)
    noise_stds = _parse_noise_stds(args.subopt_noise_std_grid)
    best_c = _best_coeffs(args)

    ensure_dir(args.output_root)
    ensure_dir(args.figure_root)

    all_seed_rows: list[dict] = []
    total_jobs = len(targets) * len(noise_stds)
    job = 0

    for target in targets:
        for sigma in noise_stds:
            job += 1
            run_dir = args.output_root / f"{target}_sigma{_noise_tag(sigma)}"
            ensure_dir(run_dir)
            resolved = _resolve_with_best_c(raw_cfg, target, best_c[target], args.subopt_prob, sigma, args.ons_lr)
            print(f"[job] {job}/{total_jobs} target={target} sigma={sigma} c={best_c[target]}")
            _, seed_df, _ = run_or_resume(resolved, run_dir, force=args.force)

            for _, row in seed_df.iterrows():
                all_seed_rows.append({
                    "algorithm": target,
                    "noise_std": sigma,
                    "seed": int(row["seed"]),
                    "final_regret": float(row["final_regret"]),
                    "cumulative_suboptimality": float(row["cumulative_suboptimality"]),
                })

    summary_df = pd.DataFrame(all_seed_rows)
    out_csv = args.output_root / "subopt_noise_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}")

    agg_rows: list[dict] = []
    for (algo, sigma), grp in summary_df.groupby(["algorithm", "noise_std"]):
        r = grp["final_regret"].to_numpy()
        n = len(r)
        t_crit = float(sp_stats.t.ppf(0.975, n - 1))
        agg_rows.append({
            "algorithm": algo,
            "noise_std": float(sigma),
            "final_regret_mean": float(r.mean()),
            "final_regret_ci95": t_crit * float(r.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan"),
        })
    agg_df = pd.DataFrame(agg_rows)

    _plot_regret_vs_delta(
        summary_df,
        args.figure_root / "regret_vs_delta.png",
    )
    _plot_regret_vs_noise_std(agg_df, args.figure_root / "regret_vs_noise_std.png")
    print(f"[write] {args.figure_root / 'regret_vs_delta.png'}")
    print(f"[write] {args.figure_root / 'regret_vs_noise_std.png'}")


if __name__ == "__main__":
    main()
