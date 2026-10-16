#!/usr/bin/env python3
"""Combined suboptimality figure: one_swap (prob sweep) + score_perturb_relative (relative-noise sweep).

Runs both sweeps with fixed best-c regularisation and produces a single
side-by-side figure:
  Left panel  : Final regret vs subopt_prob   (one_swap mode)
  Right panel : Final regret vs relative noise level (score_perturb_relative mode)

Produces:
  <output-root>/one_swap/       -- per-run seed_summary.csv
  <output-root>/score_perturb/  -- per-run seed_summary.csv
  <figure-root>/subopt_combined.png
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
    p = argparse.ArgumentParser(
        description="Run one_swap + score_perturb sweeps and produce combined figure."
    )
    p.add_argument(
        "--config", type=Path, default=Path("configs/main_linear_t10000_s10.yaml"),
        help="Experiment config (default: configs/main_linear_t10000_s10.yaml).",
    )
    p.add_argument(
        "--subopt-prob-grid",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="subopt_prob grid for one_swap sweep.",
    )
    p.add_argument(
        "--subopt-noise-std-grid",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="subopt_noise_std grid for score_perturb_relative sweep.",
    )
    p.add_argument(
        "--targets",
        type=str,
        default="corectron,ons,ogd",
        help="Comma-separated algorithms (default: corectron,ons,ogd).",
    )
    p.add_argument("--best-coeff-corectron", type=float, default=_DEFAULT_BEST_COEFFS["corectron"])
    p.add_argument("--best-coeff-ons", type=float, default=_DEFAULT_BEST_COEFFS["ons"])
    p.add_argument("--best-coeff-ogd", type=float, default=_DEFAULT_BEST_COEFFS["ogd"])
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


def _parse_floats(raw: str) -> list[float]:
    return parse_grid(raw, "sweep grid")

def _best_coeffs(args: argparse.Namespace) -> dict[str, float]:
    return {
        "corectron": finite_float(args.best_coeff_corectron, "--best-coeff-corectron", positive=True),
        "ons": finite_float(args.best_coeff_ons, "--best-coeff-ons", positive=True),
        "ogd": finite_float(args.best_coeff_ogd, "--best-coeff-ogd", positive=True),
    }


def _target_only(raw_cfg: dict[str, Any], target: str) -> dict[str, Any]:
    return target_only(raw_cfg, target)

def _resolve_one_swap(
    raw_cfg: dict[str, Any],
    target: str,
    best_c: float,
    subopt_prob: float,
    cli_ons_lr: float | None,
) -> dict[str, Any]:
    base = resolve_config(raw=raw_cfg, cli_ons_lr=cli_ons_lr, cli_horizon=None, cli_seeds=None)
    eps_ons_base = float(base["learners"].get("ons", {}).get("epsilon_theory", 0.0))
    lam_core_base = float(base["learners"].get("corectron", {}).get("lambda_reg_theory", 0.0))
    step_base = float(base["learners"].get("ogd", {}).get("ogd_lr_theory", 0.0))

    run_cfg = _target_only(raw_cfg, target)
    return resolve_config(
        raw=run_cfg,
        cli_ons_lr=cli_ons_lr,
        cli_horizon=None,
        cli_seeds=None,
        cli_ons_epsilon=best_c * eps_ons_base if target == "ons" else None,
        cli_corectron_lambda=best_c * lam_core_base if target == "corectron" else None,
        cli_ogd_step0=step_base / best_c if target == "ogd" else None,
        cli_subopt_prob=subopt_prob,
        cli_subopt_mode="one_swap",
    )


def _resolve_score_perturb(
    raw_cfg: dict[str, Any],
    target: str,
    best_c: float,
    noise_std: float,
    cli_ons_lr: float | None,
) -> dict[str, Any]:
    base = resolve_config(raw=raw_cfg, cli_ons_lr=cli_ons_lr, cli_horizon=None, cli_seeds=None)
    eps_ons_base = float(base["learners"].get("ons", {}).get("epsilon_theory", 0.0))
    lam_core_base = float(base["learners"].get("corectron", {}).get("lambda_reg_theory", 0.0))
    step_base = float(base["learners"].get("ogd", {}).get("ogd_lr_theory", 0.0))

    run_cfg = _target_only(raw_cfg, target)
    return resolve_config(
        raw=run_cfg,
        cli_ons_lr=cli_ons_lr,
        cli_horizon=None,
        cli_seeds=None,
        cli_ons_epsilon=best_c * eps_ons_base if target == "ons" else None,
        cli_corectron_lambda=best_c * lam_core_base if target == "corectron" else None,
        cli_ogd_step0=step_base / best_c if target == "ogd" else None,
        cli_subopt_prob=1.0,
        cli_subopt_mode="score_perturb_relative",
        cli_subopt_noise_std=noise_std,
    )


def _val_tag(v: float) -> str:
    return value_tag(v)


def _run_sweep(
    raw_cfg: dict[str, Any],
    targets: list[str],
    x_vals: list[float],
    best_c: dict[str, float],
    resolve_fn,
    output_root: Path,
    run_tag: str,
    x_col: str,
    force: bool,
) -> pd.DataFrame:
    """Generic sweep runner; returns per-seed DataFrame with columns
    [algorithm, <x_col>, seed, final_regret, cumulative_suboptimality]."""
    all_rows: list[dict] = []
    total = len(targets) * len(x_vals)
    job = 0
    for target in targets:
        for xv in x_vals:
            job += 1
            run_dir = output_root / f"{target}_{run_tag}{_val_tag(xv)}"
            ensure_dir(run_dir)
            resolved = resolve_fn(raw_cfg, target, best_c[target], xv)
            print(f"[job] {job}/{total} target={target} {x_col}={xv} c={best_c[target]}")
            _, seed_df, _ = run_or_resume(resolved, run_dir, force=force)

            for _, row in seed_df.iterrows():
                all_rows.append({
                    "algorithm": target,
                    x_col: xv,
                    "seed": int(row["seed"]),
                    "final_regret": float(row["final_regret"]),
                    "cumulative_suboptimality": float(row["cumulative_suboptimality"]),
                })

    return pd.DataFrame(all_rows)


def _aggregate(df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for (algo, xv), grp in df.groupby(["algorithm", x_col]):
        r = grp["final_regret"].to_numpy()
        n = len(r)
        t_crit = float(sp_stats.t.ppf(0.975, n - 1))
        rows.append({
            "algorithm": algo,
            x_col: float(xv),
            "final_regret_mean": float(r.mean()),
            "final_regret_ci95": t_crit * float(r.std(ddof=1)) / np.sqrt(n) if n > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


_CURVE_FONT_SCALE = 1.5
_LABEL_FS = 16.0 * _CURVE_FONT_SCALE   # 24
_TITLE_FS = _LABEL_FS                   # 24
_YTICK_FS = 12.5 * _CURVE_FONT_SCALE   # 18.75
_XTICK_FS = _YTICK_FS                   # same as ytick
_LEGEND_FS = 13.5 * _CURVE_FONT_SCALE  # 20.25


def _draw_panel(
    ax,
    agg: pd.DataFrame,
    x_col: str,
    xlabel: str,
    title: str,
    algos: list[str],
    show_ylabel: bool = False,
) -> list:
    """Draw one panel; returns list of (Line2D, label) for figure-level legend."""
    handles = []
    for algo in algos:
        sub = agg[agg["algorithm"] == algo].sort_values(x_col)
        if sub.empty:
            continue
        x = sub[x_col].to_numpy()
        y = sub["final_regret_mean"].to_numpy()
        ci = sub["final_regret_ci95"].to_numpy()
        line, = ax.plot(
            x, y,
            marker="o",
            linewidth=1.8,
            markersize=4.8,
            label=DISPLAY_ALGO.get(algo, algo),
            color=ALGO_COLOR.get(algo),
        )
        ax.fill_between(x, y - ci, y + ci, alpha=0.16, linewidth=0.0,
                        color=ALGO_COLOR.get(algo))
        handles.append(line)

    ax.set_xlabel(xlabel, fontsize=_LABEL_FS, labelpad=14)
    if show_ylabel:
        ax.set_ylabel("Final regret", fontsize=_LABEL_FS, labelpad=14)
    ax.set_title(title, pad=16, fontsize=_TITLE_FS)
    ax.tick_params(axis="x", labelsize=_XTICK_FS)
    ax.tick_params(axis="y", labelsize=_YTICK_FS)
    ax.grid(alpha=0.25, which="both")
    return handles


def render_combined_figure(
    df_one_swap: pd.DataFrame,
    df_score_perturb: pd.DataFrame,
    figure_root: Path,
    targets: list[str] | None = None,
) -> None:
    """Draw Figure 5 from existing per-seed summaries; no experiments are run."""
    if targets is None:
        targets = list(ALL_TARGETS)
    agg_one_swap = _aggregate(df_one_swap, "subopt_prob")
    agg_score_perturb = _aggregate(df_score_perturb, "noise_std")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))

    handles = _draw_panel(
        axes[0],
        agg=agg_one_swap,
        x_col="subopt_prob",
        xlabel=r"Swap probability $\alpha$",
        title=r"One-swap",
        algos=targets,
        show_ylabel=True,
    )
    _draw_panel(
        axes[1],
        agg=agg_score_perturb,
        x_col="noise_std",
        xlabel=r"Relative noise level $\xi$",
        title=r"Score-perturb",
        algos=targets,
        show_ylabel=False,
    )

    # Figure-level legend at top (horizontal, centered)
    labels = [DISPLAY_ALGO.get(t, t) for t in targets]
    fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(targets),
        frameon=True,
        fontsize=_LEGEND_FS,
    )

    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.74, wspace=0.35)
    out_path = figure_root / "subopt_combined.png"
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"\n[write] {out_path}")


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    targets = _parse_targets(args.targets)
    probs = parse_grid(args.subopt_prob_grid, "--subopt-prob-grid", probability=True)
    noise_stds = _parse_floats(args.subopt_noise_std_grid)
    best_c = _best_coeffs(args)

    ensure_dir(args.output_root)
    ensure_dir(args.figure_root)

    one_swap_root = args.output_root / "one_swap"
    score_perturb_root = args.output_root / "score_perturb"
    ensure_dir(one_swap_root)
    ensure_dir(score_perturb_root)

    cli_ons_lr = args.ons_lr

    # --- one_swap sweep ---
    print("\n=== one_swap sweep ===")
    df_one_swap = _run_sweep(
        raw_cfg=raw_cfg,
        targets=targets,
        x_vals=probs,
        best_c=best_c,
        resolve_fn=lambda cfg, t, c, p: _resolve_one_swap(cfg, t, c, p, cli_ons_lr),
        output_root=one_swap_root,
        run_tag="p",
        x_col="subopt_prob",
        force=args.force,
    )
    df_one_swap.to_csv(args.output_root / "one_swap_summary.csv", index=False)

    # --- score_perturb sweep ---
    print("\n=== score_perturb sweep ===")
    df_score_perturb = _run_sweep(
        raw_cfg=raw_cfg,
        targets=targets,
        x_vals=noise_stds,
        best_c=best_c,
        resolve_fn=lambda cfg, t, c, s: _resolve_score_perturb(cfg, t, c, s, cli_ons_lr),
        output_root=score_perturb_root,
        run_tag="sigma",
        x_col="noise_std",
        force=args.force,
    )
    df_score_perturb.to_csv(args.output_root / "score_perturb_summary.csv", index=False)

    render_combined_figure(df_one_swap, df_score_perturb, args.figure_root, targets)


if __name__ == "__main__":
    main()
