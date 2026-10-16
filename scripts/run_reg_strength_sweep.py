#!/usr/bin/env python3
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

ROOT = SCRIPT_ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments_core.config import load_config, resolve_config
from experiments_core.reproducibility import run_or_resume
from experiments_core.validation import parse_grid, value_tag, target_only
from experiments_core.utils import ensure_dir

ALL_TARGETS = ("ons", "corectron", "ogd", "corectron_rbf", "kons")
DISPLAY_ALGO = {
    "ons": "ons",
    "corectron": "corectron",
    "ogd": "ogd",
    "corectron_rbf": "corectron-kernel",
    "kons": "kons",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regularization-strength sweep.")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--coeff-grid", type=str, default="0.001,0.01,0.1,1,10,100,1000")
    p.add_argument("--targets", type=str, default="ons,corectron,ogd")
    p.add_argument("--ons-lr", type=float, default=None)
    p.add_argument("--kons-lr", type=float, default=None)
    p.add_argument("--kernel-type", type=str, default=None, choices=["rbf", "poly"])
    p.add_argument("--kernel-sigma", type=float, default=None)
    p.add_argument("--kernel-degree", type=int, default=None)
    p.add_argument("--kernel-coef0", type=float, default=None)
    p.add_argument("--kernel-alpha", type=float, default=None)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--figure-root", type=Path, required=True)
    p.add_argument("--include-cpu-plot", action="store_true")
    p.add_argument("--subopt-prob", type=float, default=None,
                   help="Probability of a suboptimal user action (default: 0.0 = optimal).")
    p.add_argument("--subopt-mode", type=str, default=None,
                   choices=["uniform", "one_swap"],
                   help="How the suboptimal action is generated (default: uniform).")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def parse_targets(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--targets is empty")
    bad = [v for v in vals if v not in ALL_TARGETS]
    if bad:
        raise ValueError(f"Unknown targets: {bad}; allowed={ALL_TARGETS}")
    out: list[str] = []
    seen: set[str] = set()
    for v in vals:
        if v in seen:
            continue
        out.append(v)
        seen.add(v)
    return out


def parse_coeffs(raw: str) -> list[float]:
    return parse_grid(raw, "--coeff-grid", positive=True)


def _coeff_tag(c: float) -> str:
    return value_tag(c)


def _target_only(raw_cfg: dict[str, Any], target: str) -> dict[str, Any]:
    return target_only(raw_cfg, target)


def _reg_value_from_cfg(cfg: dict[str, Any], target: str) -> float:
    if target == "ons":
        return float(cfg["learners"]["ons"]["epsilon"])
    if target == "corectron":
        return float(cfg["learners"]["corectron"]["lambda_reg"])
    if target == "ogd":
        return 1.0 / float(cfg["learners"]["ogd"]["step0"])
    if target == "corectron_rbf":
        return float(cfg["learners"]["corectron_rbf"]["lambda_reg"])
    if target == "kons":
        return float(cfg["learners"]["kons"]["epsilon"])
    raise ValueError(f"Unknown target: {target}")


def _plot_metric(df: pd.DataFrame, metric: str, y_label: str, title: str, out_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    for algo, sub in df.groupby("algorithm"):
        sub = sub.sort_values("coeff")
        x = sub["coeff"].to_numpy(dtype=float)
        y = sub[metric].to_numpy(dtype=float)
        plt.plot(x, y, marker="o", label=DISPLAY_ALGO.get(algo, algo))
    plt.xscale("log")
    plt.xlabel("coefficient c")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(alpha=0.3, which="both")
    plt.legend()
    plt.tight_layout()
    ensure_dir(out_path.parent)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    raw_cfg = load_config(args.config)
    targets = parse_targets(args.targets)
    coeffs = parse_coeffs(args.coeff_grid)

    ensure_dir(args.output_root)
    ensure_dir(args.figure_root)

    base_resolved = resolve_config(
        raw=raw_cfg,
        cli_ons_lr=args.ons_lr,
        cli_horizon=None,
        cli_seeds=None,
        cli_ons_epsilon=None,
        cli_corectron_lambda=None,
        cli_ogd_step0=None,
        cli_kons_lr=args.kons_lr,
        cli_corectron_rbf_lambda=None,
        cli_kons_epsilon=None,
        cli_kernel_type=args.kernel_type,
        cli_kernel_sigma=args.kernel_sigma,
        cli_kernel_degree=args.kernel_degree,
        cli_kernel_coef0=args.kernel_coef0,
        cli_kernel_alpha=args.kernel_alpha,
        cli_subopt_prob=args.subopt_prob,
        cli_subopt_mode=args.subopt_mode,
    )

    eps_ons_base = float(base_resolved["learners"].get("ons", {}).get("epsilon_theory", 0.0))
    lam_core_base = float(base_resolved["learners"].get("corectron", {}).get("lambda_reg_theory", 0.0))
    step_base = float(base_resolved["learners"].get("ogd", {}).get("ogd_lr_theory", 0.0))
    lam_rbf_base = float(base_resolved["learners"].get("corectron_rbf", {}).get("lambda_reg_theory", 0.0))
    eps_kons_base = float(base_resolved["learners"].get("kons", {}).get("epsilon_theory", 0.0))

    print(
        "[baseline] "
        f"config={args.config} horizon={base_resolved['experiment']['horizon']} "
        f"seeds={base_resolved['experiment']['seeds']} "
        f"targets={targets} coeffs={coeffs} "
        f"ons_lr={base_resolved['learners'].get('ons', {}).get('ons_lr', 'n/a')} "
        f"kons_lr={base_resolved['learners'].get('kons', {}).get('kons_lr', 'n/a')} "
        f"eps_ons_base={eps_ons_base:.8g} lam_core_base={lam_core_base:.8g} "
        f"step_base={step_base:.8g} lam_rbf_base={lam_rbf_base:.8g} eps_kons_base={eps_kons_base:.8g}"
    )

    rows: list[dict[str, Any]] = []
    total_jobs = len(targets) * len(coeffs)
    job = 0

    for target in targets:
        for c in coeffs:
            job += 1
            run_dir = args.output_root / f"{target}_c{_coeff_tag(c)}"
            ensure_dir(run_dir)
            run_cfg = _target_only(raw_cfg, target)

            cli_eps_ons = None
            cli_lam_core = None
            cli_step0 = None
            cli_lam_rbf = None
            cli_eps_kons = None

            if target == "ons":
                cli_eps_ons = c * eps_ons_base
            elif target == "corectron":
                cli_lam_core = c * lam_core_base
            elif target == "ogd":
                cli_step0 = step_base / c
            elif target == "corectron_rbf":
                cli_lam_rbf = c * lam_rbf_base
            elif target == "kons":
                cli_eps_kons = c * eps_kons_base
            else:
                raise ValueError(target)

            resolved = resolve_config(
                raw=run_cfg,
                cli_ons_lr=args.ons_lr,
                cli_horizon=None,
                cli_seeds=None,
                cli_ons_epsilon=cli_eps_ons,
                cli_corectron_lambda=cli_lam_core,
                cli_ogd_step0=cli_step0,
                cli_kons_lr=args.kons_lr,
                cli_corectron_rbf_lambda=cli_lam_rbf,
                cli_kons_epsilon=cli_eps_kons,
                cli_kernel_type=args.kernel_type,
                cli_kernel_sigma=args.kernel_sigma,
                cli_kernel_degree=args.kernel_degree,
                cli_kernel_coef0=args.kernel_coef0,
                cli_kernel_alpha=args.kernel_alpha,
                cli_subopt_prob=args.subopt_prob,
                cli_subopt_mode=args.subopt_mode,
            )
            print(f"[job] {job}/{total_jobs} target={target} c={c} "
                  f"reg_value={_reg_value_from_cfg(resolved, target):.8g}")
            _, _, agg_df = run_or_resume(resolved, run_dir, force=args.force)

            row = agg_df[agg_df["algorithm"] == target]
            if row.empty:
                raise ValueError(f"Missing row for target={target} c={c}")
            r = row.iloc[0]
            rows.append(
                {
                    "algorithm": target,
                    "coeff": float(c),
                    "num_seeds": int(r["num_seeds"]),
                    "reg_value": float(_reg_value_from_cfg(resolved, target)),
                    "final_regret_mean": float(r["final_regret_mean"]),
                    "final_regret_ci95": float(r["final_regret_ci95"]),
                    "total_runtime_mean": float(r["total_runtime_mean"]),
                    "total_runtime_ci95": float(r["total_runtime_ci95"]),
                    "total_cpu_time_mean": float(r.get("total_cpu_time_mean", np.nan)),
                    "total_cpu_time_ci95": float(r.get("total_cpu_time_ci95", np.nan)),
                    "projection_events_mean": float(r.get("projection_events_mean", 0.0)),
                    "projection_events_ci95": float(r.get("projection_events_ci95", 0.0)),
                }
            )

    result_df = pd.DataFrame(rows).sort_values(["algorithm", "coeff"]).reset_index(drop=True)
    out_csv = args.output_root / "sensitivity_summary.csv"
    result_df.to_csv(out_csv, index=False)

    _plot_metric(
        result_df,
        metric="final_regret_mean",
        y_label="Final cumulative regret",
        title="Regret vs coefficient",
        out_path=args.figure_root / "regret_vs_coeff.png",
    )
    _plot_metric(
        result_df,
        metric="total_runtime_mean",
        y_label="Total runtime (sec)",
        title="Runtime vs coefficient",
        out_path=args.figure_root / "runtime_vs_coeff.png",
    )
    if args.include_cpu_plot:
        _plot_metric(
            result_df,
            metric="total_cpu_time_mean",
            y_label="Total CPU time (sec)",
            title="CPU time vs coefficient",
            out_path=args.figure_root / "cpu_time_vs_coeff.png",
        )

    print(f"[write] {out_csv}")
    print(f"[write] {args.figure_root / 'regret_vs_coeff.png'}")
    print(f"[write] {args.figure_root / 'runtime_vs_coeff.png'}")
    if args.include_cpu_plot:
        print(f"[write] {args.figure_root / 'cpu_time_vs_coeff.png'}")


if __name__ == "__main__":
    main()
