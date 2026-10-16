#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from experiments_core.config import hyperparam_print_blob, load_config, resolve_config
from experiments_core.reproducibility import run_or_resume
from experiments_core.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run contextual recommendation experiments.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config.")
    parser.add_argument("--output", type=Path, default=None, help="Output directory.")
    parser.add_argument("--ons-lr", type=float, default=None, help="Override ONS learning rate.")
    parser.add_argument("--ons-epsilon", type=float, default=None, help="Override ONS epsilon.")
    parser.add_argument("--corectron-lambda", type=float, default=None, help="Override CoRectron lambda.")
    parser.add_argument("--ogd-step0", type=float, default=None, help="Override OGD step0.")
    parser.add_argument("--kons-lr", type=float, default=None, help="Override KONS learning rate.")
    parser.add_argument("--kons-epsilon", type=float, default=None, help="Override KONS epsilon.")
    parser.add_argument(
        "--corectron-rbf-lambda",
        type=float,
        default=None,
        help="Override kernel CoRectron lambda.",
    )
    parser.add_argument("--kernel-type", type=str, default=None, choices=["rbf", "poly"], help="Override dataset kernel type.")
    parser.add_argument("--kernel-sigma", type=float, default=None, help="Override dataset.kernel.sigma_true.")
    parser.add_argument("--kernel-degree", type=int, default=None, help="Override dataset.kernel.degree.")
    parser.add_argument("--kernel-coef0", type=float, default=None, help="Override dataset.kernel.coef0.")
    parser.add_argument("--kernel-alpha", type=float, default=None, help="Override dataset.kernel.alpha.")
    parser.add_argument("--horizon", type=int, default=None, help="Override horizon T.")
    parser.add_argument("--seeds", type=str, default=None, help="Override seeds as comma list.")
    parser.add_argument("--force", action="store_true", help="Replace existing outputs in --output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raw_cfg = load_config(args.config)
    cfg = resolve_config(
        raw_cfg,
        cli_ons_lr=args.ons_lr,
        cli_horizon=args.horizon,
        cli_seeds=args.seeds,
        cli_ons_epsilon=args.ons_epsilon,
        cli_corectron_lambda=args.corectron_lambda,
        cli_ogd_step0=args.ogd_step0,
        cli_kons_lr=args.kons_lr,
        cli_corectron_rbf_lambda=args.corectron_rbf_lambda,
        cli_kons_epsilon=args.kons_epsilon,
        cli_kernel_type=args.kernel_type,
        cli_kernel_sigma=args.kernel_sigma,
        cli_kernel_degree=args.kernel_degree,
        cli_kernel_coef0=args.kernel_coef0,
        cli_kernel_alpha=args.kernel_alpha,
    )

    run_name = cfg["experiment"].get("name", "run")
    if args.output is None:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("results") / f"{run_name}_{ts}"
    else:
        output_dir = args.output
    ensure_dir(output_dir)

    print("[config] Resolved configuration:")
    print(hyperparam_print_blob(cfg))

    round_df, seed_df, agg_df = run_or_resume(cfg, output_dir, force=args.force)

    print("[done] Output directory:", output_dir)
    print("[done] Files written:")
    if not round_df.empty:
        print(f"- round_metrics.csv ({len(round_df)} rows)")
    print(f"- seed_summary.csv ({len(seed_df)} rows)")
    print(f"- aggregate_summary.csv ({len(agg_df)} rows)")

    print("[summary] Aggregate results:")
    print(agg_df.to_string(index=False))


if __name__ == "__main__":
    main()
