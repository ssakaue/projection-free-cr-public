#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parent
if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = SCRIPT_ROOT / ".mplconfig"
    _mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as _stats

DISPLAY_ALGO = {
    "ons": "ons",
    "corectron": "corectron",
    "ogd": "ogd",
    "corectron_rbf": "corectron-kernel",
    "kons": "kons",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot experiment outputs.")
    parser.add_argument("--input", type=Path, required=True, help="Run output directory.")
    parser.add_argument("--output", type=Path, required=True, help="Figure output directory.")
    return parser.parse_args()


def _plot_regret(round_df: pd.DataFrame, outdir: Path) -> None:
    g = (
        round_df.groupby(["algorithm", "round_id"], as_index=False)
        .agg(
            mean_regret=("cumulative_regret", "mean"),
            std_regret=("cumulative_regret", "std"),
            n=("seed", "count"),
        )
        .sort_values(["algorithm", "round_id"])
    )
    t_crit = g["n"].apply(lambda ni: _stats.t.ppf(0.975, int(ni) - 1))
    g["ci"] = t_crit * g["std_regret"] / np.sqrt(g["n"])

    plt.figure(figsize=(8, 5))
    for algo, sub in g.groupby("algorithm"):
        x = sub["round_id"].to_numpy()
        y = sub["mean_regret"].to_numpy()
        ci = sub["ci"].to_numpy()
        plt.plot(x, y, label=DISPLAY_ALGO.get(algo, algo))
        plt.fill_between(x, y - ci, y + ci, alpha=0.2)

    plt.xlabel("Round")
    plt.ylabel("Cumulative regret")
    plt.title("Regret vs round")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "regret_vs_round.png", dpi=170)
    plt.close()


def _plot_projection_events(seed_df: pd.DataFrame, outdir: Path) -> None:
    # OGD always projects onto the Euclidean ball (cheap O(d) operation) but
    # does not track projection_events; omit it from this count-focused plot.
    df = seed_df[seed_df["algorithm"] != "ogd"]
    g = (
        df.groupby("algorithm", as_index=False)
        .agg(
            mean=("projection_events", "mean"),
            std=("projection_events", "std"),
            n=("seed", "count"),
        )
        .sort_values("algorithm")
    )
    t_crit = g["n"].apply(lambda ni: _stats.t.ppf(0.975, int(ni) - 1))
    g["ci"] = t_crit * g["std"] / np.sqrt(g["n"])

    x = np.arange(len(g))
    plt.figure(figsize=(7, 4))
    plt.bar(x, g["mean"].to_numpy(), yerr=g["ci"].to_numpy(), capsize=4)
    labels = [DISPLAY_ALGO.get(a, a) for a in g["algorithm"].tolist()]
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Projection events")
    plt.title("Projection event count")
    plt.tight_layout()
    plt.savefig(outdir / "projection_events.png", dpi=170)
    plt.close()


def _plot_runtime(seed_df: pd.DataFrame, outdir: Path) -> None:
    g = (
        seed_df.groupby("algorithm", as_index=False)
        .agg(
            mean=("total_runtime_sec", "mean"),
            std=("total_runtime_sec", "std"),
            n=("seed", "count"),
        )
        .sort_values("algorithm")
    )
    t_crit = g["n"].apply(lambda ni: _stats.t.ppf(0.975, int(ni) - 1))
    g["ci"] = t_crit * g["std"] / np.sqrt(g["n"])

    x = np.arange(len(g))
    plt.figure(figsize=(7, 4))
    plt.bar(x, g["mean"].to_numpy(), yerr=g["ci"].to_numpy(), capsize=4)
    labels = [DISPLAY_ALGO.get(a, a) for a in g["algorithm"].tolist()]
    plt.xticks(x, labels, rotation=15)
    plt.ylabel("Total runtime (sec)")
    plt.title("Total runtime")
    plt.tight_layout()
    plt.savefig(outdir / "total_runtime.png", dpi=170)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    seed_path = args.input / "seed_summary.csv"
    round_path = args.input / "round_metrics.csv"
    if not seed_path.exists():
        raise FileNotFoundError(f"Missing: {seed_path}")

    seed_df = pd.read_csv(seed_path)
    if round_path.exists():
        _plot_regret(pd.read_csv(round_path), args.output)
    _plot_projection_events(seed_df, args.output)
    _plot_runtime(seed_df, args.output)

    if round_path.exists():
        print(f"[plot] wrote {args.output / 'regret_vs_round.png'}")
    print(f"[plot] wrote {args.output / 'projection_events.png'}")
    print(f"[plot] wrote {args.output / 'total_runtime.png'}")


if __name__ == "__main__":
    main()
