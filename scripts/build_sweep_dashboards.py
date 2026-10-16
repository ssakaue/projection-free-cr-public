#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Tuple

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
import yaml
from matplotlib import colors as mcolors
from scipy import stats as _stats
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, LogFormatterMathtext, LogLocator, MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.titlesize": 15,
        "axes.labelsize": 16,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 13.5,
        "figure.titlesize": 16,
    }
)

CURVE_METRICS = [
    ("final_regret_mean", "Final regret", False),
    ("total_runtime_mean", "Runtime (s)", True),
    ("projection_events_mean", "Projection count", False),
]

SPREAD_METRICS = [
    ("final_regret_mean", "Final regret", False),
    ("total_runtime_mean", "Runtime (s)", True),
]

ALGO_ORDER = ["corectron_rbf", "kons", "corectron", "ons", "ogd"]
ALGO_LABEL = {
    "corectron_rbf": "CoRectron-K",
    "kons": "KONS",
    "corectron": "CoRectron-L",
    "ons": "ONS",
    "ogd": "OGD",
}
ALGO_COLOR = {
    "corectron_rbf": "#d62728",
    "kons": "#9467bd",
    "corectron": "#ff7f0e",
    "ons": "#1f77b4",
    "ogd": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build curve/spread dashboards from sweep summaries.")
    p.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing scenario subdirs with sensitivity_summary.csv",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Output directory for dashboard PNGs",
    )
    p.add_argument(
        "--groups",
        type=str,
        default="linear,kernel_rbf,kernel_poly",
        help="Comma-separated groups: linear,kernel_rbf,kernel_poly",
    )
    p.add_argument(
        "--band-stat",
        type=str,
        default="ci95",
        choices=["ci95", "std"],
        help="Uncertainty band for curves: ci95 or std.",
    )
    return p.parse_args()


def scenario_label(name: str) -> str:
    if name.startswith("linear_"):
        m = re.search(r"t(\d+)", name)
        t = m.group(1) if m else "?"
        return f"Linear (T={t})"
    if name.startswith("kernel_rbf_"):
        ms = re.search(r"sigma(\d+)p(\d+)", name)
        mt = re.search(r"t(\d+)", name)
        sigma = f"{ms.group(1)}.{ms.group(2)}" if ms else "?"
        t = mt.group(1) if mt else "?"
        return f"Kernel-RBF (vartheta_rbf={sigma}, T={t})"
    if name.startswith("kernel_poly_"):
        md = re.search(r"deg(\d+)", name)
        mt = re.search(r"t(\d+)", name)
        deg = md.group(1) if md else "?"
        t = mt.group(1) if mt else "?"
        return f"Kernel-Poly (deg={deg}, T={t})"
    return name


def scenario_row_tag(name: str) -> str:
    if name.startswith("linear_"):
        return rf"$\mathrm{{Linear}}$"
    if name.startswith("kernel_rbf_"):
        ms = re.search(r"sigma(\d+)p(\d+)", name)
        sigma = f"{ms.group(1)}.{ms.group(2)}" if ms else "?"
        return rf"$\mathrm{{Kernel}},\ \vartheta_{{\mathrm{{rbf}}}}={sigma}$"
    if name.startswith("kernel_poly_"):
        md = re.search(r"deg(\d+)", name)
        deg = md.group(1) if md else "?"
        return rf"$\mathrm{{Kernel}},\ d={deg}$"
    return name


def scenario_panel_title(name: str) -> str:
    if name.startswith("kernel_rbf_"):
        ms = re.search(r"sigma(\d+)p(\d+)", name)
        sigma = f"{ms.group(1)}.{ms.group(2)}" if ms else "?"
        return rf"$\vartheta_{{\mathrm{{rbf}}}}={sigma}$"
    if name.startswith("kernel_poly_"):
        md = re.search(r"deg(\d+)", name)
        deg = md.group(1) if md else "?"
        return rf"$d={deg}$"
    return ""


def sort_key(name: str) -> tuple[int, float]:
    if name.startswith("linear_"):
        return (0, 0.0)
    if name.startswith("kernel_rbf_"):
        m = re.search(r"sigma(\d+)p(\d+)", name)
        if m:
            return (1, float(f"{m.group(1)}.{m.group(2)}"))
        return (1, 999.0)
    if name.startswith("kernel_poly_"):
        m = re.search(r"deg(\d+)", name)
        if m:
            return (2, float(m.group(1)))
        return (2, 999.0)
    return (9, 999.0)


def _read_seed_count_from_scenario_dir(scenario_dir: Path) -> int | None:
    for cfg_path in sorted(scenario_dir.glob("*/resolved_config.yaml")):
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            seeds = cfg.get("experiment", {}).get("seeds", [])
            if isinstance(seeds, list) and len(seeds) > 0:
                return int(len(seeds))
        except Exception:
            continue
    return None


def read_summaries(run_root: Path) -> Tuple[dict[str, pd.DataFrame], dict[str, int | None]]:
    out: dict[str, pd.DataFrame] = {}
    seed_counts: dict[str, int | None] = {}
    for csv_path in run_root.glob("*/sensitivity_summary.csv"):
        name = csv_path.parent.name
        df = pd.read_csv(csv_path).copy()
        df["scenario"] = name
        out[name] = df
        seed_counts[name] = _read_seed_count_from_scenario_dir(csv_path.parent)
    return out, seed_counts


def _present_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].unique().tolist())
    ordered = [a for a in ALGO_ORDER if a in present]
    extra = [a for a in sorted(present) if a not in ordered]
    return ordered + extra


def _add_row_label(ax: plt.Axes, label: str, fontsize: float = 13.5, x: float = -0.22) -> None:
    ax.text(
        x,
        0.5,
        label,
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=fontsize,
    )


def _fmt_best_value(v: float) -> str:
    if abs(v) >= 100.0:
        return f"{v:.1f}"
    if abs(v) >= 10.0:
        return f"{v:.2f}"
    return f"{v:.3f}"


def build_curve_dashboard(
    scenarios: list[str],
    data: dict[str, pd.DataFrame],
    seed_counts: dict[str, int | None],
    out_path: Path,
    suptitle: str,
    curve_algorithms: list[str] | None = None,
    add_regret_zoom: bool = False,
    band_stat: str = "ci95",
) -> None:
    rows = len(scenarios)
    cols = len(CURVE_METRICS)
    all_kernel_rows = bool(scenarios) and all(s.startswith("kernel_") for s in scenarios)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.0 * rows), squeeze=False)
    seen_algos: set[str] = set()
    curve_font_scale = 1.5
    curve_label_fs = float(plt.rcParams.get("axes.labelsize", 16.0)) * curve_font_scale
    curve_title_fs = curve_label_fs
    curve_ytick_fs = float(plt.rcParams.get("ytick.labelsize", 12.5)) * curve_font_scale
    curve_xtick_fs = 12.0
    curve_legend_fs = float(plt.rcParams.get("legend.fontsize", 13.5)) * curve_font_scale
    coeff_ticks = np.unique(np.concatenate([data[s]["coeff"].to_numpy(dtype=float) for s in scenarios]))

    for r, s in enumerate(scenarios):
        df = data[s]
        seed_count = seed_counts.get(s)
        algos = _present_algorithms(df)
        if curve_algorithms is not None:
            allowed = set(curve_algorithms)
            algos = [a for a in algos if a in allowed]
        # Curves compare methods with and without Mahalanobis projections.
        if s.startswith("linear_"):
            algos = [a for a in algos if a in {"corectron", "ons"}]
        elif s.startswith("kernel_"):
            algos = [a for a in algos if a in {"corectron_rbf", "kons"}]
        for c, (metric, title, log_y) in enumerate(CURVE_METRICS):
            ax = axes[r][c]
            metric_algos = algos
            for algo in metric_algos:
                sub = df[df["algorithm"] == algo].sort_values("coeff")
                if sub.empty:
                    continue
                x = sub["coeff"].to_numpy(dtype=float)
                y = sub[metric].to_numpy(dtype=float)
                color = ALGO_COLOR.get(algo, None)
                ci_col = metric.replace("_mean", "_ci95")
                if ci_col in sub.columns:
                    ci = sub[ci_col].to_numpy(dtype=float)
                    if band_stat == "std":
                        counts = (sub["num_seeds"].to_numpy(dtype=float) if "num_seeds" in sub
                                  else np.full(len(sub), np.nan if seed_count is None else seed_count))
                        if np.isnan(counts).any():
                            raise ValueError(f"Seed counts are required to reconstruct standard deviations for {s}")
                        t_crit = _stats.t.ppf(0.975, counts - 1)
                        err = ci * np.sqrt(counts) / t_crit
                    else:
                        err = ci
                    lower = y - err
                    upper = y + err
                    if log_y:
                        lower = np.maximum(lower, np.finfo(float).tiny)
                    ax.fill_between(
                        x,
                        lower,
                        upper,
                        color=color,
                        alpha=0.16,
                        linewidth=0.0,
                        zorder=1,
                    )
                ax.plot(
                    x,
                    y,
                    marker="o",
                    linewidth=1.8,
                    markersize=4.8,
                    color=color,
                    label=ALGO_LABEL.get(algo, algo),
                    alpha=0.95,
                    zorder=2,
                )
                seen_algos.add(algo)
            ax.set_xscale("log")
            ax.set_xticks(coeff_ticks)
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            if log_y:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=6))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.grid(alpha=0.25, which="both")
            ax.tick_params(axis="x", labelsize=curve_xtick_fs)
            ax.tick_params(axis="y", labelsize=curve_ytick_fs)
            ax.set_ylabel(title, fontsize=curve_label_fs, labelpad=14)
            panel_title = scenario_panel_title(s)
            if panel_title:
                ax.set_title(panel_title, pad=16, fontsize=curve_title_fs)
            else:
                ax.set_title("")
            if r == rows - 1:
                ax.set_xlabel(r"Coefficient $c$", fontsize=curve_label_fs, labelpad=14)

            if metric == "final_regret_mean" and len(algos) >= 1:
                combined = df[df["algorithm"].isin(algos)].sort_values("coeff")
                if not combined.empty:
                    best_row = combined.sort_values(
                        by=["final_regret_mean", "algorithm", "coeff"], ascending=[True, True, True]
                    ).iloc[0]
                    best_v = float(best_row["final_regret_mean"])
                    coeffs = np.sort(combined["coeff"].unique())
                    vals = combined["final_regret_mean"].to_numpy(dtype=float)

                    if add_regret_zoom:
                        c1_vals = []
                        for algo in algos:
                            sub = combined[combined["algorithm"] == algo]
                            if sub.empty:
                                continue
                            idx = np.argmin(np.abs(sub["coeff"].to_numpy(dtype=float) - 1.0))
                            c1_vals.append(float(sub.iloc[idx]["final_regret_mean"]))
                        # Keep all minima in the inset, then zoom around low-regret band.
                        low_anchor = best_v
                        high_anchor = float(np.percentile(vals, 85))
                        if c1_vals:
                            high_anchor = max(high_anchor, float(max(c1_vals)))
                        span = max(high_anchor - low_anchor, abs(high_anchor) * 0.03, 1e-6)
                        pad = 0.20 * span
                        lower = low_anchor - pad
                        upper = high_anchor + pad

                        iax = inset_axes(ax, width="38%", height="48%", loc="center right", borderpad=1.0)
                        for algo in algos:
                            sub = df[df["algorithm"] == algo].sort_values("coeff")
                            if sub.empty:
                                continue
                            x = sub["coeff"].to_numpy(dtype=float)
                            y = sub["final_regret_mean"].to_numpy(dtype=float)
                            color = ALGO_COLOR.get(algo, None)
                            iax.plot(
                                x,
                                y,
                                marker="o",
                                linewidth=1.2,
                                markersize=3.0,
                                color=color,
                                alpha=0.9,
                            )
                        iax.set_xscale("log")
                        iax.set_xlim(float(coeffs.min()), float(coeffs.max()))
                        iax.set_ylim(lower, upper)
                        iax.grid(alpha=0.2, which="both")
                        iax.set_title("zoom", fontsize=8 * curve_font_scale)
                        iax.tick_params(labelsize=7 * curve_font_scale)

    legend_algos = [a for a in ["corectron", "ons", "corectron_rbf", "kons", "ogd"] if a in seen_algos]

    def _legend_handles(algo_list: list[str]) -> list[Line2D]:
        return [
            Line2D(
                [0],
                [0],
                marker="o",
                linewidth=1.8,
                markersize=4.8,
                color=ALGO_COLOR.get(a, "#444444"),
                label=ALGO_LABEL.get(a, a),
            )
            for a in algo_list
        ]

    linear_legend_algos = [a for a in ["corectron", "ons"] if a in legend_algos]
    kernel_legend_algos = [a for a in ["corectron_rbf", "kons"] if a in legend_algos]
    extra_legend_algos = [a for a in legend_algos if a not in set(linear_legend_algos + kernel_legend_algos)]
    legend_artists = []
    has_split_legend_boxes = bool(linear_legend_algos and kernel_legend_algos)
    has_single_family_legend = False
    if has_split_legend_boxes:
        leg_left = fig.legend(
            _legend_handles(linear_legend_algos),
            [ALGO_LABEL.get(a, a) for a in linear_legend_algos],
            title="Linear",
            loc="upper center",
            bbox_to_anchor=(0.36, 0.900),
            ncol=len(linear_legend_algos),
            frameon=True,
            borderaxespad=0.2,
            handletextpad=0.55,
            columnspacing=1.10,
            fontsize=curve_legend_fs,
            title_fontsize=curve_legend_fs,
        )
        leg_right = fig.legend(
            _legend_handles(kernel_legend_algos),
            [ALGO_LABEL.get(a, a) for a in kernel_legend_algos],
            title="Kernel",
            loc="upper center",
            bbox_to_anchor=(0.64, 0.900),
            ncol=len(kernel_legend_algos),
            frameon=True,
            borderaxespad=0.2,
            handletextpad=0.55,
            columnspacing=1.10,
            fontsize=curve_legend_fs,
            title_fontsize=curve_legend_fs,
        )
        legend_artists.extend([leg_left, leg_right])
    elif legend_algos:
        single_family_is_kernel = set(legend_algos).issubset({"corectron_rbf", "kons"})
        single_family_legend_y = 0.905 if single_family_is_kernel else 0.930
        legend_artists.append(
            fig.legend(
                _legend_handles(legend_algos),
                [ALGO_LABEL.get(a, a) for a in legend_algos],
                loc="upper center",
                bbox_to_anchor=(0.5, single_family_legend_y),
                ncol=len(legend_algos),
                frameon=True,
                borderaxespad=0.2,
                handletextpad=0.55,
                columnspacing=1.25,
                fontsize=curve_legend_fs,
            )
        )
        has_single_family_legend = True
    if extra_legend_algos:
        legend_artists.append(
            fig.legend(
                _legend_handles(extra_legend_algos),
                [ALGO_LABEL.get(a, a) for a in extra_legend_algos],
                loc="upper center",
                bbox_to_anchor=(0.5, 0.855),
                ncol=len(extra_legend_algos),
                frameon=True,
                borderaxespad=0.2,
                handletextpad=0.55,
                columnspacing=1.20,
                fontsize=curve_legend_fs,
            )
        )
    for leg in legend_artists[:-1]:
        fig.add_artist(leg)
    if rows == 1:
        if suptitle:
            fig.suptitle(suptitle, y=0.985, fontsize=14, fontweight="bold")
        top_margin = 0.70 if has_single_family_legend else 0.78
        fig.subplots_adjust(left=0.10, right=0.995, bottom=0.16, top=top_margin, hspace=0.36, wspace=0.40)
    else:
        if suptitle:
            fig.suptitle(suptitle, y=0.985, fontsize=14, fontweight="bold")
        curve_hspace = 0.44 if all_kernel_rows else 0.36
        fig.subplots_adjust(left=0.10, right=0.995, bottom=0.10, top=0.82, hspace=curve_hspace, wspace=0.40)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def _spread_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    scenario_name: str,
    metric: str,
    title: str,
    log_y: bool,
    coeff_offsets: dict[float, float],
    coeff_norm: mcolors.LogNorm,
    annotate_focus_bests: bool = False,
    annotate_fontsize: float = 10.5,
) -> None:
    algos = _present_algorithms(df)
    x_positions = np.arange(len(algos))

    best_annotations: list[tuple[float, float, str, str, str]] = []
    if annotate_focus_bests and not df.empty:
        # Show best value for every algorithm.
        for algo in algos:
            sub = df[df["algorithm"] == algo].sort_values(by=[metric, "coeff"], ascending=[True, True])
            if sub.empty:
                continue
            best_row = sub.iloc[0]
            best_coeff = float(best_row["coeff"])
            best_val = float(best_row[metric])
            best_x = float(algos.index(algo)) + float(coeff_offsets.get(best_coeff, 0.0))
            place = "top"
            # Place linear-method labels below the kernel-method labels.
            if scenario_name.startswith("kernel_") and algo in {"corectron", "ons", "ogd"}:
                place = "bottom"
            # In linear rows, place OGD and ONS labels above their points.
            if scenario_name.startswith("linear_") and algo == "ogd":
                place = "top"
            best_annotations.append((best_x, best_val, _fmt_best_value(best_val), place, algo))

    for x, algo in zip(x_positions, algos):
        sub = df[df["algorithm"] == algo].sort_values("coeff")
        vals = sub[metric].to_numpy(dtype=float)
        if vals.size == 0:
            continue
        color = ALGO_COLOR.get(algo, "#444444")
        for row in sub.itertuples(index=False):
            c = float(row.coeff)
            v = float(getattr(row, metric))
            x_pt = float(x) + float(coeff_offsets.get(c, 0.0))
            if coeff_norm.vmax <= coeff_norm.vmin:
                norm_v = 0.0
            else:
                norm_v = float(coeff_norm(c))
            # Keep method hue fixed; encode coeff by shade (dark=small c, light=large c).
            base_rgb = np.array(mcolors.to_rgb(color), dtype=float)
            blend_to_white = 0.72 * norm_v
            coeff_rgb = (1.0 - blend_to_white) * base_rgb + blend_to_white * np.ones(3)
            ax.scatter(
                [x_pt],
                [v],
                marker="o",
                s=34.0,
                facecolor=coeff_rgb,
                edgecolor=(0.0, 0.0, 0.0, 0.20),
                linewidths=0.28,
                alpha=0.92,
                zorder=3,
            )

    if annotate_focus_bests and best_annotations and not log_y:
        all_vals = df[metric].to_numpy(dtype=float)
        y_min, y_max = float(all_vals.min()), float(all_vals.max())
        y_span = max(y_max - y_min, abs(y_max) * 0.05, 1e-6)
        top_count = sum(1 for _, _, _, place, _ in best_annotations if place == "top")
        lower_pad = 0.05 * y_span
        upper_pad = 0.24 * y_span if top_count else 0.08 * y_span
        ax.set_ylim(y_min - lower_pad, y_max + upper_pad)
        # Reserve horizontal room so left/right-edge labels remain visible without breaking vertical leaders.
        ax.set_xlim(-0.60, len(algos) - 0.40)
        top_rank = 0
        bottom_rank = 0
        for anchor_x, anchor_y, label, place, algo in best_annotations:
            if place == "bottom":
                y_text = anchor_y - (0.10 + 0.03 * (bottom_rank % 2)) * y_span
                y_text = max(y_text, y_min + 0.02 * y_span)
                va = "top"
                bottom_rank += 1
            else:
                y_text = anchor_y + (0.11 + 0.03 * (top_rank % 3)) * y_span
                if scenario_name == "kernel_rbf_sigma0p5_t1000_pow10_s10_all5" and algo == "kons":
                    y_text += 0.14 * y_span
                if scenario_name == "kernel_rbf_sigma2p0_t1000_pow10_s10_all5" and algo == "kons":
                    y_text = anchor_y + 0.045 * y_span
                va = "bottom"
                top_rank += 1
            ax.annotate(
                label,
                xy=(anchor_x, anchor_y),
                xytext=(anchor_x, y_text),
                textcoords="data",
                fontsize=annotate_fontsize,
                color="black",
                ha="center",
                va=va,
                arrowprops=dict(arrowstyle="-", color=(0, 0, 0, 0.55), lw=0.9),
                zorder=7,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([ALGO_LABEL.get(a, a) for a in algos], rotation=18, ha="right")
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.18, which="major", axis="y")
    ax.set_title(title)


def build_spread_dashboard(
    scenarios: list[str],
    data: dict[str, pd.DataFrame],
    out_path: Path,
    suptitle: str,
) -> None:
    rows = len(scenarios)
    cols = len(SPREAD_METRICS)
    all_kernel_rows = bool(scenarios) and all(s.startswith("kernel_") for s in scenarios)
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.8 * rows), squeeze=False)
    spread_font_scale = 1.35
    spread_label_fs = float(plt.rcParams.get("axes.labelsize", 16.0)) * spread_font_scale
    spread_title_fs = spread_label_fs
    spread_tick_fs = float(plt.rcParams.get("ytick.labelsize", 12.5)) * 1.20
    coeff_union = sorted({float(c) for s in scenarios for c in data[s]["coeff"].unique()})
    if not coeff_union:
        raise ValueError("No coefficient points found for spread dashboard.")
    if any(c <= 0.0 for c in coeff_union):
        raise ValueError("All coefficients must be positive for log-scale coeff coloring.")
    coeff_norm = mcolors.LogNorm(vmin=float(min(coeff_union)), vmax=float(max(coeff_union)))
    if len(coeff_union) == 1:
        coeff_offsets = {coeff_union[0]: 0.0}
    else:
        half = (len(coeff_union) - 1) / 2.0
        coeff_offsets = {c: 0.13 * ((i - half) / half) for i, c in enumerate(coeff_union)}

    for r, s in enumerate(scenarios):
        df = data[s]
        for c, (metric, title, log_y) in enumerate(SPREAD_METRICS):
            ax = axes[r][c]
            _spread_panel(
                ax,
                df,
                scenario_name=s,
                metric=metric,
                title=title,
                log_y=log_y,
                coeff_offsets=coeff_offsets,
                coeff_norm=coeff_norm,
                annotate_focus_bests=(metric == "final_regret_mean"),
                annotate_fontsize=spread_tick_fs,
            )
            ax.tick_params(axis="x", labelsize=spread_tick_fs)
            ax.tick_params(axis="y", labelsize=spread_tick_fs)
            ax.set_xlabel("")
            ax.set_ylabel(title, fontsize=spread_label_fs, labelpad=14)
            panel_title = scenario_panel_title(s)
            if panel_title:
                ax.set_title(panel_title, pad=12, fontsize=spread_title_fs)
            else:
                ax.set_title("")

    if rows == 1:
        if suptitle:
            fig.suptitle(suptitle, y=0.985, fontsize=14, fontweight="bold")
        fig.subplots_adjust(left=0.09, right=0.995, top=0.90, bottom=0.20, hspace=0.26, wspace=0.34)
    else:
        if suptitle:
            fig.suptitle(suptitle, y=0.985, fontsize=14, fontweight="bold")
        spread_hspace = 0.42 if all_kernel_rows else 0.26
        fig.subplots_adjust(left=0.09, right=0.995, top=0.93, bottom=0.10, hspace=spread_hspace, wspace=0.34)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_root = args.out_root
    if out_root is None:
        out_root = Path("figures") / args.run_root.name / "dashboards"

    data, seed_counts = read_summaries(args.run_root)
    if not data:
        raise RuntimeError(f"No sensitivity_summary.csv found under: {args.run_root}")

    names = sorted(data.keys(), key=sort_key)
    linear = [n for n in names if n.startswith("linear_")]
    rbf = [n for n in names if n.startswith("kernel_rbf_")]
    poly = [n for n in names if n.startswith("kernel_poly_")]

    groups = {
        "linear": ("linear", linear, "Linear Sweep Dashboard"),
        "kernel_rbf": ("kernel_rbf", rbf, "Kernel-RBF Sweep Dashboard"),
        "kernel_poly": ("kernel_poly", poly, "Kernel-Poly Sweep Dashboard"),
        "main": ("main", linear + rbf, "Main (Linear + Kernel-RBF) Sweep Dashboard"),
    }
    requested = [g.strip() for g in args.groups.split(",") if g.strip()]
    if not requested:
        raise ValueError("--groups must not be empty")
    bad = [g for g in requested if g not in groups]
    if bad:
        raise ValueError(f"Unknown group(s): {bad}")

    for g in requested:
        group_tag, scenarios, _ = groups[g]
        if not scenarios:
            continue
        curve_algorithms = None
        add_zoom = False
        if group_tag == "linear":
            add_zoom = True
        if group_tag.startswith("kernel_"):
            curve_algorithms = ["corectron_rbf", "kons"]
            add_zoom = True
        build_curve_dashboard(
            scenarios=scenarios,
            data=data,
            seed_counts=seed_counts,
            out_path=out_root / f"curves_dashboard_{group_tag}.png",
            suptitle="",
            curve_algorithms=curve_algorithms,
            add_regret_zoom=add_zoom,
            band_stat=args.band_stat,
        )
        build_spread_dashboard(
            scenarios=scenarios,
            data=data,
            out_path=out_root / f"spread_dashboard_{group_tag}.png",
            suptitle="",
        )
        print(f"[write] {out_root / f'curves_dashboard_{group_tag}.png'}")
        print(f"[write] {out_root / f'spread_dashboard_{group_tag}.png'}")


if __name__ == "__main__":
    main()
