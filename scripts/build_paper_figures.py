#!/usr/bin/env python3
"""Validate saved experiment summaries and render Figures 1–5 without rerunning learners."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import build_sweep_dashboards as dashboards
from run_subopt_combined import render_combined_figure

ROOT = Path(__file__).resolve().parents[1]
SEEDS = list(range(10))
COEFFICIENTS = [10.0**power for power in range(-3, 4)]
NOISE_LEVELS = [i / 10 for i in range(11)]
SCENARIOS = {
    "linear_t10000_pow10_s10": (10000, ["corectron", "ons", "ogd"]),
    **{
        f"kernel_rbf_sigma{sigma}_t1000_pow10_s10_all5":
        (1000, ["corectron_rbf", "kons", "corectron", "ons", "ogd"])
        for sigma in ("0p5", "1p0", "2p0")
    },
}
METRICS = {
    "final_regret": "final_regret",
    "total_runtime": "total_runtime_sec",
    "total_cpu_time": "total_cpu_time_sec",
    "projection_events": "projection_events",
}


def value_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(data_root: Path) -> dict:
    """Verify that the bundled data still match the measured files."""
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1, "Unsupported paper-data manifest version")
    hashes = manifest["files_sha256"]
    require(len(hashes) == 390, "The paper-data manifest must cover all 390 data files")
    for relative, expected in hashes.items():
        path = data_root / relative
        require(path.resolve().is_relative_to(data_root.resolve()), "Invalid manifest path")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"Paper data do not match their checksum: {relative}")
    return manifest


def read_seed_data(directory: Path, algorithm: str, horizon: int,
                   reference: Path) -> tuple[pd.DataFrame, dict]:
    """Check complete seed coverage and the matching resolved experiment config."""
    frame = pd.read_csv(directory / "seed_summary.csv")
    config = yaml.safe_load((directory / "resolved_config.yaml").read_text(encoding="utf-8"))
    require(len(frame) == 10 and sorted(frame["seed"].tolist()) == SEEDS,
            f"Expected each of seeds 0–9 exactly once: {directory}")
    require(set(frame["algorithm"]) == {algorithm}, f"Algorithm mismatch: {directory}")
    require(config["experiment"]["seeds"] == SEEDS, f"Config seed mismatch: {directory}")
    require(config["experiment"]["horizon"] == horizon, f"Horizon mismatch: {directory}")
    enabled = {key for key, value in config["learners"].items() if value.get("enabled")}
    require(enabled == {algorithm}, f"Enabled learner mismatch: {directory}")
    expected = yaml.safe_load(reference.read_text(encoding="utf-8"))
    for section in ("normalization", "dataset", "learners", "user_policy"):
        require(config[section] == expected[section],
                f"The {section} configuration differs from the paper experiment: {directory}")
    numeric = frame.select_dtypes(include=[np.number])
    require(np.isfinite(numeric.to_numpy()).all(), f"Non-finite seed measurement: {directory}")
    for column in ("total_runtime_sec", "total_cpu_time_sec", "projection_events"):
        require((frame[column] >= 0).all(), f"Negative {column}: {directory}")
    counts = frame["projection_events"].to_numpy()
    require(np.equal(counts, np.floor(counts)).all() and (counts <= horizon).all(),
            f"Invalid projection count: {directory}")
    return frame.sort_values("seed"), config


def check_close(actual, expected, label: str) -> None:
    require(np.allclose(actual, expected, rtol=1e-11, atol=1e-12),
            f"Saved aggregate disagrees with seed measurements: {label}")


def validate_comparison(root: Path) -> None:
    observed = {p.parent.name for p in root.glob("*/sensitivity_summary.csv")}
    require(observed == set(SCENARIOS), "Expected the four paper comparison scenarios")
    critical = float(stats.t.ppf(0.975, 9))
    for scenario, (horizon, algorithms) in SCENARIOS.items():
        directory = root / scenario
        summary = pd.read_csv(directory / "sensitivity_summary.csv")
        expected = {(algorithm, coefficient) for algorithm in algorithms for coefficient in COEFFICIENTS}
        actual = set(zip(summary["algorithm"], summary["coeff"]))
        require(len(summary) == len(expected) and actual == expected,
                f"Missing or repeated algorithm/coefficient: {scenario}")
        for row in summary.itertuples(index=False):
            label = f"{scenario}/{row.algorithm}_c{value_tag(row.coeff)}"
            setting = f"{row.algorithm}_c{value_tag(row.coeff)}"
            reference = ROOT / "data/paper/comparison" / scenario / setting / "resolved_config.yaml"
            frame, config = read_seed_data(directory / setting, row.algorithm, horizon, reference)
            require(row.num_seeds == 10, f"Incorrect summary seed count: {label}")
            for metric, column in METRICS.items():
                values = frame[column].to_numpy(dtype=float)
                check_close(getattr(row, metric + "_mean"), values.mean(), label + "/" + metric)
                check_close(getattr(row, metric + "_ci95"),
                            critical * values.std(ddof=1) / np.sqrt(10), label + "/" + metric + " CI")
            learner = config["learners"][row.algorithm]
            if row.algorithm.startswith("corectron"):
                reg_value = learner["lambda_reg"]
                expected_value = row.coeff * learner["lambda_reg_theory"]
            elif row.algorithm in {"ons", "kons"}:
                reg_value = learner["epsilon"]
                expected_value = row.coeff * learner["epsilon_theory"]
            else:
                reg_value = 1.0 / learner["step0"]
                expected_value = row.coeff / learner["ogd_lr_theory"]
            check_close(row.reg_value, reg_value, label + "/regularization")
            check_close(reg_value, expected_value, label + "/coefficient")


def validate_feedback(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    for model, column, tag in (("one_swap", "subopt_prob", "p"),
                               ("score_perturb", "noise_std", "sigma")):
        summary = pd.read_csv(root / f"{model}_summary.csv")
        expected = {(algorithm, value, seed) for algorithm in ("corectron", "ons", "ogd")
                    for value in NOISE_LEVELS for seed in SEEDS}
        actual = set(zip(summary["algorithm"], summary[column], summary["seed"]))
        require(len(summary) == len(expected) and actual == expected,
                f"Missing or repeated feedback measurement: {model}")
        for (algorithm, value), group in summary.groupby(["algorithm", column]):
            setting = f"{algorithm}_{tag}{value_tag(value)}"
            directory = root / model / setting
            reference = ROOT / "data/paper/feedback" / model / setting / "resolved_config.yaml"
            frame, config = read_seed_data(directory, algorithm, 10000, reference)
            policy = config["user_policy"]
            require(policy["subopt_mode"] == ("one_swap" if model == "one_swap" else "score_perturb_relative"),
                    f"Feedback model mismatch: {directory}")
            check_close(policy["subopt_prob" if model == "one_swap" else "subopt_noise_std"],
                        value, str(directory))
            if model == "score_perturb":
                require(policy["subopt_prob"] == 1.0, f"Perturbation probability mismatch: {directory}")
            group = group.sort_values("seed")
            for metric in ("final_regret", "cumulative_suboptimality"):
                check_close(group[metric].to_numpy(), frame[metric].to_numpy(), f"{directory}/{metric}")
        frames.append(summary)
    return frames[0], frames[1]


def render(comparison_root: Path, feedback_root: Path, out_root: Path) -> None:
    validate_comparison(comparison_root)
    one_swap, score_perturb = validate_feedback(feedback_root)
    data, seed_counts = dashboards.read_summaries(comparison_root)
    for group in ("linear", "kernel_rbf"):
        scenarios = sorted((name for name in data if name.startswith(group + "_")), key=dashboards.sort_key)
        dashboards.build_curve_dashboard(
            scenarios, data, seed_counts, out_root / f"curves_dashboard_{group}.png", "",
            add_regret_zoom=True, band_stat="ci95",
        )
        dashboards.build_spread_dashboard(
            scenarios, data, out_root / f"spread_dashboard_{group}.png", "",
        )
    render_combined_figure(one_swap, score_perturb, out_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Bundled paper data, including manifest.json (default: data/paper in this repository)")
    parser.add_argument("--comparison-root", type=Path, help="New comparison sweep results; requires --feedback-root")
    parser.add_argument("--feedback-root", type=Path, help="New combined feedback results; requires --comparison-root")
    parser.add_argument("--out-root", type=Path, default=Path("figures/paper"))
    parser.add_argument("--validate-only", action="store_true", help="Check data without writing figures")
    args = parser.parse_args()
    custom = args.comparison_root is not None or args.feedback_root is not None
    if custom:
        if args.data_root is not None or args.comparison_root is None or args.feedback_root is None:
            parser.error("Use either --data-root or both --comparison-root and --feedback-root")
        comparison, feedback = args.comparison_root, args.feedback_root
        validate_manifest(ROOT / "data/paper")
    else:
        data_root = args.data_root or ROOT / "data/paper"
        validate_manifest(data_root)
        comparison, feedback = data_root / "comparison", data_root / "feedback"
    if args.validate_only:
        validate_comparison(comparison)
        validate_feedback(feedback)
        print("Validated 192 settings and 1920 seed runs.")
    else:
        render(comparison, feedback, args.out_root)
        print(f"Saved Figures 1–5 to {args.out_root}; no experiments were run.")


if __name__ == "__main__":
    main()
