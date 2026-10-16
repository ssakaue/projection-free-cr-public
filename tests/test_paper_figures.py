"""Scientific integrity of the saved paper data and plotting-only entry point."""
from pathlib import Path
import shutil
import sys

import pandas as pd
from PIL import Image
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_paper_figures as figures
import run_subopt_combined

DATA = ROOT / "data/paper"


def test_complete_data_and_plotting_without_experiments(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Plotting must not run experiments")

    monkeypatch.setattr(run_subopt_combined, "_run_sweep", forbidden)
    manifest = figures.validate_manifest(DATA)
    assert manifest["counts"]["total_seed_runs"] == 1920
    figures.render(DATA / "comparison", DATA / "feedback", tmp_path)
    expected = {
        "spread_dashboard_linear.png", "spread_dashboard_kernel_rbf.png",
        "curves_dashboard_linear.png", "curves_dashboard_kernel_rbf.png",
        "subopt_combined.png",
    }
    assert {path.name for path in tmp_path.glob("*.png")} == expected
    for path in tmp_path.glob("*.png"):
        with Image.open(path) as image:
            assert image.width >= 1000 and image.height >= 500
            assert image.convert("RGB").getextrema() != ((255, 255),) * 3


@pytest.mark.parametrize("case", ["duplicate_seed", "wrong_bandwidth", "wrong_feedback_coefficient"])
def test_seed_or_configuration_mismatch_is_rejected(tmp_path, case):
    if case == "wrong_feedback_coefficient":
        source = DATA / "feedback/one_swap/ons_p0p1"
        algorithm, horizon = "ons", 10000
    else:
        source = DATA / "comparison/kernel_rbf_sigma0p5_t1000_pow10_s10_all5/kons_c0p001"
        algorithm, horizon = "kons", 1000
    destination = tmp_path / "setting"
    shutil.copytree(source, destination)
    if case == "duplicate_seed":
        path = destination / "seed_summary.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "seed"] = frame.loc[1, "seed"]
        frame.to_csv(path, index=False)
    else:
        path = destination / "resolved_config.yaml"
        config = yaml.safe_load(path.read_text())
        if case == "wrong_bandwidth":
            config["dataset"]["kernel"]["sigma_true"] = 1.0
        else:
            config["learners"]["ons"]["epsilon"] *= 10
        path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        figures.read_seed_data(destination, algorithm, horizon, source / "resolved_config.yaml")


def test_summary_interval_must_match_seed_measurements(tmp_path):
    root = tmp_path / "comparison"
    shutil.copytree(DATA / "comparison", root)
    path = root / "linear_t10000_pow10_s10/sensitivity_summary.csv"
    summary = pd.read_csv(path)
    summary.loc[0, "final_regret_ci95"] += 0.1
    summary.to_csv(path, index=False)
    with pytest.raises(ValueError, match="final_regret CI"):
        figures.validate_comparison(root)


def test_bundled_data_modification_is_detected(tmp_path):
    root = tmp_path / "paper"
    shutil.copytree(DATA, root)
    path = root / "feedback/one_swap_summary.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="checksum"):
        figures.validate_manifest(root)
