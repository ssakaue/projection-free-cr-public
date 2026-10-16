"""Exercise campaign defaults and failure/resume boundaries without long runs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns(
        ".git", ".venv", "__pycache__", ".pytest_cache", ".mplconfig", "results", "figures", "data"))
    calls = tmp_path / "calls.jsonl"
    fake = tmp_path / "python"
    fake.write_text(f"#!{sys.executable}\n" + '''import json, os, subprocess, sys
from pathlib import Path
if sys.argv[1] == "--version" or sys.argv[1].endswith("campaign_manifest.py"):
    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))
with Path(os.environ["CALLS"]).open("a") as handle:
    handle.write(json.dumps({"args": sys.argv[1:], "env": {
        key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")}}) + "\\n")
if os.environ.get("CHANGE_SOURCE"):
    with Path("run_experiment.py").open("a") as handle:
        handle.write("\\n# Modified during campaign test.\\n")
raise SystemExit(17 if os.environ.get("FAIL_SCRIPT") == sys.argv[1] else 0)
''')
    fake.chmod(0o755)
    env = {**os.environ, "PYTHON_BIN": str(fake), "CALLS": str(calls)}

    def run(kind, *args, **extra_env):
        return subprocess.run(["bash", str(repo / f"scripts/run_{kind}_campaign_unattended.sh"), *args],
                              cwd=tmp_path, env={**env, **extra_env}, capture_output=True, text=True)

    def read_calls():
        return [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []

    return repo, run, read_calls


def option(args, name):
    return args[args.index(name) + 1]


def test_comparison_default_design_and_thread_environment(launcher):
    repo, run, read_calls = launcher
    result = run("main", "comparison")
    assert result.returncode == 0, result.stderr
    calls = read_calls()
    sweeps = [call for call in calls if call["args"][0].endswith("run_reg_strength_sweep.py")]
    assert len(sweeps) == 4
    assert sum(len(option(call["args"], "--targets").split(",")) *
               len(option(call["args"], "--coeff-grid").split(",")) for call in sweeps) == 126
    assert [option(call["args"], "--kernel-sigma") for call in sweeps[1:]] == ["0.5", "1.0", "2.0"]
    for call in calls:
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert call["env"][name] == "1"
    assert (repo / "results/comparison/campaign_status_latest.tsv").is_file()


def test_feedback_default_design(launcher):
    _, run, read_calls = launcher
    result = run("subopt", "feedback")
    assert result.returncode == 0, result.stderr
    args = read_calls()[0]["args"]
    grid = [float(x) for x in option(args, "--subopt-prob-grid").split(",")]
    noise = [float(x) for x in option(args, "--subopt-noise-std-grid").split(",")]
    assert grid == noise == [x / 10 for x in range(11)]
    assert len(option(args, "--targets").split(",")) * (len(grid) + len(noise)) == 66
    assert option(args, "--ons-lr") == "2.0"


@pytest.mark.parametrize("kind,script", [
    ("main", "scripts/run_reg_strength_sweep.py"),
    ("subopt", "scripts/run_subopt_combined.py"),
])
def test_failure_stops_without_retry_and_preserves_invocation_logs(launcher, kind, script):
    repo, run, read_calls = launcher
    for _ in range(2):
        result = run(kind, "failure", FAIL_SCRIPT=script)
        assert result.returncode == 17
    assert len(read_calls()) == 2
    invocations = list((repo / "results/failure/invocations").iterdir())
    assert len(invocations) == 2
    for invocation in invocations:
        assert (invocation / "environment_snapshot.txt").is_file()
        assert "campaign\tfailed" in (invocation / "status.tsv").read_text()
        assert len(list((invocation / "logs").glob("*.log"))) == 1
    assert not (repo / "results/failure/.campaign-lock").exists()


def test_source_change_between_jobs_stops_campaign(launcher):
    _, run, read_calls = launcher
    result = run("main", "source-change", CHANGE_SOURCE="1")
    assert result.returncode != 0 and "changed" in result.stderr
    assert len(read_calls()) == 1


def test_changed_campaign_rejected_before_any_job(launcher):
    _, run, read_calls = launcher
    assert run("main", "same").returncode == 0
    before = read_calls()
    result = run("main", "same", "0.01,1")
    assert result.returncode != 0 and "changed" in result.stderr
    assert read_calls() == before


@pytest.mark.parametrize("kind,args", [
    ("main", ["../outside"]), ("subopt", ["../outside"]),
    ("main", ["bad", "nan"]), ("subopt", ["bad", "1.1"]),
    ("main", ["bad", "1", "2"]),
])
def test_invalid_inputs_fail_before_outputs_or_jobs(launcher, kind, args):
    repo, run, read_calls = launcher
    assert run(kind, *args).returncode != 0
    assert not read_calls()
    assert not (repo / "results").exists()


def test_standalone_feedback_defaults_match_combined(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from scripts import run_subopt_sweep, run_subopt_noise_sweep, run_subopt_combined
    from experiments_core.config import load_config
    base = load_config(ROOT / "configs/main_linear_t10000_s10.yaml")
    for module in (run_subopt_sweep, run_subopt_noise_sweep):
        monkeypatch.setattr(sys, "argv", ["script", "--config", "config.yaml",
                                          "--output-root", "results", "--figure-root", "figures"])
        args = module.parse_args()
        assert args.targets == "corectron,ons,ogd"
        grid = args.subopt_prob_grid if module is run_subopt_sweep else args.subopt_noise_std_grid
        assert [float(value) for value in grid.split(",")] == [x / 10 for x in range(11)]
    for target, coefficient in run_subopt_combined._DEFAULT_BEST_COEFFS.items():
        assert run_subopt_sweep._resolve_with_best_c(base, target, coefficient, .4, "one_swap", None) == \
            run_subopt_combined._resolve_one_swap(base, target, coefficient, .4, None)
        assert run_subopt_noise_sweep._resolve_with_best_c(base, target, coefficient, 1., .4, None) == \
            run_subopt_combined._resolve_score_perturb(base, target, coefficient, .4, None)
