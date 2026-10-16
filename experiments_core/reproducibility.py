"""Versioned run metadata and checked reuse of completed experiment outputs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
RESULT_NAMES = ("round_metrics.csv", "seed_summary.csv", "aggregate_summary.csv")
STATE_NAMES = ("resolved_config.yaml", "manifest.json", "SUCCESS", "RUNNING", "FAILED")
THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_hashes() -> dict[str, str]:
    paths = set(ROOT.glob("*.py"))
    paths.update((ROOT / "experiments_core").rglob("*.py"))
    paths.update((ROOT / "scripts").glob("*.py"))
    paths.update((ROOT / "scripts").glob("*.sh"))
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in sorted(paths)}


# A sweep executes imported modules, so it must retain one source version.
_PROCESS_CODE_HASHES = code_hashes()


def _blas_configuration(module) -> dict[str, Any]:
    config = getattr(module.__config__, "CONFIG", {})
    dependencies = config.get("Build Dependencies", {})
    if dependencies:
        # Build paths are not needed to identify the linked numerical backend.
        allowed = {"name", "found", "version", "detection method", "openblas configuration", "has ilp64"}
        return {name: {key: value for key, value in dependencies.get(name, {}).items() if key in allowed}
                for name in ("blas", "lapack")}
    # NumPy versions before the structured configuration API expose get_info.
    get_info = getattr(module.__config__, "get_info", None)
    if get_info is not None:
        return {name: {key: value for key, value in get_info(name).items()
                       if key in {"libraries", "define_macros", "language"}}
                for name in ("blas_opt_info", "blas_ilp64_opt_info", "lapack_opt_info")}
    return {"available": False}


def _cpu_model() -> str:
    if platform.system() == "Darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           text=True, stderr=subprocess.DEVNULL, timeout=2).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def environment() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "matplotlib", "PyYAML"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": _cpu_model(), "packages": packages,
        "numerical_backends": {"numpy": _blas_configuration(np), "scipy": _blas_configuration(scipy)},
        "thread_environment": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def build_manifest(cfg: dict[str, Any]) -> dict[str, Any]:
    if code_hashes() != _PROCESS_CODE_HASHES:
        raise RuntimeError("Source files changed since this process started; restart before running or reusing results.")
    encoded = json.dumps(cfg, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"schema_version": 2, "resolved_config_sha256": hashlib.sha256(encoded).hexdigest(),
            "code_sha256": _PROCESS_CODE_HASHES, "environment": environment()}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_or_resume(cfg: dict[str, Any], output_dir: Path, *, force: bool = False):
    """Run once or reuse identical, checksum-verified outputs.

    Existing unmatched, incomplete, or modified results require explicit --force.
    No historical output is silently overwritten. A failed forced run removes its
    partial result files and cannot retain an old SUCCESS marker.
    """
    from .config import dump_resolved_config
    from .runner import run_experiment

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "RUNNING").exists():
        raise ValueError(f"RUNNING exists in {output_dir}; confirm that no process is active before removing this marker.")
    expected = build_manifest(cfg)
    required = ["seed_summary.csv", "aggregate_summary.csv", "resolved_config.yaml"]
    if cfg.get("runtime", {}).get("save_round_metrics", True):
        required.append("round_metrics.csv")
    found = _read_json(output_dir / "manifest.json")
    success = _read_json(output_dir / "SUCCESS")
    if not force and found == expected and isinstance(success, dict):
        expected_hashes = success.get("output_sha256", {})
        if set(expected_hashes) == set(required) and all(
            (output_dir / name).is_file() and sha256_file(output_dir / name) == expected_hashes[name]
            for name in required
        ):
            print(f"[skip] {output_dir} (configuration, code, environment, outputs matched)")
            round_df = pd.read_csv(output_dir / "round_metrics.csv") if "round_metrics.csv" in required else pd.DataFrame()
            return round_df, pd.read_csv(output_dir / "seed_summary.csv"), pd.read_csv(output_dir / "aggregate_summary.csv")
    existing = [name for name in RESULT_NAMES + STATE_NAMES if (output_dir / name).exists()]
    if existing and not force:
        raise ValueError(f"Existing outputs in {output_dir} do not match this run or are incomplete; "
                         "choose a new output directory or use --force to replace them.")
    running = output_dir / "RUNNING"
    if running.exists():
        raise ValueError(f"RUNNING exists in {output_dir}; confirm that no process is active before removing this marker.")
    started = datetime.now(timezone.utc).isoformat()
    try:
        with running.open("x") as handle:
            json.dump({"pid": os.getpid(), "started_at_utc": started}, handle)
    except FileExistsError as exc:
        raise ValueError(f"Another run is using {output_dir}") from exc
    try:
        for name in RESULT_NAMES + ("SUCCESS", "FAILED", "manifest.json", "resolved_config.yaml"):
            (output_dir / name).unlink(missing_ok=True)
        dump_resolved_config(output_dir / "resolved_config.yaml", cfg)
        result = run_experiment(cfg, output_dir)
        if code_hashes() != expected["code_sha256"]:
            raise RuntimeError("Source files changed during the run; results cannot be marked complete.")
        (output_dir / "manifest.json").write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        hashes = {name: sha256_file(output_dir / name) for name in required}
        (output_dir / "SUCCESS").write_text(json.dumps({"output_sha256": hashes, "started_at_utc": started,
                                                               "finished_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2, sort_keys=True) + "\n")
        return result
    except BaseException as exc:
        for name in RESULT_NAMES + ("SUCCESS", "manifest.json"):
            (output_dir / name).unlink(missing_ok=True)
        (output_dir / "FAILED").write_text(json.dumps({"error": repr(exc)}, indent=2) + "\n")
        raise
    finally:
        running.unlink(missing_ok=True)
