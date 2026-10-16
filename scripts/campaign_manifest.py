#!/usr/bin/env python3
"""Validate a campaign's design and identity before running any configurations."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments_core.validation import parse_grid


def initialize_campaign(output_root: Path, identity: dict) -> None:
    """Preserve an existing campaign; only an identical invocation may resume it."""
    output_root = Path(output_root)
    path = output_root / "campaign_manifest.json"
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"Invalid campaign manifest in {output_root}; use a new run identifier.") from exc
        if existing != identity:
            raise ValueError("Campaign design, source, or environment changed; use a new run identifier.")
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"Existing directory {output_root} has no campaign manifest; use a new run identifier.")
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError(f"Concurrent initialization of {output_root}; retry after the other process finishes.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    commands = parser.add_subparsers(dest="kind", required=True)
    comparison = commands.add_parser("comparison")
    comparison.add_argument("--coeff-grid", required=True)
    comparison.add_argument("--include-poly", choices=("0", "1"), default="0")
    feedback = commands.add_parser("feedback")
    feedback.add_argument("--prob-grid", required=True)
    feedback.add_argument("--noise-grid", required=True)
    args = parser.parse_args()
    if args.kind == "comparison":
        design = {"kind": args.kind,
                  "coefficients": parse_grid(args.coeff_grid, "coefficient grid", positive=True),
                  "include_poly": args.include_poly == "1"}
        config_names = ["main_linear_t10000_s10.yaml", "main_kernel_t1000_s10.yaml"]
    else:
        design = {"kind": args.kind,
                  "swap_probabilities": parse_grid(args.prob_grid, "probability grid", probability=True),
                  "relative_noise_levels": parse_grid(args.noise_grid, "noise grid")}
        config_names = ["main_linear_t10000_s10.yaml"]
    from experiments_core.reproducibility import code_hashes, environment, sha256_file
    identity = {"schema_version": 1, "design": design, "source_sha256": code_hashes(),
                "configuration_sha256": {name: sha256_file(ROOT / "configs" / name) for name in config_names},
                "environment": environment()}
    initialize_campaign(args.output_root, identity)


if __name__ == "__main__":
    main()
