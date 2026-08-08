#!/usr/bin/env python3
"""Verify a completed CV replay against the committed reference contract.

Given a finished ``cv_ncf`` (or ``cv_hybrid``) run directory, this re-reads the
run's ``summary.json`` and compares its aggregate metrics, per-fold metrics, and
fold-split fingerprints against the frozen reference in ``configs/cv_<model>.json``.
It exits 0 only when every recorded reference field is reproduced exactly.

This is a metric-reproduction check for the source-faithful historical CV. It is
not a claim of clean nested cross-validation; see the leakage warning embedded in
the config and the reproducibility documentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .common import read_json


HERE = Path(__file__).resolve().parent
CONFIG_PATHS = {
    "ncf": HERE / "configs" / "cv_ncf.json",
    "hybrid": HERE / "configs" / "cv_hybrid.json",
}

PER_FOLD_FIELDS = ("rmse", "mse", "mae", "validation_rows")


def load_summary(path: Path) -> Dict[str, Any]:
    summary = path if path.name == "summary.json" else path / "summary.json"
    return read_json(summary)


def aggregate_checks(
    summary: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, bool]:
    actual = summary["aggregate"]
    expected = config["reference_aggregate"]
    return {key: actual.get(key) == value for key, value in expected.items()}


def per_fold_checks(
    summary: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, bool]:
    actual = {int(record["fold"]): record for record in summary["per_fold"]}
    checks: Dict[str, bool] = {}
    for reference in config["reference_per_fold"]:
        fold = int(reference["fold"])
        record = actual.get(fold, {})
        for field in PER_FOLD_FIELDS:
            checks[f"fold_{fold:02d}_{field}"] = (
                record.get(field) == reference[field]
            )
        expected_sha = reference.get("validation_indices_sha256")
        if expected_sha is not None:
            observed = next(
                (
                    diag.get("validation_indices_sha256")
                    for diag in summary.get("split_diagnostics", [])
                    if int(diag.get("fold", -1)) == fold
                ),
                None,
            )
            checks[f"fold_{fold:02d}_validation_indices_sha256"] = (
                observed == expected_sha
            )
    return checks


def configuration_checks(
    summary: Dict[str, Any], config: Dict[str, Any]
) -> Dict[str, bool]:
    actual = summary.get("configuration", {})
    expected = config["configuration"]
    return {key: actual.get(key) == value for key, value in expected.items()}


def verify(model: str, run_dir: Path) -> Dict[str, Any]:
    config = read_json(CONFIG_PATHS[model])
    summary = load_summary(run_dir)
    aggregate = aggregate_checks(summary, config)
    per_fold = per_fold_checks(summary, config)
    configuration = configuration_checks(summary, config)
    protocol_ok = summary.get("protocol") == config.get("protocol", "faithful")
    all_exact = (
        protocol_ok
        and all(aggregate.values())
        and all(per_fold.values())
        and all(configuration.values())
    )
    return {
        "model": model,
        "run_directory": str(run_dir.resolve()),
        "protocol": summary.get("protocol"),
        "protocol_matches_reference": protocol_ok,
        "reported_manuscript_rmse": config.get("reported_manuscript_rmse"),
        "reproduced_arithmetic_mean_rmse": summary["aggregate"].get(
            "arithmetic_mean_rmse"
        ),
        "reproduced_pooled_rmse": summary["aggregate"].get("pooled_rmse"),
        "all_reference_metrics_exact": all_exact,
        "aggregate_checks": aggregate,
        "per_fold_checks": per_fold,
        "configuration_checks": configuration,
        "leakage_warning": config.get("leakage_warning"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(CONFIG_PATHS))
    parser.add_argument("run_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = verify(args.model, args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_reference_metrics_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
