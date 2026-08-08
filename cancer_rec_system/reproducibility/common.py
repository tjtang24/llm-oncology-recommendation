"""Shared helpers for the fixed-configuration reproducibility runners."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_MANIFEST_PATH = HERE / "data_manifest.json"
ENVIRONMENT_PATH = HERE / "environment.json"


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_csv(
    path: Path,
    frame: pd.DataFrame,
    contract_name: str,
    *,
    allow_unverified: bool,
) -> Dict[str, Any]:
    contract = read_json(DATA_MANIFEST_PATH)["inputs"][contract_name]
    actual = {
        "bytes": path.stat().st_size,
        "data_rows": len(frame),
        "sha256": sha256_file(path),
        "columns_present": sorted(frame.columns.astype(str).tolist()),
    }
    checks = {
        "bytes": actual["bytes"] == contract["bytes"],
        "data_rows": actual["data_rows"] == contract["data_rows"],
        "sha256": actual["sha256"] == contract["sha256"],
        "required_columns": set(contract["required_columns"]).issubset(frame.columns),
    }
    if not all(checks.values()) and not allow_unverified:
        failures = ", ".join(name for name, passed in checks.items() if not passed)
        raise ValueError(
            f"Input {path} does not match {contract_name}: {failures}. "
            "Use --allow-unverified-inputs only for a deliberate sensitivity run."
        )
    return {"path": str(path), "actual": actual, "checks": checks}


def prepare_legacy_interactions(
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Reproduce the order-sensitive preprocessing used by MF and paper NCF."""
    data = raw.copy()
    counts: Dict[str, int] = {"raw_rows": len(data)}
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(subset=["Cancer", "Intervention"])
    counts["after_dropna_rows"] = len(data)
    data["intervention_id"] = (
        data["Intervention"].astype("category").cat.codes + 1
    )
    data["Cancer"] = data["Cancer"].astype(str).str.split(",")
    data = data.explode("Cancer").reset_index(drop=True)
    data["Cancer"] = data["Cancer"].str.lower()
    data["row_indices"] = data.index + 1
    counts["expanded_rows"] = len(data)
    treatment_counts = data.groupby("Cancer")["Intervention"].count()
    retained = treatment_counts[treatment_counts >= 20].index
    data = data[data["Cancer"].isin(retained)].copy()
    counts["filtered_rows"] = len(data)

    user_mapping = {value: index for index, value in enumerate(data["Cancer"].unique())}
    item_mapping = {
        value: index for index, value in enumerate(data["Intervention"].unique())
    }
    data["user_id"] = data["Cancer"].map(user_mapping)
    data["item_id"] = data["Intervention"].map(item_mapping)
    counts.update(
        {
            "users": len(user_mapping),
            "items": len(item_mapping),
            "unique_cancer_intervention_pairs": int(
                data[["Cancer", "Intervention"]].drop_duplicates().shape[0]
            ),
        }
    )
    return data, user_mapping, item_mapping, counts


def package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment(
    package_names: Iterable[str],
    *,
    torch_details: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_full": sys.version,
        "python_build": platform.python_compiler().strip(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: package_version(name) for name in package_names},
        "torch_execution": dict(torch_details or {}),
    }


def compare_environment(
    actual: Mapping[str, Any],
    *,
    required_packages: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    reference = read_json(ENVIRONMENT_PATH)
    expected_packages = reference["packages"]
    actual_packages = actual["packages"]
    package_names = list(required_packages or expected_packages)
    package_checks = {
        name: actual_packages.get(name) == version
        for name, version in expected_packages.items()
        if name in package_names
    }
    actual_python_build = actual.get("python_build")
    if actual_python_build is None and reference["python_build"] in actual.get(
        "python_full", ""
    ):
        actual_python_build = reference["python_build"]
    checks = {
        "python": actual["python"] == reference["python"],
        "python_build": str(actual_python_build).strip()
        == reference["python_build"].strip(),
        "platform": actual["platform"] == reference["platform"],
        "machine": actual["machine"] == reference["machine"],
        "packages": all(package_checks.values()),
    }
    torch_checks: Dict[str, bool] = {}
    if "torch" in package_names:
        expected_torch = reference["torch_execution"]
        actual_torch = dict(actual.get("torch_execution") or {})
        if (
            "mps_available_but_unused" not in actual_torch
            and "mps_available" in actual_torch
        ):
            actual_torch["mps_available_but_unused"] = actual_torch[
                "mps_available"
            ]
        torch_checks = {
            name: actual_torch.get(name) == expected_value
            for name, expected_value in expected_torch.items()
        }
        checks["torch_execution"] = bool(actual_torch) and all(
            torch_checks.values()
        )
    return {
        "all_recorded_fields_match": all(checks.values()),
        "checks": checks,
        "package_checks": package_checks,
        "torch_checks": torch_checks,
    }


def ensure_new_output(path: Path) -> Path:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists: {path}. Choose a new run directory."
        )
    path.mkdir(parents=True)
    return path


def require_new_output_path(path: Path) -> Path:
    """Fail before expensive work when a requested output already exists.

    ``ensure_new_output`` remains the atomic creation step immediately before
    artifacts are written. Calling this preflight first avoids hours of
    training when the destination is already invalid, while the later check
    still protects against another process creating the path in the meantime.
    """
    path = path.resolve()
    if path.exists():
        raise FileExistsError(
            f"Output directory already exists: {path}. Choose a new run directory."
        )
    return path


def reference_checks(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> Dict[str, Any]:
    checks = {
        name: actual.get(name) == expected_value
        for name, expected_value in expected.items()
    }
    return {
        "all_expected_outputs_exact": bool(checks) and all(checks.values()),
        "checks": checks,
        "expected": dict(expected),
        "actual": dict(actual),
    }
