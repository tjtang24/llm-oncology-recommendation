#!/usr/bin/env python3
"""Verify fixed-refit inputs and actual artifacts against committed contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd

from .common import (
    compare_environment,
    read_json,
    sha256_array,
    sha256_file,
    verify_csv,
)


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PRIVATE_DATA = APP_ROOT.parent / "private_data"
DATA_MANIFEST = HERE / "data_manifest.json"
CONFIG_PATHS = {
    "mf": HERE / "configs" / "mf.json",
    "ncf": HERE / "configs" / "ncf.json",
}
REQUIRED_PACKAGES = {
    "mf": [
        "numpy",
        "pandas",
        "scikit-learn",
        "scikit-surprise",
        "scipy",
        "joblib",
        "threadpoolctl",
    ],
    "ncf": [
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "joblib",
        "threadpoolctl",
        "torch",
    ],
}
HASH_KEYS = {
    "mf": [
        "model_file_sha256",
        "bu_sha256",
        "bi_sha256",
        "pu_sha256",
        "qi_sha256",
        "catalog_raw_sha256",
        "training_top20_csv_sha256",
        "item_mapping_json_sha256",
        "user_mapping_json_sha256",
    ],
    "ncf": [
        "model_file_sha256",
        "canonical_tensor_sha256",
        "catalog_raw_sha256",
        "training_top20_csv_sha256",
        "user_mapping_json_sha256",
        "item_mapping_json_sha256",
        "epoch_history_csv_sha256",
    ],
}


def load_run(path: Path) -> Dict[str, Any]:
    manifest = path if path.name == "run_manifest.json" else path / "run_manifest.json"
    run = read_json(manifest)
    run["_verified_directory"] = str(manifest.resolve().parent)
    return run


def canonical_state_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def load_state_dict(path: Path) -> Mapping[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True)


def actual_output_hashes(run: Dict[str, Any], model: str) -> Dict[str, str]:
    root = Path(run["_verified_directory"])
    if model == "mf":
        return {
            "model_file_sha256": sha256_file(root / "final_model.pkl"),
            "bu_sha256": sha256_array(
                np.load(root / "bu.npy", allow_pickle=False)
            ),
            "bi_sha256": sha256_array(
                np.load(root / "bi.npy", allow_pickle=False)
            ),
            "pu_sha256": sha256_array(
                np.load(root / "pu.npy", allow_pickle=False)
            ),
            "qi_sha256": sha256_array(
                np.load(root / "qi.npy", allow_pickle=False)
            ),
            "catalog_raw_sha256": sha256_array(
                np.load(root / "full_catalog_predictions.npy", allow_pickle=False)
            ),
            "training_top20_csv_sha256": sha256_file(
                root / "top20_all_cancers.csv"
            ),
            "item_mapping_json_sha256": sha256_file(root / "item_mapping.json"),
            "user_mapping_json_sha256": sha256_file(root / "user_mapping.json"),
        }

    state = load_state_dict(root / "final_model.pth")
    return {
        "model_file_sha256": sha256_file(root / "final_model.pth"),
        "canonical_tensor_sha256": canonical_state_hash(state),
        "catalog_raw_sha256": sha256_array(
            np.load(root / "full_catalog_predictions.npy", allow_pickle=False)
        ),
        "training_top20_csv_sha256": sha256_file(
            root / "top20_all_cancers.csv"
        ),
        "user_mapping_json_sha256": sha256_file(root / "user_mapping.json"),
        "item_mapping_json_sha256": sha256_file(root / "item_mapping.json"),
        "epoch_history_csv_sha256": sha256_file(root / "epoch_history.csv"),
    }


def direct_input_records(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    training = pd.read_csv(args.data)
    return {
        "training_snapshot": verify_csv(
            args.data,
            training,
            "training_snapshot",
            allow_unverified=False,
        )
    }


def reference_status(
    run: Dict[str, Any],
    model: str,
    inputs: Mapping[str, Dict[str, Any]],
) -> Dict[str, Any]:
    committed_config = read_json(CONFIG_PATHS[model])
    data_contract = read_json(DATA_MANIFEST)
    hashes = actual_output_hashes(run, model)
    expected = committed_config["reference_outputs"]
    output_checks = {
        key: hashes.get(key) == expected[key] for key in HASH_KEYS[model]
    }

    input_key = "input" if model == "mf" else "training_input"
    run_input = run.get(input_key, {})
    direct_training = inputs["training_snapshot"]
    processed = run.get("processed_data", {})
    expected_processed = data_contract["processed_training_contract"]
    processed_checks = {
        "rows": processed.get("filtered_rows") == expected_processed["rows"],
        "cancers": processed.get("users") == expected_processed["cancers"],
        "items": processed.get("items")
        == expected_processed["order_sensitive_intervention_ids"],
        "pairs": processed.get("unique_cancer_intervention_pairs")
        == expected_processed["unique_cancer_intervention_pairs"],
    }
    if model == "mf":
        processed_checks["canonical_display_items"] = processed.get(
            "canonical_display_items"
        ) == expected_processed["canonical_display_interventions"]

    embedded_training_config = {
        key: value
        for key, value in run.get("config", {}).items()
        if key not in {"reference_outputs", "audit_result"}
    }
    committed_training_config = {
        key: value
        for key, value in committed_config.items()
        if key not in {"reference_outputs", "audit_result"}
    }
    recorded_config_sha = run.get("config_sha256")
    contract_checks = {
        "model_name": run.get("model") == model,
        "committed_training_config": embedded_training_config
        == committed_training_config,
        "committed_config_sha256": recorded_config_sha is None
        or recorded_config_sha == sha256_file(CONFIG_PATHS[model]),
        "training_input_direct": all(direct_training["checks"].values()),
        "training_input_manifest": run_input.get("actual", {}).get("sha256")
        == direct_training["actual"]["sha256"],
        "processed_data": all(processed_checks.values()),
        "actual_artifacts": all(output_checks.values()),
    }
    if model == "ncf":
        contract_checks["epoch_trace"] = (
            run.get("epoch_trace") == committed_config["reference_epoch_trace"]
        )

    environment = compare_environment(
        run["environment"], required_packages=REQUIRED_PACKAGES[model]
    )
    return {
        "all_reference_contract_exact": all(contract_checks.values()),
        "contract_checks": contract_checks,
        "output_checks": output_checks,
        "processed_checks": processed_checks,
        "actual_hashes": hashes,
        "environment_matches_record": environment["all_recorded_fields_match"],
        "environment_checks": environment,
        "input_sha256": direct_training["actual"]["sha256"],
    }


def compare_runs(
    left: Dict[str, Any], right: Dict[str, Any], model: str
) -> Dict[str, Any]:
    left_hashes = actual_output_hashes(left, model)
    right_hashes = actual_output_hashes(right, model)
    hash_checks = {
        key: left_hashes[key] == right_hashes[key] for key in HASH_KEYS[model]
    }
    committed_config = read_json(CONFIG_PATHS[model])
    committed_training_config = {
        key: value
        for key, value in committed_config.items()
        if key not in {"reference_outputs", "audit_result"}
    }
    left_training_config = {
        key: value
        for key, value in left["config"].items()
        if key not in {"reference_outputs", "audit_result"}
    }
    right_training_config = {
        key: value
        for key, value in right["config"].items()
        if key not in {"reference_outputs", "audit_result"}
    }
    checks = {
        "model_name": left["model"] == right["model"] == model,
        "seed": left["seed"] == right["seed"],
        "committed_training_config": left_training_config
        == right_training_config
        == committed_training_config,
        "training_input": (
            left["input" if model == "mf" else "training_input"]["actual"][
                "sha256"
            ]
            == right["input" if model == "mf" else "training_input"]["actual"][
                "sha256"
            ]
        ),
        "all_scientific_hashes": all(hash_checks.values()),
    }
    if model == "ncf":
        checks["epoch_trace"] = left["epoch_trace"] == right["epoch_trace"]
    return {
        "all_repeat_run_checks_exact": all(checks.values()),
        "checks": checks,
        "hash_checks": hash_checks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=sorted(HASH_KEYS))
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path, nargs="?")
    parser.add_argument(
        "--data",
        type=Path,
        default=PRIVATE_DATA / "summary_w_score.csv",
        help="Private training snapshot; its bytes are verified directly.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = direct_input_records(args)
    run_a = load_run(args.run_a)
    result: Dict[str, Any] = {
        "model": args.model,
        "run_a_vs_reference": reference_status(run_a, args.model, inputs),
    }
    success = result["run_a_vs_reference"]["all_reference_contract_exact"]
    environments_match = result["run_a_vs_reference"][
        "environment_matches_record"
    ]
    if args.run_b is not None:
        run_b = load_run(args.run_b)
        result["run_b_vs_reference"] = reference_status(
            run_b, args.model, inputs
        )
        result["run_a_vs_run_b"] = compare_runs(run_a, run_b, args.model)
        success = success and result["run_b_vs_reference"][
            "all_reference_contract_exact"
        ]
        success = success and result["run_a_vs_run_b"][
            "all_repeat_run_checks_exact"
        ]
        environments_match = environments_match and result[
            "run_b_vs_reference"
        ]["environment_matches_record"]

    if success and environments_match:
        result["status"] = "PASS_SAME_RECORDED_ENVIRONMENT"
    elif success:
        result["status"] = "OUTPUTS_EXACT_ENVIRONMENT_DIFFERS"
    else:
        result["status"] = "FAIL"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
