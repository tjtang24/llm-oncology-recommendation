#!/usr/bin/env python3
"""Refit the manuscript MF baseline with its frozen configuration.

This is intentionally a fixed-configuration refit, not a hyperparameter search.
It writes all generated models, mappings, predictions, and manifests to a new
output directory; the default location is git-ignored.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
from surprise import Dataset, Reader, SVD

from .common import (
    collect_environment,
    compare_environment,
    ensure_new_output,
    prepare_legacy_interactions,
    read_json,
    reference_checks,
    require_new_output_path,
    sha256_array,
    sha256_file,
    verify_csv,
    write_json,
)


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PRIVATE_DATA = APP_ROOT.parent / "private_data"
DEFAULT_CONFIG = HERE / "configs" / "mf.json"
DEFAULT_DATA = PRIVATE_DATA / "summary_w_score.csv"
DEFAULT_OUTPUT = APP_ROOT / "params" / "mf-retrained"


def make_model(hyperparameters: Dict[str, Any], seed: int) -> SVD:
    return SVD(
        n_factors=int(hyperparameters["n_factors"]),
        n_epochs=int(hyperparameters["n_epochs"]),
        biased=bool(hyperparameters["biased"]),
        init_mean=hyperparameters["init_mean"],
        init_std_dev=float(hyperparameters["init_std_dev"]),
        lr_all=float(hyperparameters["lr_all"]),
        reg_all=float(hyperparameters["reg_all"]),
        random_state=seed,
    )


def full_catalog(model: SVD) -> np.ndarray:
    scores = (
        model.trainset.global_mean
        + model.bu[:, None]
        + model.bi[None, :]
        + model.pu @ model.qi.T
    )
    low, high = model.trainset.rating_scale
    return np.clip(scores, low, high).reshape(-1)


def canonical_intervention(value: object) -> str:
    return ", ".join(sorted(part.strip() for part in str(value).split(",")))


def mf_display_mapping(
    data: pd.DataFrame,
) -> tuple[Dict[str, int], Dict[int, str]]:
    """Reproduce the historical canonical display mapping after raw-ID creation."""
    display = data[["intervention_id", "Intervention"]].copy()
    display["Intervention"] = display["Intervention"].map(canonical_intervention)
    item_mapping = {
        value: index for index, value in enumerate(display["Intervention"].unique())
    }
    display["display_item_id"] = display["Intervention"].map(item_mapping)
    inverse = {item_id: name for name, item_id in item_mapping.items()}
    raw_to_display = {
        int(raw_id): inverse[int(display_id)]
        for raw_id, display_id in display[
            ["intervention_id", "display_item_id"]
        ].drop_duplicates("intervention_id").itertuples(index=False, name=None)
    }
    return item_mapping, raw_to_display


def top20_recommendations(
    model: SVD,
    data: pd.DataFrame,
    raw_to_display: Mapping[int, str],
    top_n: int = 20,
) -> pd.DataFrame:
    """Match the audited raw-item ranking and stable Python sort."""
    trainset = model.trainset
    all_raw_items = [
        trainset.to_raw_iid(inner) for inner in trainset.all_items()
    ]
    known: Dict[Any, set[Any]] = {}
    for cancer, intervention_id in data[
        ["Cancer", "intervention_id"]
    ].itertuples(index=False, name=None):
        known.setdefault(cancer, set()).add(intervention_id)

    rows = []
    for cancer in trainset._raw2inner_id_users:
        candidates = [
            (cancer, raw_item, trainset.global_mean)
            for raw_item in all_raw_items
            if raw_item not in known[cancer]
        ]
        predictions = model.test(candidates)
        predictions.sort(key=lambda prediction: prediction.est, reverse=True)
        for rank, prediction in enumerate(predictions[:top_n], 1):
            raw_item = int(prediction.iid)
            rows.append(
                {
                    "Cancer": cancer,
                    "Rank": rank,
                    "Surprise_raw_iid": raw_item,
                    "Intervention": raw_to_display[raw_item],
                    "Predicted_Score": float(prediction.est),
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit the fixed manuscript MF configuration."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help=(
            "Run on a different input while recording that reference checks "
            "do not apply."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_new_output_path(args.output)
    config = read_json(DEFAULT_CONFIG)
    seed = int(config["seed"])
    np.random.seed(seed)

    raw = pd.read_csv(args.data)
    input_record = verify_csv(
        args.data,
        raw,
        "training_snapshot",
        allow_unverified=args.allow_unverified_inputs,
    )
    data, user_mapping, _raw_item_mapping, counts = prepare_legacy_interactions(raw)
    display_item_mapping, raw_to_display = mf_display_mapping(data)
    counts["canonical_display_items"] = len(display_item_mapping)
    expected_counts = read_json(HERE / "data_manifest.json")[
        "processed_training_contract"
    ]
    count_checks = {
        "rows": counts["filtered_rows"] == expected_counts["rows"],
        "cancers": counts["users"] == expected_counts["cancers"],
        "items": counts["items"]
        == expected_counts["order_sensitive_intervention_ids"],
        "pairs": counts["unique_cancer_intervention_pairs"]
        == expected_counts["unique_cancer_intervention_pairs"],
        "canonical_display_items": counts["canonical_display_items"]
        == expected_counts["canonical_display_interventions"],
    }
    if not all(count_checks.values()) and not args.allow_unverified_inputs:
        raise ValueError(f"Processed data contract mismatch: {count_checks}")

    reader = Reader(rating_scale=(data["Score"].min(), data["Score"].max()))
    surprise_data = Dataset.load_from_df(
        data[["Cancer", "intervention_id", "Score"]], reader
    )
    model = make_model(config["hyperparameters"], seed)
    model.fit(surprise_data.build_full_trainset())

    output = ensure_new_output(args.output)
    model_path = output / "final_model.pkl"
    with model_path.open("wb") as stream:
        pickle.dump(model, stream)

    recommendations = top20_recommendations(model, data, raw_to_display)
    recommendation_path = output / "top20_all_cancers.csv"
    recommendations.to_csv(recommendation_path, index=False)

    catalog = full_catalog(model)
    catalog_path = output / "full_catalog_predictions.npy"
    np.save(catalog_path, catalog)
    np.save(output / "bu.npy", model.bu)
    np.save(output / "bi.npy", model.bi)
    np.save(output / "pu.npy", model.pu)
    np.save(output / "qi.npy", model.qi)
    user_mapping_path = output / "user_mapping.json"
    write_json(user_mapping_path, user_mapping)
    item_mapping_path = output / "item_mapping.json"
    write_json(item_mapping_path, display_item_mapping)

    environment = collect_environment(
        [
            "numpy",
            "pandas",
            "scikit-learn",
            "scikit-surprise",
            "scipy",
            "joblib",
            "threadpoolctl",
            "torch",
        ]
    )
    hashes = {
        "model_file_sha256": sha256_file(model_path),
        "bu_sha256": sha256_array(model.bu),
        "bi_sha256": sha256_array(model.bi),
        "pu_sha256": sha256_array(model.pu),
        "qi_sha256": sha256_array(model.qi),
        "catalog_raw_sha256": sha256_array(catalog),
        "training_top20_csv_sha256": sha256_file(recommendation_path),
        "item_mapping_json_sha256": sha256_file(item_mapping_path),
        "user_mapping_json_sha256": sha256_file(user_mapping_path),
    }
    output_comparison = reference_checks(config["reference_outputs"], hashes)
    reference_contract_checks = {
        "committed_config": config == read_json(DEFAULT_CONFIG),
        "training_input": all(input_record["checks"].values()),
        "processed_data": all(count_checks.values()),
        "scientific_outputs": output_comparison["all_expected_outputs_exact"],
    }
    manifest = {
        "model": "mf",
        "mode": config["training_mode"],
        "seed": seed,
        "config": config,
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "input": input_record,
        "processed_data": counts,
        "processed_data_checks": count_checks,
        "environment": environment,
        "environment_comparison": compare_environment(
            environment,
            required_packages=[
                "numpy",
                "pandas",
                "scikit-learn",
                "scikit-surprise",
                "scipy",
                "joblib",
                "threadpoolctl",
            ],
        ),
        "outputs": {
            "directory": str(output),
            "catalog_shape": [model.trainset.n_users, model.trainset.n_items],
            "top20_rows": len(recommendations),
            "hashes": hashes,
        },
        "reference_comparison": output_comparison,
        "reference_contract": {
            "all_checks_exact": all(reference_contract_checks.values()),
            "checks": reference_contract_checks,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps(manifest["reference_contract"], indent=2))
    return 0 if manifest["reference_contract"]["all_checks_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
