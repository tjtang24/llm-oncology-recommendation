"""Create and validate the eight-file Hybrid NCF inference bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import torch

from cancer_rec_system.model import HybridNCF
from cancer_rec_system.training.features import mapping_names_by_id


BUNDLE_FILES = {
    "best_hyperparameters.json",
    "final_model.pth",
    "item_features_tensor.pt",
    "item_mapping.json",
    "known_items.json",
    "metadata_lookup.json",
    "user_features_tensor.pt",
    "user_mapping.json",
}

REQUIRED_HYPERPARAMETERS = {
    "embedding_dim",
    "gmf_dim",
    "hidden_units",
    "dropout_rate",
    "learning_rate",
    "batch_size",
    "weight_decay",
}


def validate_hyperparameters(hyperparameters: Mapping[str, Any]) -> None:
    """Reject malformed model and optimization parameters before training."""
    missing = sorted(REQUIRED_HYPERPARAMETERS - set(hyperparameters))
    if missing:
        raise ValueError(f"Hyperparameter file is missing: {missing}.")

    for name in ("embedding_dim", "gmf_dim", "batch_size"):
        value = hyperparameters[name]
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")

    hidden_units = hyperparameters["hidden_units"]
    if (
        not isinstance(hidden_units, list)
        or not hidden_units
        or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            for value in hidden_units
        )
    ):
        raise ValueError("hidden_units must be a non-empty list of positive integers.")
    if (
        "num_hidden_layers" in hyperparameters
        and hyperparameters["num_hidden_layers"] != len(hidden_units)
    ):
        raise ValueError("num_hidden_layers does not match hidden_units.")

    for name, lower, upper in (
        ("dropout_rate", 0.0, 1.0),
        ("learning_rate", 0.0, None),
        ("weight_decay", 0.0, None),
    ):
        value = hyperparameters[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be numeric.")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be finite.")
        if name == "dropout_rate":
            if not lower <= numeric < upper:
                raise ValueError("dropout_rate must be in [0, 1).")
        elif name == "learning_rate":
            if numeric <= lower:
                raise ValueError("learning_rate must be positive.")
        elif numeric < lower:
            raise ValueError("weight_decay cannot be negative.")


def require_new_bundle_destination(output_dir: str | Path) -> Path:
    """Resolve an output path and refuse any pre-existing filesystem entry."""
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(
            "Bundle destination already exists; choose a new versioned path: "
            f"{destination}"
        )
    return destination


def _strict_json_dump(value: Any, destination: Path) -> None:
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(
            value,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def _cpu_state_dict(model: HybridNCF) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }


def _as_feature_tensor(value: torch.Tensor, label: str) -> torch.Tensor:
    tensor = value.detach().cpu().to(dtype=torch.float32).contiguous()
    if tensor.ndim != 2:
        raise ValueError(f"{label} must be a two-dimensional tensor.")
    if tensor.shape[1] < 1:
        raise ValueError(f"{label} must contain at least one feature.")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} contains a non-finite value.")
    if not torch.all((tensor == 0) | (tensor == 1)):
        raise ValueError(f"{label} must contain binary features.")
    return tensor


def validate_bundle(bundle_dir: str | Path) -> None:
    """Validate a staged or published inference bundle."""
    root = Path(bundle_dir)
    actual_files = {
        path.name
        for path in root.iterdir()
        if path.is_file()
    }
    if actual_files != BUNDLE_FILES:
        missing = sorted(BUNDLE_FILES - actual_files)
        extra = sorted(actual_files - BUNDLE_FILES)
        raise ValueError(
            f"Inference bundle file mismatch; missing={missing}, extra={extra}."
        )

    with (root / "best_hyperparameters.json").open(
        "r", encoding="utf-8"
    ) as stream:
        hyperparameters = json.load(stream)
    with (root / "user_mapping.json").open("r", encoding="utf-8") as stream:
        user_mapping = json.load(stream)
    with (root / "item_mapping.json").open("r", encoding="utf-8") as stream:
        item_mapping = json.load(stream)
    with (root / "known_items.json").open("r", encoding="utf-8") as stream:
        known_items = json.load(stream)
    with (root / "metadata_lookup.json").open("r", encoding="utf-8") as stream:
        metadata_lookup = json.load(stream)

    validate_hyperparameters(hyperparameters)
    hidden_units = hyperparameters["hidden_units"]

    user_names = mapping_names_by_id(user_mapping)
    item_names = mapping_names_by_id(item_mapping)
    if set(known_items) != {str(user_id) for user_id in user_mapping.values()}:
        raise ValueError("Known-item keys do not match the user mapping.")
    mapped_items = set(item_names)
    for user_id, interventions in known_items.items():
        if (
            not isinstance(interventions, list)
            or not all(isinstance(item, str) for item in interventions)
        ):
            raise ValueError(f"Known items for user {user_id} are not a list.")
        if not set(interventions).issubset(mapped_items):
            raise ValueError(f"Known items for user {user_id} are not mapped.")
    if not mapped_items.issubset(metadata_lookup):
        raise ValueError("Metadata does not cover every mapped intervention.")
    required_metadata_fields = {"Cancer", "Phases", "Year"}
    for intervention, metadata in metadata_lookup.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Metadata for {intervention} is not an object.")
        if set(metadata) != required_metadata_fields:
            raise ValueError(
                f"Metadata fields for {intervention} must be "
                f"{sorted(required_metadata_fields)}."
            )
        if not isinstance(metadata["Cancer"], str):
            raise ValueError(f"Metadata Cancer for {intervention} must be text.")
        if not isinstance(metadata["Phases"], str):
            raise ValueError(f"Metadata Phases for {intervention} must be text.")
        year = metadata["Year"]
        if year is not None and (
            isinstance(year, bool)
            or not isinstance(year, (int, float))
            or not math.isfinite(float(year))
        ):
            raise ValueError(f"Metadata Year for {intervention} is invalid.")

    user_features = torch.load(
        root / "user_features_tensor.pt",
        map_location="cpu",
        weights_only=True,
    )
    item_features = torch.load(
        root / "item_features_tensor.pt",
        map_location="cpu",
        weights_only=True,
    )
    user_features = _as_feature_tensor(user_features, "user features")
    item_features = _as_feature_tensor(item_features, "item features")
    if user_features.shape[0] != len(user_names):
        raise ValueError("User feature rows do not match the user mapping.")
    if item_features.shape[0] != len(item_names):
        raise ValueError("Item feature rows do not match the item mapping.")
    if user_features.shape[1] != item_features.shape[1]:
        raise ValueError("User and item feature dimensions do not match.")

    model = HybridNCF(
        num_users=len(user_mapping),
        num_items=len(item_mapping),
        embedding_dim=int(hyperparameters["embedding_dim"]),
        user_feature_dim=user_features.shape[1],
        item_feature_dim=item_features.shape[1],
        gmf_dim=int(hyperparameters["gmf_dim"]),
        hidden_units=[int(value) for value in hidden_units],
        dropout_rate=float(hyperparameters["dropout_rate"]),
    )
    state_dict = torch.load(
        root / "final_model.pth",
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.tensor([0], dtype=torch.long),
            torch.tensor([0], dtype=torch.long),
            user_features[:1],
            item_features[:1],
        )
    if prediction.shape != (1,) or not torch.isfinite(prediction).all():
        raise ValueError("Model bundle failed a finite forward-pass check.")


def export_inference_bundle(
    output_dir: str | Path,
    *,
    model: HybridNCF,
    hyperparameters: Mapping[str, Any],
    user_mapping: Mapping[str, int],
    item_mapping: Mapping[str, int],
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    known_items: Mapping[str, Sequence[str]],
    metadata_lookup: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Stage, validate, and publish a complete inference bundle.

    The destination must be new. Promotion of a reviewed bundle into production
    is deliberately kept separate from training and export.
    """
    destination = require_new_bundle_destination(output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    hyperparameter_payload = dict(hyperparameters)
    user_mapping_payload = {
        str(name): int(entity_id)
        for name, entity_id in user_mapping.items()
    }
    item_mapping_payload = {
        str(name): int(entity_id)
        for name, entity_id in item_mapping.items()
    }
    known_items_payload = {
        str(user_id): [str(item) for item in items]
        for user_id, items in known_items.items()
    }
    metadata_payload = {
        str(intervention): dict(metadata)
        for intervention, metadata in metadata_lookup.items()
    }
    user_feature_tensor = _as_feature_tensor(user_features, "user features")
    item_feature_tensor = _as_feature_tensor(item_features, "item features")

    stage = Path(
        tempfile.mkdtemp(
            prefix=".hybrid-ncf-bundle-",
            dir=destination.parent,
        )
    )
    try:
        _strict_json_dump(
            hyperparameter_payload,
            stage / "best_hyperparameters.json",
        )
        _strict_json_dump(user_mapping_payload, stage / "user_mapping.json")
        _strict_json_dump(item_mapping_payload, stage / "item_mapping.json")
        _strict_json_dump(known_items_payload, stage / "known_items.json")
        _strict_json_dump(metadata_payload, stage / "metadata_lookup.json")
        torch.save(_cpu_state_dict(model), stage / "final_model.pth")
        torch.save(user_feature_tensor, stage / "user_features_tensor.pt")
        torch.save(item_feature_tensor, stage / "item_features_tensor.pt")
        validate_bundle(stage)

        require_new_bundle_destination(destination)
        os.replace(stage, destination)

        validate_bundle(destination)
        return destination
    finally:
        if stage.exists():
            shutil.rmtree(stage)
