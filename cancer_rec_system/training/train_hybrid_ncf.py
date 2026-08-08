"""Train Hybrid NCF and export a production-compatible inference bundle.

Run this module from the repository root:

    python -m cancer_rec_system.training.train_hybrid_ncf

The local CSV inputs are intentionally git-ignored and must be supplied
separately. This fixed-hyperparameter path is not a reproduction of the
historical cross-validation or hyperparameter-search results.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from cancer_rec_system.model import HybridNCF
from cancer_rec_system.training.export_bundle import (
    export_inference_bundle,
    require_new_bundle_destination,
    validate_hyperparameters,
)
from cancer_rec_system.training.features import (
    build_feature_frames,
    build_intervention_vectors,
    build_known_items,
    build_mappings,
    build_metadata_lookup,
    load_source_frames,
    mapping_names_by_id,
    prepare_summary_frames,
)


APP_ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DATA = APP_ROOT.parent / "private_data"
DEFAULT_HYPERPARAMETERS = (
    APP_ROOT / "params" / "cancer-ncf-pretrain" / "best_hyperparameters.json"
)
DEFAULT_OUTPUT = APP_ROOT / "params" / "cancer-ncf-pretrain-retrained"
DEFAULT_DATA_MANIFEST = APP_ROOT / "training" / "data_manifest.json"


class HybridGMF(nn.Module):
    """Pretraining model for the generalized matrix-factorization path."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_feature_dim: int,
        item_feature_dim: int,
        gmf_dim: int,
    ) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gmf_user_proj = nn.Linear(
            embedding_dim + user_feature_dim,
            gmf_dim,
        )
        self.gmf_item_proj = nn.Linear(
            embedding_dim + item_feature_dim,
            gmf_dim,
        )
        self.output_layer = nn.Linear(gmf_dim, 1)

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_vector = torch.cat(
            (self.user_embedding(user_input), user_features),
            dim=-1,
        )
        item_vector = torch.cat(
            (self.item_embedding(item_input), item_features),
            dim=-1,
        )
        output = self.output_layer(
            self.gmf_user_proj(user_vector) * self.gmf_item_proj(item_vector)
        )
        return output.view(-1)


class HybridMLP(nn.Module):
    """Pretraining model for the multilayer-perceptron path."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        user_feature_dim: int,
        item_feature_dim: int,
        hidden_units: list[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if not hidden_units:
            raise ValueError("hidden_units must contain at least one layer size.")
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        input_dim = (
            embedding_dim
            + user_feature_dim
            + embedding_dim
            + item_feature_dim
        )
        layers: list[nn.Module] = []
        for units in hidden_units:
            layers.extend(
                (
                    nn.Linear(input_dim, units),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                )
            )
            input_dim = units
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_units[-1], 1)

    def forward(
        self,
        user_input: torch.Tensor,
        item_input: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        user_vector = torch.cat(
            (self.user_embedding(user_input), user_features),
            dim=-1,
        )
        item_vector = torch.cat(
            (self.item_embedding(item_input), item_features),
            dim=-1,
        )
        output = self.output_layer(
            self.mlp_layers(torch.cat((user_vector, item_vector), dim=-1))
        )
        return output.view(-1)


@dataclass(frozen=True)
class PreparedTrainingData:
    interactions: pd.DataFrame
    user_mapping: dict[str, int]
    item_mapping: dict[str, int]
    user_features: torch.Tensor
    item_features: torch.Tensor
    known_items: dict[str, list[str]]
    metadata_lookup: dict[str, dict[str, Any]]
    feature_names: list[str]


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch random-number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic Torch training device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if (
        device.type == "mps"
        and not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    ):
        raise ValueError("MPS was requested but is not available.")
    return device


def load_hyperparameters(path: str | Path) -> dict[str, Any]:
    """Load and type-check the fixed Hybrid NCF configuration."""
    with Path(path).open("r", encoding="utf-8") as stream:
        hyperparameters = json.load(stream)
    validate_hyperparameters(hyperparameters)
    return hyperparameters


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_snapshots(
    *,
    summary_path: Path,
    targeted_path: Path,
    summary: pd.DataFrame,
    targeted: pd.DataFrame,
    manifest_path: Path,
    allow_unverified: bool,
) -> dict[str, Any]:
    """Check local inputs against the recorded private-snapshot fingerprints."""
    try:
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
        expected_inputs = manifest["inputs"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        if not allow_unverified:
            raise ValueError(
                f"Unable to verify the training data manifest: {exc}"
            ) from exc
        expected_inputs = {}

    sources = {
        "summary_w_score.csv": (summary_path, summary),
        "targeted_therapy.csv": (targeted_path, targeted),
    }
    actual_inputs: dict[str, dict[str, Any]] = {}
    mismatches: list[str] = []
    for logical_name, (path, frame) in sources.items():
        actual = {
            "bytes": path.stat().st_size,
            "data_rows": len(frame),
            "sha256": sha256(path),
        }
        actual_inputs[logical_name] = actual
        expected = expected_inputs.get(logical_name, {})
        for field in ("bytes", "data_rows", "sha256"):
            if expected.get(field) != actual[field]:
                mismatches.append(
                    f"{logical_name}.{field}: "
                    f"expected={expected.get(field)!r}, "
                    f"actual={actual[field]!r}"
                )

    if mismatches and not allow_unverified:
        raise ValueError(
            "Training inputs do not match the recorded research snapshots. "
            "Use --allow-unverified-inputs only after reviewing the new data:\n"
            + "\n".join(mismatches)
        )
    if mismatches:
        print(
            "Warning: continuing with explicitly allowed, unverified inputs:\n"
            + "\n".join(mismatches)
        )
    return {
        "verified_against_manifest": not mismatches,
        "reference_manifest_sha256": (
            sha256(manifest_path) if manifest_path.is_file() else None
        ),
        "inputs": actual_inputs,
    }


def training_code_hashes() -> dict[str, str]:
    """Fingerprint the source modules that define a training run."""
    paths = (
        APP_ROOT / "model.py",
        APP_ROOT / "training" / "features.py",
        APP_ROOT / "training" / "export_bundle.py",
        Path(__file__).resolve(),
    )
    return {
        path.relative_to(APP_ROOT).as_posix(): sha256(path)
        for path in paths
    }


def prepare_training_data(
    summary: pd.DataFrame,
    targeted: pd.DataFrame,
    *,
    min_interactions_per_cancer: int = 20,
) -> PreparedTrainingData:
    """Build mappings, aligned feature tensors, novelty sets, and metadata."""
    metadata, interactions = prepare_summary_frames(
        summary,
        min_interactions_per_cancer=min_interactions_per_cancer,
    )
    user_mapping, item_mapping = build_mappings(interactions)
    interactions = interactions.copy()
    interactions["user_id"] = interactions["Cancer"].map(user_mapping)
    interactions["item_id"] = interactions["Intervention"].map(item_mapping)

    cancer_vectors, drug_vectors, feature_names = build_feature_frames(
        interactions,
        targeted,
    )
    intervention_vectors = build_intervention_vectors(
        item_mapping,
        drug_vectors,
    )
    user_names = mapping_names_by_id(user_mapping)
    item_names = mapping_names_by_id(item_mapping)
    user_features = torch.tensor(
        cancer_vectors.reindex(user_names).fillna(0.0).to_numpy(),
        dtype=torch.float32,
    )
    item_features = torch.tensor(
        intervention_vectors.reindex(item_names).fillna(0.0).to_numpy(),
        dtype=torch.float32,
    )

    return PreparedTrainingData(
        interactions=interactions,
        user_mapping=user_mapping,
        item_mapping=item_mapping,
        user_features=user_features,
        item_features=item_features,
        known_items=build_known_items(interactions, user_mapping),
        metadata_lookup=build_metadata_lookup(metadata),
        feature_names=feature_names,
    )


def split_rows_by_pair(
    interactions: pd.DataFrame,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a warm-item split with duplicate pairs kept together.

    Every validation user and item retains at least one distinct pair in the
    training set. This mirrors ranking among catalog items whose embeddings
    have received training updates.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one.")

    pair_keys = list(
        zip(
            interactions["user_id"].astype(int),
            interactions["item_id"].astype(int),
        )
    )
    unique_pairs = list(dict.fromkeys(pair_keys))
    if len(unique_pairs) < 2:
        raise ValueError("At least two unique user-item pairs are required.")

    validation_pair_target = max(
        1,
        min(
            len(unique_pairs) - 1,
            round(len(unique_pairs) * validation_fraction),
        ),
    )
    rng = np.random.default_rng(seed)
    user_pairs = Counter(user_id for user_id, _item_id in unique_pairs)
    item_pairs = Counter(item_id for _user_id, item_id in unique_pairs)
    validation_set: set[tuple[int, int]] = set()
    for index in rng.permutation(len(unique_pairs)):
        user_id, item_id = unique_pairs[index]
        if user_pairs[user_id] <= 1 or item_pairs[item_id] <= 1:
            continue
        validation_set.add((user_id, item_id))
        user_pairs[user_id] -= 1
        item_pairs[item_id] -= 1
        if len(validation_set) == validation_pair_target:
            break
    if len(validation_set) != validation_pair_target:
        raise ValueError(
            "Unable to create the requested warm-item validation split. "
            "Lower validation_fraction or add cross-cancer item coverage."
        )

    validation_mask = np.fromiter(
        (pair in validation_set for pair in pair_keys),
        dtype=bool,
        count=len(pair_keys),
    )
    train_indices = np.flatnonzero(~validation_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Unable to create non-empty train and validation sets.")
    return train_indices, validation_indices


def make_loader(
    users: torch.Tensor,
    items: torch.Tensor,
    ratings: torch.Tensor,
    indices: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """Build a deterministic loader over selected interaction rows."""
    selected = torch.as_tensor(indices, dtype=torch.long)
    dataset = TensorDataset(
        users[selected],
        items[selected],
        ratings[selected],
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _copy_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[dict[str, torch.Tensor], float]:
    """Fit a model and return the state with the lowest validation MSE."""
    if epochs < 1 or patience < 1:
        raise ValueError("epochs and patience must be positive.")
    model = model.to(device)
    user_features = user_features.to(device)
    item_features = item_features.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.7,
        patience=max(1, patience // 2),
    )
    criterion = nn.MSELoss(reduction="sum")
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0

    for _epoch in range(epochs):
        model.train()
        for user_ids, item_ids, ratings in train_loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad()
            predictions = model(
                user_ids,
                item_ids,
                user_features[user_ids],
                item_features[item_ids],
            )
            loss = torch.mean((predictions - ratings) ** 2)
            loss.backward()
            optimizer.step()

        model.eval()
        squared_error = 0.0
        observations = 0
        with torch.no_grad():
            for user_ids, item_ids, ratings in validation_loader:
                user_ids = user_ids.to(device)
                item_ids = item_ids.to(device)
                ratings = ratings.to(device)
                predictions = model(
                    user_ids,
                    item_ids,
                    user_features[user_ids],
                    item_features[item_ids],
                )
                squared_error += criterion(predictions, ratings).item()
                observations += ratings.numel()
        validation_loss = squared_error / observations
        scheduler.step(validation_loss)

        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = _copy_state_dict(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a model state.")
    return best_state, best_loss


def transfer_pretrained_weights(
    model: HybridNCF,
    gmf_state: dict[str, torch.Tensor],
    mlp_state: dict[str, torch.Tensor],
    *,
    alpha: float = 0.5,
) -> HybridNCF:
    """Initialize the combined model from the two pretrained paths."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one.")
    state = copy.deepcopy(model.state_dict())
    for name in (
        "gmf_user_proj.weight",
        "gmf_user_proj.bias",
        "gmf_item_proj.weight",
        "gmf_item_proj.bias",
    ):
        state[name] = gmf_state[name]
    for name in state:
        if name.startswith("mlp_layers."):
            state[name] = mlp_state[name]
    state["user_embedding.weight"] = (
        gmf_state["user_embedding.weight"]
        + mlp_state["user_embedding.weight"]
    ) / 2.0
    state["item_embedding.weight"] = (
        gmf_state["item_embedding.weight"]
        + mlp_state["item_embedding.weight"]
    ) / 2.0
    state["output_layer.weight"] = alpha * torch.cat(
        (
            gmf_state["output_layer.weight"],
            mlp_state["output_layer.weight"],
        ),
        dim=1,
    )
    state["output_layer.bias"] = (
        alpha
        * (
            gmf_state["output_layer.bias"]
            + mlp_state["output_layer.bias"]
        )
    )
    model.load_state_dict(state, strict=True)
    return model


def fit_full_epochs(
    model: HybridNCF,
    loader: DataLoader,
    *,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
) -> HybridNCF:
    """Run the final fixed number of optimization epochs on all rows."""
    if epochs < 0:
        raise ValueError("final epochs cannot be negative.")
    model = model.to(device)
    user_features = user_features.to(device)
    item_features = item_features.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    for _epoch in range(epochs):
        model.train()
        for user_ids, item_ids, ratings in loader:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)
            ratings = ratings.to(device)
            optimizer.zero_grad()
            predictions = model(
                user_ids,
                item_ids,
                user_features[user_ids],
                item_features[item_ids],
            )
            loss = torch.mean((predictions - ratings) ** 2)
            loss.backward()
            optimizer.step()
    return model.cpu()


def train_hybrid_ncf(
    prepared: PreparedTrainingData,
    hyperparameters: dict[str, Any],
    *,
    device: torch.device,
    seed: int = 2000,
    validation_fraction: float = 0.2,
    pretrain_epochs: int = 30,
    pretrain_patience: int = 5,
    finetune_epochs: int = 20,
    finetune_patience: int = 3,
    finetune_lr_scale: float = 0.1,
    final_epochs: int = 10,
) -> tuple[HybridNCF, float]:
    """Pretrain GMF/MLP paths, fine-tune Hybrid NCF, and refit all rows."""
    seed_everything(seed)
    hp = hyperparameters
    validate_hyperparameters(hp)
    if (
        not math.isfinite(finetune_lr_scale)
        or finetune_lr_scale <= 0
    ):
        raise ValueError("finetune_lr_scale must be positive and finite.")
    if min(
        pretrain_epochs,
        pretrain_patience,
        finetune_epochs,
        finetune_patience,
    ) < 1:
        raise ValueError("Training epochs and patience values must be positive.")
    if final_epochs < 0:
        raise ValueError("final_epochs cannot be negative.")
    users = torch.tensor(
        prepared.interactions["user_id"].to_numpy(),
        dtype=torch.long,
    )
    items = torch.tensor(
        prepared.interactions["item_id"].to_numpy(),
        dtype=torch.long,
    )
    ratings = torch.tensor(
        prepared.interactions["Score"].to_numpy(),
        dtype=torch.float32,
    )
    train_indices, validation_indices = split_rows_by_pair(
        prepared.interactions,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    batch_size = int(hp["batch_size"])
    train_loader = make_loader(
        users,
        items,
        ratings,
        train_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    validation_loader = make_loader(
        users,
        items,
        ratings,
        validation_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )

    common = {
        "num_users": len(prepared.user_mapping),
        "num_items": len(prepared.item_mapping),
        "embedding_dim": int(hp["embedding_dim"]),
        "user_feature_dim": prepared.user_features.shape[1],
        "item_feature_dim": prepared.item_features.shape[1],
    }
    fit_options = {
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "user_features": prepared.user_features,
        "item_features": prepared.item_features,
        "device": device,
        "learning_rate": float(hp["learning_rate"]),
        "weight_decay": float(hp["weight_decay"]),
        "epochs": pretrain_epochs,
        "patience": pretrain_patience,
    }

    gmf = HybridGMF(**common, gmf_dim=int(hp["gmf_dim"]))
    gmf_state, _gmf_loss = train_with_early_stopping(gmf, **fit_options)
    mlp = HybridMLP(
        **common,
        hidden_units=[int(value) for value in hp["hidden_units"]],
        dropout_rate=float(hp["dropout_rate"]),
    )
    mlp_state, _mlp_loss = train_with_early_stopping(mlp, **fit_options)

    combined = HybridNCF(
        **common,
        gmf_dim=int(hp["gmf_dim"]),
        hidden_units=[int(value) for value in hp["hidden_units"]],
        dropout_rate=float(hp["dropout_rate"]),
    )
    transfer_pretrained_weights(combined, gmf_state, mlp_state)
    fine_tune_lr = float(hp["learning_rate"]) * finetune_lr_scale
    combined_state, validation_loss = train_with_early_stopping(
        combined,
        train_loader,
        validation_loader,
        user_features=prepared.user_features,
        item_features=prepared.item_features,
        device=device,
        learning_rate=fine_tune_lr,
        weight_decay=float(hp["weight_decay"]),
        epochs=finetune_epochs,
        patience=finetune_patience,
    )
    combined.load_state_dict(combined_state, strict=True)

    all_indices = np.arange(len(prepared.interactions))
    full_loader = make_loader(
        users,
        items,
        ratings,
        all_indices,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    combined = fit_full_epochs(
        combined,
        full_loader,
        user_features=prepared.user_features,
        item_features=prepared.item_features,
        device=device,
        learning_rate=fine_tune_lr,
        weight_decay=float(hp["weight_decay"]),
        epochs=final_epochs,
    )
    return combined, validation_loss


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train Hybrid NCF from local source CSVs and export an "
            "eight-file inference bundle."
        )
    )
    parser.add_argument(
        "--summary-data",
        type=Path,
        default=PRIVATE_DATA / "summary_w_score.csv",
    )
    parser.add_argument(
        "--targeted-therapy",
        type=Path,
        default=PRIVATE_DATA / "targeted_therapy.csv",
    )
    parser.add_argument(
        "--hyperparameters",
        type=Path,
        default=DEFAULT_HYPERPARAMETERS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-interactions-per-cancer", type=int, default=20)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
    )
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.1)
    parser.add_argument("--final-epochs", type=int, default=10)
    parser.add_argument(
        "--data-manifest",
        type=Path,
        default=DEFAULT_DATA_MANIFEST,
    )
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help=(
            "Train from inputs whose hashes or row counts differ from the "
            "recorded private research snapshots."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    destination = require_new_bundle_destination(args.output_dir)
    hyperparameters = load_hyperparameters(args.hyperparameters)
    summary, targeted = load_source_frames(
        args.summary_data,
        args.targeted_therapy,
    )
    source_provenance = verify_source_snapshots(
        summary_path=args.summary_data,
        targeted_path=args.targeted_therapy,
        summary=summary,
        targeted=targeted,
        manifest_path=args.data_manifest,
        allow_unverified=args.allow_unverified_inputs,
    )
    prepared = prepare_training_data(
        summary,
        targeted,
        min_interactions_per_cancer=args.min_interactions_per_cancer,
    )
    device = resolve_device(args.device)
    print(
        "Prepared training data: "
        f"{len(prepared.interactions):,} rows, "
        f"{len(prepared.user_mapping):,} cancers, "
        f"{len(prepared.item_mapping):,} interventions, "
        f"{len(prepared.feature_names):,} mutation features."
    )
    model, validation_loss = train_hybrid_ncf(
        prepared,
        hyperparameters,
        device=device,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        pretrain_epochs=args.pretrain_epochs,
        pretrain_patience=args.pretrain_patience,
        finetune_epochs=args.finetune_epochs,
        finetune_patience=args.finetune_patience,
        finetune_lr_scale=args.finetune_lr_scale,
        final_epochs=args.final_epochs,
    )
    bundle_hyperparameters = dict(hyperparameters)
    bundle_hyperparameters["training_provenance"] = {
        **source_provenance,
        "code_sha256": training_code_hashes(),
        "seed": args.seed,
        "min_interactions_per_cancer": args.min_interactions_per_cancer,
        "validation": {
            "strategy": "pair-grouped warm-catalog holdout",
            "fraction": args.validation_fraction,
            "diagnostic_mse_before_full_refit": validation_loss,
        },
        "epochs": {
            "pretrain": args.pretrain_epochs,
            "finetune": args.finetune_epochs,
            "final_full_data": args.final_epochs,
        },
        "feature_names": prepared.feature_names,
    }
    destination = export_inference_bundle(
        destination,
        model=model,
        hyperparameters=bundle_hyperparameters,
        user_mapping=prepared.user_mapping,
        item_mapping=prepared.item_mapping,
        user_features=prepared.user_features,
        item_features=prepared.item_features,
        known_items=prepared.known_items,
        metadata_lookup=prepared.metadata_lookup,
    )
    print(
        "Training complete. Pair-grouped validation MSE "
        f"(training diagnostic only): {validation_loss:.6f}"
    )
    print(f"Validated inference bundle: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
