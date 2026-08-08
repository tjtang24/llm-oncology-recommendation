#!/usr/bin/env python3
"""Refit the manuscript's pretrained ID-only NCF with frozen settings.

The model has GMF and MLP paths but no mutation side features. GMF and MLP are
pretrained separately, transferred into NCF, fine-tuned, and then trained for a
fixed ten epochs on all rows. The RNG call order intentionally matches the
audited historical source.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

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
DEFAULT_CONFIG = HERE / "configs" / "ncf.json"
DEFAULT_DATA = PRIVATE_DATA / "summary_w_score.csv"
DEFAULT_OUTPUT = APP_ROOT / "params" / "ncf-retrained"


class GMF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        gmf_dim: int,
    ) -> None:
        super().__init__()
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        self.gmf_user_proj = nn.Linear(embedding_dim, gmf_dim)
        self.gmf_item_proj = nn.Linear(embedding_dim, gmf_dim)
        self.output_layer = nn.Linear(gmf_dim, 1)

    def forward(
        self, user_input: torch.Tensor, item_input: torch.Tensor
    ) -> torch.Tensor:
        user = self.gmf_user_proj(self.gmf_user_embedding(user_input))
        item = self.gmf_item_proj(self.gmf_item_embedding(item_input))
        return self.output_layer(user * item).view(-1)


class MLP(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_units: Sequence[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        input_dim = embedding_dim * 2
        layers: List[nn.Module] = []
        for units in hidden_units:
            layers.extend(
                [
                    nn.Linear(input_dim, units),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_dim = units
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_units[-1], 1)

    def forward(
        self, user_input: torch.Tensor, item_input: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat(
            [
                self.mlp_user_embedding(user_input),
                self.mlp_item_embedding(item_input),
            ],
            dim=-1,
        )
        return self.output_layer(self.mlp_layers(inputs)).view(-1)


class PretrainedNCF(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        gmf_embedding_dim: int,
        mlp_embedding_dim: int,
        gmf_dim: int,
        hidden_units: Sequence[int],
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.gmf_user_embedding = nn.Embedding(num_users, gmf_embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, gmf_embedding_dim)
        self.gmf_user_proj = nn.Linear(gmf_embedding_dim, gmf_dim)
        self.gmf_item_proj = nn.Linear(gmf_embedding_dim, gmf_dim)
        self.mlp_user_embedding = nn.Embedding(num_users, mlp_embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, mlp_embedding_dim)
        input_dim = mlp_embedding_dim * 2
        layers: List[nn.Module] = []
        for units in hidden_units:
            layers.extend(
                [
                    nn.Linear(input_dim, units),
                    nn.GELU(),
                    nn.Dropout(dropout_rate),
                ]
            )
            input_dim = units
        self.mlp_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(gmf_dim + hidden_units[-1], 1)

    def forward(
        self, user_input: torch.Tensor, item_input: torch.Tensor
    ) -> torch.Tensor:
        gmf_user = self.gmf_user_proj(self.gmf_user_embedding(user_input))
        gmf_item = self.gmf_item_proj(self.gmf_item_embedding(item_input))
        gmf = gmf_user * gmf_item
        mlp_inputs = torch.cat(
            [
                self.mlp_user_embedding(user_input),
                self.mlp_item_embedding(item_input),
            ],
            dim=-1,
        )
        mlp = self.mlp_layers(mlp_inputs)
        return self.output_layer(torch.cat([gmf, mlp], dim=-1)).view(-1)


def build_ncf(
    num_users: int, num_items: int, hyperparameters: Mapping[str, Any]
) -> PretrainedNCF:
    return PretrainedNCF(
        num_users,
        num_items,
        int(hyperparameters["gmf_embedding_dim"]),
        int(hyperparameters["mlp_embedding_dim"]),
        int(hyperparameters["gmf_dim"]),
        [int(value) for value in hyperparameters["hidden_units"]],
        float(hyperparameters["dropout_rate"]),
    )


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    learning_rate: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    phase: str,
    history: List[Dict[str, Any]],
    scheduler_factor: float,
) -> Tuple[Dict[str, torch.Tensor], float, int, int]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=patience // 2,
    )
    best_loss = float("inf")
    best_weights: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    no_improve = 0
    epochs_run = 0
    for epoch in range(epochs):
        model.train()
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()

        model.eval()
        batch_loss_sum = 0.0
        squared_error_sum = 0.0
        observations = 0
        batches = 0
        with torch.no_grad():
            for user, item, rating in validation_loader:
                output = model(user, item)
                batch_loss_sum += float(criterion(output, rating))
                squared_error_sum += float(torch.sum((output - rating) ** 2))
                observations += len(rating)
                batches += 1
        validation_loss = batch_loss_sum / batches
        scheduler.step(validation_loss)
        improved = validation_loss < best_loss
        if improved:
            best_loss = validation_loss
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            no_improve = 0
        else:
            no_improve += 1
        epochs_run = epoch + 1
        row = {
            "phase": phase,
            "epoch": epochs_run,
            "val_mse_legacy_unweighted_batches": validation_loss,
            "val_mse_sample_weighted": squared_error_sum / observations,
            "improved": improved,
            "no_improve": no_improve,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if no_improve == patience:
            break
    if best_weights is None:
        raise RuntimeError(f"{phase} did not produce a checkpoint")
    return best_weights, best_loss, best_epoch, epochs_run


def load_pretrained_paths(
    ncf: PretrainedNCF,
    gmf_state: Mapping[str, torch.Tensor],
    mlp_state: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> PretrainedNCF:
    state = ncf.state_dict()
    for key in [
        "gmf_user_embedding.weight",
        "gmf_item_embedding.weight",
        "gmf_user_proj.weight",
        "gmf_user_proj.bias",
        "gmf_item_proj.weight",
        "gmf_item_proj.bias",
    ]:
        state[key] = gmf_state[key]
    for key in state:
        if key.startswith("mlp_"):
            state[key] = mlp_state[key]
    state["output_layer.weight"] = alpha * torch.cat(
        [gmf_state["output_layer.weight"], mlp_state["output_layer.weight"]],
        dim=1,
    )
    state["output_layer.bias"] = alpha * (
        gmf_state["output_layer.bias"] + mlp_state["output_layer.bias"]
    )
    ncf.load_state_dict(state)
    return ncf


def predict_rows(
    model: nn.Module,
    users: Sequence[int],
    items: Sequence[int],
    *,
    batch_size: int = 2048,
) -> np.ndarray:
    dataset = TensorDataset(
        torch.as_tensor(users, dtype=torch.long),
        torch.as_tensor(items, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    predictions: List[np.ndarray] = []
    model.cpu().eval()
    with torch.no_grad():
        for user, item in loader:
            predictions.append(model(user, item).numpy())
    return np.concatenate(predictions)


def full_catalog(
    model: nn.Module,
    user_mapping: Mapping[str, int],
    item_mapping: Mapping[str, int],
) -> np.ndarray:
    item_ids = np.asarray(list(item_mapping.values()), dtype=np.int64)
    matrix = np.empty(
        (len(user_mapping), len(item_mapping)), dtype=np.float32
    )
    for _cancer, user_id in user_mapping.items():
        matrix[user_id] = predict_rows(
            model,
            np.full(len(item_mapping), user_id, dtype=np.int64),
            item_ids,
        )
    return matrix


def training_snapshot_recommendations(
    model: nn.Module,
    data: pd.DataFrame,
    user_mapping: Mapping[str, int],
    item_mapping: Mapping[str, int],
) -> pd.DataFrame:
    item_names = list(item_mapping.keys())
    matrix = full_catalog(model, user_mapping, item_mapping)
    rows: List[Dict[str, Any]] = []
    for cancer, user_id in user_mapping.items():
        known = set(
            data.loc[data["user_id"] == user_id, "Intervention"].unique()
        )
        frame = pd.DataFrame(
            {
                "item_id": np.arange(len(item_mapping), dtype=int),
                "Intervention": item_names,
                "Predicted_Score_raw": matrix[user_id],
            }
        )
        novel = frame[~frame["Intervention"].isin(known)]
        top = novel.sort_values("Predicted_Score_raw", ascending=False).head(20)
        for rank, row in enumerate(top.itertuples(index=False), 1):
            rows.append(
                {
                    "Cancer": cancer,
                    "Rank": rank,
                    "item_id": row.item_id,
                    "Intervention": row.Intervention,
                    "Predicted_Score_raw": row.Predicted_Score_raw,
                    "Predicted_Score_rounded4": round(
                        float(row.Predicted_Score_raw), 4
                    ),
                }
            )
    return pd.DataFrame(rows)


def canonical_state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit the fixed manuscript ID-only NCF configuration."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help=(
            "Run on different inputs while recording that reference checks "
            "do not apply."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    require_new_output_path(args.output)
    config = read_json(DEFAULT_CONFIG)
    hp = config["hyperparameters"]
    schedule = config["schedule"]
    seed = int(config["seed"])

    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)
    torch.manual_seed(seed)
    np.random.seed(seed)

    raw = pd.read_csv(args.data)
    input_record = verify_csv(
        args.data,
        raw,
        "training_snapshot",
        allow_unverified=args.allow_unverified_inputs,
    )
    data, user_mapping, item_mapping, counts = prepare_legacy_interactions(raw)
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
    }
    if not all(count_checks.values()) and not args.allow_unverified_inputs:
        raise ValueError(f"Processed data contract mismatch: {count_checks}")

    # Preserve the audited RNG sequence: construct NCF, reset, split, then train.
    ncf = build_ncf(len(user_mapping), len(item_mapping), hp)
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = TensorDataset(
        torch.tensor(data["user_id"].to_numpy(), dtype=torch.long),
        torch.tensor(data["item_id"].to_numpy(), dtype=torch.long),
        torch.tensor(data["Score"].to_numpy(), dtype=torch.float),
    )
    validation_size = int(len(dataset) * float(schedule["validation_fraction"]))
    training_size = len(dataset) - validation_size
    training_set, validation_set = random_split(
        dataset, [training_size, validation_size]
    )
    training_loader = DataLoader(
        training_set, batch_size=int(hp["batch_size"]), shuffle=True
    )
    validation_loader = DataLoader(
        validation_set, batch_size=int(hp["batch_size"])
    )
    history: List[Dict[str, Any]] = []

    gmf = GMF(
        len(user_mapping),
        len(item_mapping),
        int(hp["gmf_embedding_dim"]),
        int(hp["gmf_dim"]),
    )
    gmf_state, _gmf_loss, gmf_best, gmf_run = train_with_early_stopping(
        gmf,
        training_loader,
        validation_loader,
        learning_rate=float(hp["learning_rate"]),
        epochs=int(schedule["gmf_pretrain_max_epochs"]),
        patience=int(schedule["pretrain_patience"]),
        weight_decay=float(schedule["pretrain_weight_decay"]),
        phase="gmf_pretrain",
        history=history,
        scheduler_factor=float(schedule["scheduler_factor"]),
    )
    mlp = MLP(
        len(user_mapping),
        len(item_mapping),
        int(hp["mlp_embedding_dim"]),
        [int(value) for value in hp["hidden_units"]],
        float(hp["dropout_rate"]),
    )
    mlp_state, _mlp_loss, mlp_best, mlp_run = train_with_early_stopping(
        mlp,
        training_loader,
        validation_loader,
        learning_rate=float(hp["learning_rate"]),
        epochs=int(schedule["mlp_pretrain_max_epochs"]),
        patience=int(schedule["pretrain_patience"]),
        weight_decay=float(schedule["pretrain_weight_decay"]),
        phase="mlp_pretrain",
        history=history,
        scheduler_factor=float(schedule["scheduler_factor"]),
    )
    ncf = load_pretrained_paths(
        ncf,
        gmf_state,
        mlp_state,
        alpha=float(schedule["pretrained_output_alpha"]),
    )
    ncf_state, _ncf_loss, ncf_best, ncf_run = train_with_early_stopping(
        ncf,
        training_loader,
        validation_loader,
        learning_rate=float(hp["learning_rate"])
        * float(schedule["finetune_lr_scale"]),
        epochs=int(schedule["ncf_finetune_max_epochs"]),
        patience=int(schedule["finetune_patience"]),
        weight_decay=float(schedule["finetune_weight_decay"]),
        phase="ncf_finetune",
        history=history,
        scheduler_factor=float(schedule["scheduler_factor"]),
    )
    ncf.load_state_dict(ncf_state)

    full_loader = DataLoader(
        dataset, batch_size=int(hp["batch_size"]), shuffle=True
    )
    optimizer = optim.Adam(
        ncf.parameters(),
        lr=float(hp["learning_rate"])
        * float(schedule["finetune_lr_scale"]),
        weight_decay=float(schedule["full_data_weight_decay"]),
    )
    criterion = nn.MSELoss()
    ncf.train()
    for epoch in range(int(schedule["full_data_epochs"])):
        for user, item, rating in full_loader:
            optimizer.zero_grad()
            output = ncf(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
        row = {"phase": "full_refit", "epoch": epoch + 1}
        history.append(row)
        print(json.dumps(row), flush=True)
    ncf.eval()

    output = ensure_new_output(args.output)
    model_path = output / "final_model.pth"
    torch.save(ncf.state_dict(), model_path)
    history_path = output / "epoch_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    user_mapping_path = output / "user_mapping.json"
    item_mapping_path = output / "item_mapping.json"
    write_json(user_mapping_path, user_mapping)
    write_json(item_mapping_path, item_mapping)

    catalog = full_catalog(ncf, user_mapping, item_mapping)
    np.save(output / "full_catalog_predictions.npy", catalog)
    training_top20 = training_snapshot_recommendations(
        ncf, data, user_mapping, item_mapping
    )
    training_top20_path = output / "top20_all_cancers.csv"
    training_top20.to_csv(training_top20_path, index=False)

    environment = collect_environment(
        [
            "numpy",
            "pandas",
            "scikit-learn",
            "scipy",
            "joblib",
            "threadpoolctl",
            "torch",
            "scikit-surprise",
        ],
        torch_details={
            "device": "cpu",
            "cuda_available": torch.cuda.is_available(),
            "mps_available_but_unused": bool(
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
        },
    )
    hashes = {
        "model_file_sha256": sha256_file(model_path),
        "canonical_tensor_sha256": canonical_state_hash(ncf.state_dict()),
        "catalog_raw_sha256": sha256_array(catalog),
        "training_top20_csv_sha256": sha256_file(training_top20_path),
        "user_mapping_json_sha256": sha256_file(user_mapping_path),
        "item_mapping_json_sha256": sha256_file(item_mapping_path),
        "epoch_history_csv_sha256": sha256_file(history_path),
    }

    epoch_trace = {
        "gmf_run": gmf_run,
        "gmf_best": gmf_best,
        "mlp_run": mlp_run,
        "mlp_best": mlp_best,
        "ncf_run": ncf_run,
        "ncf_best": ncf_best,
        "full_data": int(schedule["full_data_epochs"]),
    }
    output_comparison = reference_checks(config["reference_outputs"], hashes)
    reference_contract_checks = {
        "committed_config": config == read_json(DEFAULT_CONFIG),
        "training_input": all(input_record["checks"].values()),
        "processed_data": all(count_checks.values()),
        "epoch_trace": epoch_trace == config["reference_epoch_trace"],
        "scientific_outputs": output_comparison["all_expected_outputs_exact"],
    }
    manifest = {
        "model": "ncf",
        "mode": config["training_mode"],
        "seed": seed,
        "device": "cpu",
        "config": config,
        "config_sha256": sha256_file(DEFAULT_CONFIG),
        "training_input": input_record,
        "processed_data": counts,
        "processed_data_checks": count_checks,
        "rng_sequence": [
            "seed",
            "construct empty NCF",
            "reset seed",
            "random_split",
            "GMF pretrain",
            "MLP pretrain",
            "NCF fine-tune",
            "full-data refit",
        ],
        "epoch_trace": epoch_trace,
        "epoch_trace_matches_reference": epoch_trace
        == config["reference_epoch_trace"],
        "environment": environment,
        "environment_comparison": compare_environment(
            environment,
            required_packages=[
                "numpy",
                "pandas",
                "scikit-learn",
                "scipy",
                "joblib",
                "threadpoolctl",
                "torch",
            ],
        ),
        "outputs": {
            "directory": str(output),
            "catalog_shape": list(catalog.shape),
            "training_top20_rows": len(training_top20),
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
