#!/usr/bin/env python3
"""Resumable historical CV replay for the ID-only pretrained NCF baseline.

This is the released, source-faithful port of the private audit runner. It
regenerates the manuscript's cross-validated RMSE for the ID-only pretrained
NCF from the exact private training snapshot and the frozen configuration. The
model classes, ``build_ncf``, and ``load_pretrained`` transfer are imported
from :mod:`train_ncf_fixed`; the order-sensitive preprocessing is imported from
:mod:`common`. The tuning orchestration, hyperparameter sampler, per-stage
training loop, and fold-model evaluation are copied verbatim from the audited
historical source so the torch/NumPy RNG consumption order is preserved.

Two protocols are intentionally separate:

* ``faithful`` reproduces the historical code, including a single tuner whose
  best score and model persist across folds, and row-level KFold. This is the
  configuration that regenerates the reported manuscript RMSE.
* ``corrected`` resets the tuner per fold and assigns canonical
  cancer-treatment pairs wholly to one fold.

The corrected mode removes the two documented leakage mechanisms but still
uses each fold's validation set for hyperparameter selection. It is therefore
NOT a nested-CV estimate and is labelled accordingly. In the faithful protocol
the global tuner carries state across folds, so the reproduced numbers are
source-faithful historical statistics, not clean nested validation.

The runner requires only the private training snapshot; it needs no model
weights, mappings, or recommendation CSVs.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

from .common import (
    prepare_legacy_interactions,
    read_json,
    sha256_file,
    verify_csv,
)
from .train_ncf_fixed import GMF, MLP, build_ncf, load_pretrained_paths


HERE = Path(__file__).resolve().parent
APP_ROOT = HERE.parent
PRIVATE_DATA = APP_ROOT.parent / "private_data"
DEFAULT_DATA = PRIVATE_DATA / "summary_w_score.csv"
RUNS_DIR = APP_ROOT / "params" / "cv-ncf-runs"
DATA_MANIFEST = HERE / "data_manifest.json"
CONFIG_PATH = HERE / "configs" / "cv_ncf.json"
SEED = 2000
PAPER_RMSE = 0.4947
RESUME_STATE_VERSION = 4
PRETRAINED_ALPHA = 0.5

# Local implementation files whose bytes are fingerprinted so a resumed or
# certified run cannot silently mix in changed numeric code.
RUNTIME_CODE_FILES = [
    HERE / "cv_ncf.py",
    HERE / "train_ncf_fixed.py",
    HERE / "common.py",
]


def load_pretrained(
    ncf,
    gmf_state: Mapping[str, torch.Tensor],
    mlp_state: Mapping[str, torch.Tensor],
):
    """Historical transfer with the audited default alpha of 0.5."""
    return load_pretrained_paths(
        ncf, gmf_state, mlp_state, alpha=PRETRAINED_ALPHA
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def input_hashes_for(data_path: Path) -> Dict[str, str]:
    return {str(data_path): sha256_file(data_path)}


def runtime_code_hashes() -> Dict[str, str]:
    """Fingerprint every local implementation used by the resumable run."""
    return {
        str(path.resolve()): sha256_file(path) for path in RUNTIME_CODE_FILES
    }


def dump_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def save_resume(path: Path, state: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def environment() -> Dict[str, object]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "cuda": torch.cuda.is_available(),
        "mps": bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
    }


def canonical_intervention(value: object) -> str:
    return ", ".join(sorted(part.strip() for part in str(value).split(",")))


def create_row_splits(n_rows: int, folds: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    return list(
        KFold(
            n_splits=folds, shuffle=True, random_state=SEED
        ).split(np.arange(n_rows))
    )


def create_pair_grouped_splits(
    data: pd.DataFrame, folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Deterministic shuffled-greedy allocation of canonical pair groups."""
    canonical = data["Intervention"].map(canonical_intervention)
    group_keys = list(zip(data["Cancer"].astype(str), canonical))
    grouped: Dict[Tuple[str, str], List[int]] = {}
    for index, key in enumerate(group_keys):
        grouped.setdefault(key, []).append(index)
    rng = np.random.RandomState(SEED)
    keys = list(grouped)
    random_tiebreak = {key: int(value) for key, value in zip(keys, rng.permutation(len(keys)))}
    keys.sort(key=lambda key: (-len(grouped[key]), random_tiebreak[key]))
    buckets: List[List[int]] = [[] for _ in range(folds)]
    bucket_sizes = [0] * folds
    for key in keys:
        target = min(range(folds), key=lambda fold: (bucket_sizes[fold], fold))
        buckets[target].extend(grouped[key])
        bucket_sizes[target] += len(grouped[key])
    all_indices = np.arange(len(data), dtype=np.int64)
    splits: List[Tuple[np.ndarray, np.ndarray]] = []
    for bucket in buckets:
        validation = np.asarray(sorted(bucket), dtype=np.int64)
        mask = np.ones(len(data), dtype=bool)
        mask[validation] = False
        training = all_indices[mask]
        splits.append((training, validation))
    return splits


def split_diagnostics(
    data: pd.DataFrame,
    splits: Sequence[Tuple[np.ndarray, np.ndarray]],
) -> List[Dict[str, object]]:
    canonical = data["Intervention"].map(canonical_intervention)
    result: List[Dict[str, object]] = []
    for fold, (train_index, validation_index) in enumerate(splits, 1):
        raw_train = set(
            zip(
                data.iloc[train_index]["Cancer"],
                data.iloc[train_index]["Intervention"],
            )
        )
        canonical_train = set(
            zip(
                data.iloc[train_index]["Cancer"],
                canonical.iloc[train_index],
            )
        )
        raw_validation = list(
            zip(
                data.iloc[validation_index]["Cancer"],
                data.iloc[validation_index]["Intervention"],
            )
        )
        canonical_validation = list(
            zip(
                data.iloc[validation_index]["Cancer"],
                canonical.iloc[validation_index],
            )
        )
        result.append(
            {
                "fold": fold,
                "train_rows": len(train_index),
                "validation_rows": len(validation_index),
                "exact_pair_seen_fraction": sum(
                    pair in raw_train for pair in raw_validation
                )
                / len(validation_index),
                "canonical_pair_seen_fraction": sum(
                    pair in canonical_train
                    for pair in canonical_validation
                )
                / len(validation_index),
                "train_indices_sha256": sha256_bytes(
                    np.asarray(train_index, dtype=np.int64).tobytes()
                ),
                "validation_indices_sha256": sha256_bytes(
                    np.asarray(validation_index, dtype=np.int64).tobytes()
                ),
            }
        )
    return result


def sample_hyperparameters() -> Dict[str, object]:
    gmf_embedding_dim = int(np.random.choice([32, 64, 128]))
    mlp_embedding_dim = int(np.random.choice([32, 64, 128]))
    gmf_dim = gmf_embedding_dim // 2
    number_hidden = int(np.random.randint(1, 4))
    hidden_units = sorted(
        [
            int(np.random.choice([64, 128, 256]))
            for _ in range(number_hidden)
        ],
        reverse=True,
    )
    return {
        "gmf_embedding_dim": gmf_embedding_dim,
        "mlp_embedding_dim": mlp_embedding_dim,
        "gmf_dim": gmf_dim,
        "num_hidden_layers": number_hidden,
        "hidden_units": hidden_units,
        "dropout_rate": float(np.random.uniform(0.1, 0.5)),
        "learning_rate": float(10 ** np.random.uniform(-4, -2)),
        "batch_size": int(np.random.choice([512, 1024, 2048])),
    }


def train_stage(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    learning_rate: float,
    epochs: int,
    patience: int,
) -> Tuple[Dict[str, torch.Tensor], float, int, int, List[Dict[str, object]]]:
    """Literal recovered train_model_with_early_stopping behavior."""
    model.cpu()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.7, patience=patience // 2
    )
    best_loss = float("inf")
    best_weights = None
    best_epoch = 0
    no_improve = 0
    history: List[Dict[str, object]] = []
    for epoch in range(epochs):
        epoch_started = time.time()
        model.train()
        for user, item, rating in train_loader:
            optimizer.zero_grad()
            output = model(user, item)
            loss = criterion(output, rating)
            loss.backward()
            optimizer.step()
        model.eval()
        batch_loss_sum = 0.0
        sample_sse = 0.0
        samples = 0
        batches = 0
        with torch.no_grad():
            for user, item, rating in validation_loader:
                output = model(user, item)
                batch_loss_sum += float(criterion(output, rating))
                sample_sse += float(torch.sum((output - rating) ** 2))
                samples += len(rating)
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
        history.append(
            {
                "epoch": epoch + 1,
                "legacy_unweighted_batch_mse": validation_loss,
                "sample_weighted_mse": sample_sse / samples,
                "improved": improved,
                "no_improve": no_improve,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "seconds": time.time() - epoch_started,
            }
        )
        if no_improve == patience:
            break
    if best_weights is None:
        raise RuntimeError("Training stage failed to produce weights")
    return best_weights, best_loss, best_epoch, epoch + 1, history


def run_trial(
    hp: Mapping[str, object],
    number_users: int,
    number_items: int,
    train_inputs: Sequence[np.ndarray],
    train_labels: np.ndarray,
    validation_inputs: Sequence[np.ndarray],
    validation_labels: np.ndarray,
    pretrain_epochs: int,
    pretrain_patience: int,
    finetune_epochs: int,
    finetune_patience: int,
) -> Tuple[Dict[str, torch.Tensor], float, Dict[str, object]]:
    batch_size = int(hp["batch_size"])
    train_dataset = TensorDataset(
        torch.tensor(train_inputs[0], dtype=torch.long),
        torch.tensor(train_inputs[1], dtype=torch.long),
        torch.tensor(train_labels, dtype=torch.float),
    )
    validation_dataset = TensorDataset(
        torch.tensor(validation_inputs[0], dtype=torch.long),
        torch.tensor(validation_inputs[1], dtype=torch.long),
        torch.tensor(validation_labels, dtype=torch.float),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size
    )
    gmf = GMF(
        number_users,
        number_items,
        int(hp["gmf_embedding_dim"]),
        int(hp["gmf_dim"]),
    )
    gmf_state, gmf_loss, gmf_best, gmf_run, gmf_history = train_stage(
        gmf,
        train_loader,
        validation_loader,
        float(hp["learning_rate"]),
        pretrain_epochs,
        pretrain_patience,
    )
    del gmf
    mlp = MLP(
        number_users,
        number_items,
        int(hp["mlp_embedding_dim"]),
        [int(value) for value in hp["hidden_units"]],
        float(hp["dropout_rate"]),
    )
    mlp_state, mlp_loss, mlp_best, mlp_run, mlp_history = train_stage(
        mlp,
        train_loader,
        validation_loader,
        float(hp["learning_rate"]),
        pretrain_epochs,
        pretrain_patience,
    )
    del mlp
    ncf = build_ncf(number_users, number_items, hp)
    ncf = load_pretrained(ncf, gmf_state, mlp_state)
    ncf_state, ncf_loss, ncf_best, ncf_run, ncf_history = train_stage(
        ncf,
        train_loader,
        validation_loader,
        float(hp["learning_rate"]) * 0.1,
        finetune_epochs,
        finetune_patience,
    )
    record = {
        "hyperparameters": dict(hp),
        "gmf": {
            "best_legacy_mse": gmf_loss,
            "best_epoch": gmf_best,
            "epochs_run": gmf_run,
            "history": gmf_history,
        },
        "mlp": {
            "best_legacy_mse": mlp_loss,
            "best_epoch": mlp_best,
            "epochs_run": mlp_run,
            "history": mlp_history,
        },
        "ncf": {
            "best_legacy_mse": ncf_loss,
            "best_epoch": ncf_best,
            "epochs_run": ncf_run,
            "history": ncf_history,
        },
    }
    del ncf, train_loader, validation_loader, train_dataset, validation_dataset
    gc.collect()
    return ncf_state, ncf_loss, record


def evaluate_fold_model(
    weights: Mapping[str, torch.Tensor],
    hp: Mapping[str, object],
    number_users: int,
    number_items: int,
    validation_inputs: Sequence[np.ndarray],
    validation_labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Building this model before loading weights intentionally consumes torch
    # RNG, matching RandomSearch.search at the end of each historical fold.
    model = build_ncf(number_users, number_items, hp)
    model.load_state_dict(weights)
    model.cpu().eval()
    validation_dataset = TensorDataset(
        torch.tensor(validation_inputs[0], dtype=torch.long),
        torch.tensor(validation_inputs[1], dtype=torch.long),
    )
    validation_loader = DataLoader(validation_dataset, batch_size=1024)
    predictions: List[float] = []
    with torch.no_grad():
        for user, item in validation_loader:
            predictions.extend(model(user, item).tolist())
    values = np.asarray(predictions, dtype=np.float64)
    mse = float(mean_squared_error(validation_labels, values))
    result = {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": float(mean_absolute_error(validation_labels, values)),
    }
    del model, validation_loader, validation_dataset
    gc.collect()
    return values, result


def initial_resume_state(
    protocol: str, run_identity: Mapping[str, object]
) -> Dict[str, object]:
    return {
        "version": RESUME_STATE_VERSION,
        "protocol": protocol,
        "run_identity": copy.deepcopy(dict(run_identity)),
        "next_fold": 0,
        "next_trial": 0,
        "tuner_best_score": float("inf"),
        "tuner_best_weights": None,
        "tuner_best_hp": None,
        "tuner_source_fold": None,
        "tuner_source_trial": None,
        "fold_records": [],
        "trial_records": [],
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "python_rng": random.getstate(),
        "elapsed_seconds": 0.0,
        "max_rss_raw": 0,
        "complete": False,
    }


def restore_rng(state: Mapping[str, object]) -> None:
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    random.setstate(state["python_rng"])


def capture_rng(state: Dict[str, object]) -> None:
    state["numpy_rng"] = np.random.get_state()
    state["torch_rng"] = torch.get_rng_state()
    state["python_rng"] = random.getstate()


def checkpoint_resume(
    path: Path,
    state: Dict[str, object],
    invocation_started: float,
    elapsed_before_invocation: float,
) -> None:
    capture_rng(state)
    state["elapsed_seconds"] = (
        elapsed_before_invocation + time.time() - invocation_started
    )
    state["max_rss_raw"] = max(
        int(state.get("max_rss_raw", 0)),
        int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    )
    save_resume(path, state)


def write_progress(run_dir: Path, state: Mapping[str, object]) -> None:
    trial_rows = []
    for record in state["trial_records"]:
        trial_rows.append(
            {
                "fold": record["fold"],
                "trial": record["trial"],
                "ncf_best_legacy_mse": record["ncf_best_legacy_mse"],
                "updated_tuner": record["updated_tuner"],
                "tuner_best_score_after": record["tuner_best_score_after"],
                "tuner_source_fold_after": record["tuner_source_fold_after"],
                "tuner_source_trial_after": record["tuner_source_trial_after"],
                "seconds": record["seconds"],
                "hyperparameters_json": json.dumps(
                    record["hyperparameters"], sort_keys=True
                ),
            }
        )
    pd.DataFrame(trial_rows).to_csv(
        run_dir / "trial_metrics.csv", index=False
    )
    pd.DataFrame(state["fold_records"]).to_csv(
        run_dir / "fold_metrics.csv", index=False
    )


def final_summary(
    run_dir: Path,
    state: Mapping[str, object],
    protocol: str,
    config: Mapping[str, object],
    diagnostics: Sequence[Mapping[str, object]],
    input_hashes: Mapping[str, str],
    data_path: Path,
) -> Dict[str, object]:
    ending_code_hashes = runtime_code_hashes()
    expected_code_hashes = state["run_identity"]["runtime_code_hashes"]
    if ending_code_hashes != expected_code_hashes:
        raise RuntimeError(
            "Runner implementation changed during execution; refusing to "
            "certify a mixed-code result"
        )
    ending_environment = environment()
    expected_environment = state["run_identity"]["environment"]
    if ending_environment != expected_environment:
        raise RuntimeError(
            "Execution environment changed during execution; refusing to "
            "certify a mixed-environment result"
        )
    fold_records = list(state["fold_records"])
    expected_folds = int(config["folds"])
    observed_fold_ids = [int(record["fold"]) for record in fold_records]
    if observed_fold_ids != list(range(1, expected_folds + 1)):
        raise RuntimeError(
            "Fold records are missing, duplicated, or out of order; "
            "refusing to certify the result"
        )
    rmse_values = [float(record["rmse"]) for record in fold_records]
    mse_values = [float(record["mse"]) for record in fold_records]
    mae_values = [float(record["mae"]) for record in fold_records]
    total_sse = sum(
        float(record["mse"]) * int(record["validation_rows"])
        for record in fold_records
    )
    total_n = sum(int(record["validation_rows"]) for record in fold_records)
    output = {
        "status": "complete",
        "protocol": protocol,
        "protocol_interpretation": {
            "faithful": "source-faithful leaky historical behavior",
            "corrected": (
                "fresh tuner and canonical pair-grouped folds; not nested CV "
                "because outer validation selects hyperparameters"
            ),
        }[protocol],
        "configuration": dict(config),
        "paper_rmse": PAPER_RMSE,
        "folds_completed": len(fold_records),
        "per_fold": fold_records,
        "aggregate": {
            "arithmetic_mean_rmse": float(np.mean(rmse_values)),
            "minimum_rmse": float(np.min(rmse_values)),
            "pooled_rmse": math.sqrt(total_sse / total_n),
            "arithmetic_mean_mse": float(np.mean(mse_values)),
            "minimum_mse": float(np.min(mse_values)),
            "arithmetic_mean_mae": float(np.mean(mae_values)),
            "minimum_mae": float(np.min(mae_values)),
            "absolute_mean_rmse_difference_from_paper": abs(
                float(np.mean(rmse_values)) - PAPER_RMSE
            ),
        },
        "split_diagnostics": list(diagnostics),
        "executions_per_trial_declared_by_source": 2,
        "executions_per_trial_actually_run": 1,
        "input_hashes_start": dict(input_hashes),
        "input_hashes_end": input_hashes_for(data_path),
        "runtime_code_hashes_end": ending_code_hashes,
        "runtime_code_unchanged": True,
        "runtime_seconds": float(state["elapsed_seconds"]),
        "max_rss_raw": int(state["max_rss_raw"]),
        "environment": ending_environment,
        "environment_unchanged": True,
    }
    dump_json(run_dir / "summary.json", output)
    return output


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(12)
    torch.set_num_interop_threads(12)

    data_path = args.data.resolve()
    raw = pd.read_csv(data_path)
    input_record = verify_csv(
        data_path,
        raw,
        "training_snapshot",
        allow_unverified=args.allow_unverified_inputs,
    )
    input_hashes = input_hashes_for(data_path)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = RUNS_DIR / args.run_id
    resume_path = run_dir / "resume_state.pth"
    if run_dir.exists() and not args.resume:
        raise SystemExit(
            f"{run_dir} already exists; use --resume or a new --run-id"
        )
    run_dir.mkdir(exist_ok=True)
    (run_dir / "trials").mkdir(exist_ok=True)
    (run_dir / "fold_predictions").mkdir(exist_ok=True)
    invocation_started = time.time()

    data, user_mapping, item_mapping, counts = prepare_legacy_interactions(raw)
    expected_counts = read_json(DATA_MANIFEST)["processed_training_contract"]
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

    inputs = [
        data["user_id"].to_numpy(dtype=np.int64),
        data["item_id"].to_numpy(dtype=np.int64),
    ]
    labels = data["Score"].to_numpy(dtype=np.float64)
    if args.protocol == "faithful":
        splits = create_row_splits(len(data), args.folds)
    else:
        splits = create_pair_grouped_splits(data, args.folds)
    diagnostics = split_diagnostics(data, splits)
    config = {
        "seed": SEED,
        "folds": args.folds,
        "trials_per_fold": args.trials,
        "pretrain_epochs": args.pretrain_epochs,
        "pretrain_patience": args.pretrain_patience,
        "finetune_epochs": args.finetune_epochs,
        "finetune_patience": args.finetune_patience,
        "device": "cpu",
        "split": (
            "row_kfold"
            if args.protocol == "faithful"
            else "canonical_pair_grouped"
        ),
        "tuner_scope": {
            "faithful": "global_across_folds",
            "corrected": "fresh_per_fold",
        }[args.protocol],
        "full_historical_configuration": (
            args.protocol == "faithful"
            and args.folds == 10
            and args.trials == 50
            and args.pretrain_epochs == 30
            and args.pretrain_patience == 5
            and args.finetune_epochs == 20
            and args.finetune_patience == 3
        ),
    }
    execution_environment = environment()
    run_identity = {
        "protocol": args.protocol,
        "configuration": config,
        "input_hashes": input_hashes,
        "runtime_code_hashes": runtime_code_hashes(),
        "split_diagnostics": diagnostics,
        "environment": execution_environment,
    }
    manifest = {
        "status": "running",
        "protocol": args.protocol,
        "configuration": config,
        "data": counts,
        "data_checks": count_checks,
        "training_input": input_record,
        "input_hashes": input_hashes,
        "split_diagnostics": diagnostics,
        "environment": execution_environment,
        "runtime_code_hashes": run_identity["runtime_code_hashes"],
    }

    if args.resume:
        if not resume_path.exists():
            raise SystemExit("--resume requested but resume_state.pth is absent")
        state = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if int(state.get("version", -1)) != RESUME_STATE_VERSION:
            raise RuntimeError(
                "Resume-state version mismatch; refusing to combine runs"
            )
        if state.get("run_identity") != run_identity:
            raise RuntimeError(
                "Resume identity mismatch (configuration, inputs, splits, "
                "runner code, or execution environment changed); refusing "
                "to combine runs"
            )
        if bool(state.get("complete")):
            raise SystemExit(
                f"{run_dir} is already complete; refusing to alter it"
            )
        elapsed_before_invocation = float(state["elapsed_seconds"])
        restore_rng(state)
    else:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        elapsed_before_invocation = 0.0
        state = initial_resume_state(args.protocol, run_identity)
        checkpoint_resume(
            resume_path,
            state,
            invocation_started,
            elapsed_before_invocation,
        )
    dump_json(run_dir / "manifest.json", manifest)

    number_users = len(user_mapping)
    number_items = len(item_mapping)
    for fold_index in range(int(state["next_fold"]), args.folds):
        train_index, validation_index = splits[fold_index]
        train_inputs = [
            inputs[0][train_index],
            inputs[1][train_index],
        ]
        validation_inputs = [
            inputs[0][validation_index],
            inputs[1][validation_index],
        ]
        train_labels = labels[train_index]
        validation_labels = labels[validation_index]
        first_trial = (
            int(state["next_trial"])
            if fold_index == int(state["next_fold"])
            else 0
        )
        if (
            args.protocol == "corrected"
            and first_trial == 0
            and state["tuner_source_fold"] is not None
        ):
            state["tuner_best_score"] = float("inf")
            state["tuner_best_weights"] = None
            state["tuner_best_hp"] = None
            state["tuner_source_fold"] = None
            state["tuner_source_trial"] = None
        for trial_index in range(first_trial, args.trials):
            trial_started = time.time()
            hp = sample_hyperparameters()
            weights, trial_loss, detail = run_trial(
                hp,
                number_users,
                number_items,
                train_inputs,
                train_labels,
                validation_inputs,
                validation_labels,
                args.pretrain_epochs,
                args.pretrain_patience,
                args.finetune_epochs,
                args.finetune_patience,
            )
            updated = trial_loss < float(state["tuner_best_score"])
            if updated:
                state["tuner_best_score"] = trial_loss
                state["tuner_best_weights"] = weights
                state["tuner_best_hp"] = hp
                state["tuner_source_fold"] = fold_index + 1
                state["tuner_source_trial"] = trial_index + 1
            record = {
                "fold": fold_index + 1,
                "trial": trial_index + 1,
                "hyperparameters": hp,
                "ncf_best_legacy_mse": trial_loss,
                "updated_tuner": updated,
                "tuner_best_score_after": float(
                    state["tuner_best_score"]
                ),
                "tuner_source_fold_after": state["tuner_source_fold"],
                "tuner_source_trial_after": state["tuner_source_trial"],
                "seconds": time.time() - trial_started,
            }
            state["trial_records"].append(record)
            dump_json(
                run_dir
                / "trials"
                / f"fold_{fold_index + 1:02d}_trial_{trial_index + 1:03d}.json",
                {**record, "stage_detail": detail},
            )
            state["next_fold"] = fold_index
            state["next_trial"] = trial_index + 1
            checkpoint_resume(
                resume_path,
                state,
                invocation_started,
                elapsed_before_invocation,
            )
            write_progress(run_dir, state)
            print(
                json.dumps(
                    {
                        "event": "trial_complete",
                        **record,
                    }
                ),
                flush=True,
            )

        if state["tuner_best_weights"] is None:
            raise RuntimeError("Fold ended without tuner weights")
        predictions, metrics = evaluate_fold_model(
            state["tuner_best_weights"],
            state["tuner_best_hp"],
            number_users,
            number_items,
            validation_inputs,
            validation_labels,
        )
        prediction_frame = pd.DataFrame(
            {
                "source_row_index": validation_index,
                "label": validation_labels,
                "prediction": predictions,
                "squared_error": (
                    predictions - validation_labels
                )
                ** 2,
            }
        )
        prediction_path = (
            run_dir
            / "fold_predictions"
            / f"fold_{fold_index + 1:02d}.csv"
        )
        prediction_frame.to_csv(prediction_path, index=False)
        source_fold = int(state["tuner_source_fold"])
        source_train = splits[source_fold - 1][0]
        source_train_set = set(source_train.tolist())
        prior_model_row_seen_fraction = sum(
            int(index) in source_train_set for index in validation_index
        ) / len(validation_index)
        fold_record = {
            "fold": fold_index + 1,
            **metrics,
            "validation_rows": len(validation_index),
            "selected_tuner_score": float(state["tuner_best_score"]),
            "selected_model_source_fold": source_fold,
            "selected_model_source_trial": int(
                state["tuner_source_trial"]
            ),
            "validation_rows_seen_by_selected_model_source_training_fraction": (
                prior_model_row_seen_fraction
            ),
            "prediction_csv_sha256": sha256_file(prediction_path),
            "selected_hyperparameters_json": json.dumps(
                state["tuner_best_hp"], sort_keys=True
            ),
        }
        state["fold_records"].append(fold_record)
        state["next_fold"] = fold_index + 1
        state["next_trial"] = 0
        if args.protocol == "corrected":
            state["tuner_best_score"] = float("inf")
            state["tuner_best_weights"] = None
            state["tuner_best_hp"] = None
            state["tuner_source_fold"] = None
            state["tuner_source_trial"] = None
        checkpoint_resume(
            resume_path,
            state,
            invocation_started,
            elapsed_before_invocation,
        )
        write_progress(run_dir, state)
        print(
            json.dumps(
                {"event": "fold_complete", **fold_record}
            ),
            flush=True,
        )

    checkpoint_resume(
        resume_path,
        state,
        invocation_started,
        elapsed_before_invocation,
    )
    write_progress(run_dir, state)
    summary = final_summary(
        run_dir,
        state,
        args.protocol,
        config,
        diagnostics,
        input_hashes,
        data_path,
    )
    manifest["status"] = "complete"
    dump_json(run_dir / "manifest.json", manifest)
    state["complete"] = True
    checkpoint_resume(
        resume_path,
        state,
        invocation_started,
        elapsed_before_invocation,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resumable historical CV replay for the ID-only pretrained NCF "
            "baseline. The faithful protocol regenerates the reported "
            "manuscript RMSE."
        )
    )
    parser.add_argument(
        "--protocol",
        choices=["faithful", "corrected"],
        default="faithful",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument(
        "--allow-unverified-inputs",
        action="store_true",
        help=(
            "Run on different inputs while recording that reference checks "
            "do not apply."
        ),
    )
    args = parser.parse_args()
    if args.folds < 2:
        parser.error("--folds must be at least 2")
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("--run-id must be one local directory name")
    if min(
        args.trials,
        args.pretrain_epochs,
        args.pretrain_patience,
        args.finetune_epochs,
        args.finetune_patience,
    ) < 1:
        parser.error("trial, epoch, and patience values must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
