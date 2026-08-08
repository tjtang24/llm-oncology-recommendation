#!/usr/bin/env python3
"""Run scoring-rule sensitivity analysis for the Hybrid-NCF recommender.

This script is intentionally standalone: it reads the existing
``cancer_rec_system`` data/model artifacts but writes all new outputs under
``results/sensitivity`` by default.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pickle
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


REAL_NCT_RE = re.compile(r"^NCT\d{8}$")
BAD_STATUSES = {"terminated", "suspended", "withdrawn"}
UNKNOWN_STATUS_PENALTY_STATUSES = {"active not recruiting", "not yet recruiting", "unknown"}
DEFAULT_TARGET_CANCERS = ["non-small cell lung cancer", "melanoma: cutaneous"]
NSCLC_SIGNAL_INTERVENTIONS = {
    "ROS1/NTRK cluster": "entrectinib, larotrectinib, repotrectinib",
    "ALK cluster": "alectinib, brigatinib, ceritinib, crizotinib, lorlatinib",
    "Mixed targeted therapy cluster": (
        "anlotinib, dabrafenib, entrectinib, larotrectinib, lenvatinib, "
        "pralsetinib, selpercatinib, trametinib"
    ),
}
MELANOMA_SIGNAL_INTERVENTIONS = {
    "PARP inhibitor cluster": "niraparib, olaparib, rucaparib",
}


@dataclass(frozen=True)
class ScoringConfig:
    phase_weights: Dict[int, float] = None
    bad_status_penalty: float = 1.0
    nonpromising_penalty: float = 1.0
    unknown_penalty: float = 0.5
    time_decay: float = 0.05
    nccn_score: float = 4.0
    use_year_adjustment: bool = True
    use_status_outcome_adjustment: bool = True

    def __post_init__(self):
        if self.phase_weights is None:
            object.__setattr__(
                self,
                "phase_weights",
                {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0},
            )


def build_scoring_variants() -> Dict[str, ScoringConfig]:
    baseline = ScoringConfig()
    return {
        "baseline": baseline,
        "mild_status_outcome_penalties": replace(
            baseline,
            bad_status_penalty=0.5,
            nonpromising_penalty=0.5,
            unknown_penalty=0.25,
        ),
        "strong_status_outcome_penalties": replace(
            baseline,
            bad_status_penalty=1.5,
            nonpromising_penalty=1.5,
            unknown_penalty=0.75,
        ),
        "no_year_adjustment": replace(baseline, use_year_adjustment=False),
        "nccn_score_3_0": replace(baseline, nccn_score=3.0),
        "nccn_score_3_5": replace(baseline, nccn_score=3.5),
        "phase_nccn_neutral": replace(
            baseline,
            phase_weights={1: 2.5, 2: 2.5, 3: 2.5, 4: 2.5},
            nccn_score=2.5,
        ),
        "no_status_outcome_adjustment": replace(
            baseline,
            use_status_outcome_adjustment=False,
        ),
    }


def canonical_intervention(value: object) -> str:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    # Match train_cancer_ncf_pretrain.py:get_canonical_name exactly.
    return ", ".join(sorted(parts))


def normalized_intervention_key(value: object) -> str:
    return re.sub(r"\s+", " ", canonical_intervention(value)).strip().lower()


def intervention_token_set(value: object) -> set:
    return set(part.strip().lower() for part in str(value).split(",") if part.strip())


def parse_phase(value: object) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    if text == "":
        return None
    if "|" in text:
        parsed = [parse_phase(part) for part in text.split("|")]
        parsed = [phase for phase in parsed if phase is not None]
        return max(parsed) if parsed else None
    match = re.search(r"([1-4])", text)
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if 1 <= number <= 4 else None


def is_real_nct(value: object) -> bool:
    return bool(REAL_NCT_RE.match(str(value).strip()))


def nccn_mask(df: pd.DataFrame) -> pd.Series:
    arm_name = df.get("Arm Name", pd.Series("", index=df.index)).fillna("").astype(str)
    arm_type = df.get("Arm Type", pd.Series("", index=df.index)).fillna("").astype(str)
    sponsor = df.get("Sponsor", pd.Series("", index=df.index)).fillna("").astype(str)
    nct = df.get("NCT Number", pd.Series("", index=df.index)).fillna("").astype(str)
    phase = df.get("Phases", pd.Series("", index=df.index)).fillna("").astype(str)
    return (
        arm_name.str.fullmatch("Arm NCCN", case=False)
        | arm_name.str.contains("NCCN First/Second Line", case=False, regex=False)
        | arm_type.str.fullmatch("First/Second Line", case=False)
        | sponsor.str.fullmatch("First/Second Line", case=False)
        | (nct.str.startswith("NCT") & ~nct.str.match(REAL_NCT_RE))
        | (
            phase.str.upper().eq("PHASE4")
            & arm_type.str.fullmatch("First/Second Line", case=False)
        )
    )


def compute_scores(df: pd.DataFrame, config: ScoringConfig) -> pd.Series:
    raw_scores: List[float] = []
    is_nccn = nccn_mask(df)
    for idx, row in df.iterrows():
        if bool(is_nccn.loc[idx]):
            raw_scores.append(float(config.nccn_score))
            continue

        phase_number = parse_phase(row.get("Phases"))
        phase_score = config.phase_weights.get(phase_number, float("nan"))
        if math.isnan(phase_score):
            raw_scores.append(float("nan"))
            continue

        score = phase_score
        if config.use_status_outcome_adjustment:
            status = str(row.get("Study Status", "")).strip().lower()
            promising = str(row.get("Promising_Status", "")).strip().lower()
            if status in BAD_STATUSES:
                score = phase_score - config.bad_status_penalty
            elif status in UNKNOWN_STATUS_PENALTY_STATUSES:
                score = phase_score - config.unknown_penalty
            elif promising == "promising":
                score = phase_score
            elif promising == "not promising":
                score = phase_score - config.nonpromising_penalty
            else:
                score = phase_score - config.unknown_penalty
        raw_scores.append(float(score))

    score_series = pd.Series(raw_scores, index=df.index, dtype=float)
    if {"NCT Number", "Arm Name"}.issubset(df.columns):
        score_series = score_series.groupby(
            [df["NCT Number"], df["Arm Name"]],
            dropna=False,
        ).transform("mean")

    adjusted_scores: List[float] = []
    for idx, row in df.iterrows():
        score = float(score_series.loc[idx])
        if bool(is_nccn.loc[idx]):
            adjusted_scores.append(score)
            continue
        year = row.get("Year")
        if config.use_year_adjustment and pd.notna(year):
            try:
                year_value = float(year)
            except (TypeError, ValueError):
                year_value = float("nan")
            if pd.notna(year_value) and year_value < 2020:
                score -= config.time_decay * (2020 - year_value)
        adjusted_scores.append(float(score))
    return pd.Series(adjusted_scores, index=df.index, dtype=float)


def source_id_type(row: pd.Series) -> str:
    if bool(nccn_mask(pd.DataFrame([row])).iloc[0]):
        return "nccn_internal_id"
    source_id = str(row.get("NCT Number", "")).strip()
    if is_real_nct(source_id):
        return "clinicaltrials_nct"
    if source_id.startswith("NCT"):
        return "internal_or_nonstandard_nct"
    return "unknown"


def prepare_modeling_data(cancer_dir: Path) -> pd.DataFrame:
    data_path = cancer_dir.parent / "private_data" / "summary_w_score.csv"
    data = pd.read_csv(data_path)
    data.reset_index(drop=True, inplace=True)
    data = data.dropna(subset=["Cancer", "Intervention"]).copy()
    data["Cancer"] = data["Cancer"].astype(str).str.split(",")
    data = data.explode("Cancer").reset_index(drop=True)
    data["Cancer"] = data["Cancer"].astype(str).str.strip().str.lower()
    data["Intervention"] = data["Intervention"].apply(canonical_intervention)
    data["row_indices"] = np.arange(len(data)) + 1

    treatment_counts = data.groupby("Cancer")["Intervention"].count()
    eligible_cancers = treatment_counts[treatment_counts >= 20].index
    data = data[data["Cancer"].isin(eligible_cancers)].copy().reset_index(drop=True)
    return data


def load_pickle(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@contextlib.contextmanager
def pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_mutation_module(cancer_dir: Path):
    if str(cancer_dir) not in sys.path:
        sys.path.insert(0, str(cancer_dir))
    with pushd(cancer_dir):
        import mutation  # type: ignore

    return mutation


def build_intervention_vectors(
    interventions: Sequence[str],
    drug_vectors: pd.DataFrame,
) -> pd.DataFrame:
    vectors = pd.DataFrame(columns=drug_vectors.columns, index=list(interventions))
    for intervention in interventions:
        drugs = [drug.strip() for drug in str(intervention).split(",") if drug.strip()]
        relevant = drug_vectors.reindex(drugs).fillna(0)
        vectors.loc[intervention] = relevant.sum().clip(upper=1)
    return vectors.astype(float)


def make_fixed_mappings(data: pd.DataFrame, param_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    user_mapping = load_pickle(param_dir / "user_mapping.pkl")
    item_mapping = load_pickle(param_dir / "item_mapping.pkl")
    missing_users = sorted(set(data["Cancer"]) - set(user_mapping))
    missing_items = sorted(set(data["Intervention"]) - set(item_mapping))
    if missing_users or missing_items:
        raise ValueError(
            "Prepared data does not match saved baseline mappings: "
            f"{len(missing_users)} missing users, {len(missing_items)} missing items"
        )
    return user_mapping, item_mapping


def fixed_split(n_rows: int, seed: int, train_frac: float = 0.70, val_frac: float = 0.15):
    rng = np.random.RandomState(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    train_end = int(n_rows * train_frac)
    val_end = train_end + int(n_rows * val_frac)
    return {
        "train": indices[:train_end],
        "validation": indices[train_end:val_end],
        "test": indices[val_end:],
    }


def score_distribution(variant: str, scores: pd.Series) -> Dict[str, float]:
    clean = scores.dropna()
    return {
        "variant": variant,
        "n": int(clean.shape[0]),
        "min": clean.min(),
        "q1": clean.quantile(0.25),
        "median": clean.median(),
        "q3": clean.quantile(0.75),
        "max": clean.max(),
        "mean": clean.mean(),
        "sd": clean.std(ddof=1),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.std(y_true, ddof=1))
    return rmse(y_true, y_pred) / denom if denom > 0 else float("nan")


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def build_torch_model_classes():
    import torch
    import torch.nn as nn

    class HybridGMF(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, gmf_dim):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.gmf_user_proj = nn.Linear(embedding_dim + user_feature_dim, gmf_dim)
            self.gmf_item_proj = nn.Linear(embedding_dim + item_feature_dim, gmf_dim)
            self.output_layer = nn.Linear(gmf_dim, 1)

        def forward(self, user_input, item_input, user_features, item_features):
            user_emb = self.user_embedding(user_input)
            item_emb = self.item_embedding(item_input)
            user_vec = torch.cat([user_emb, user_features], dim=-1)
            item_vec = torch.cat([item_emb, item_features], dim=-1)
            gmf = self.gmf_user_proj(user_vec) * self.gmf_item_proj(item_vec)
            return self.output_layer(gmf).view(-1)

    class HybridMLP(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, hidden_units, dropout_rate):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            input_dim = embedding_dim + user_feature_dim + embedding_dim + item_feature_dim
            layers = []
            for units in hidden_units:
                layers.extend([nn.Linear(input_dim, units), nn.GELU(), nn.Dropout(dropout_rate)])
                input_dim = units
            self.mlp_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(hidden_units[-1], 1)

        def forward(self, user_input, item_input, user_features, item_features):
            user_emb = self.user_embedding(user_input)
            item_emb = self.item_embedding(item_input)
            user_vec = torch.cat([user_emb, user_features], dim=-1)
            item_vec = torch.cat([item_emb, item_features], dim=-1)
            mlp = self.mlp_layers(torch.cat([user_vec, item_vec], dim=-1))
            return self.output_layer(mlp).view(-1)

    class HybridNCF(nn.Module):
        def __init__(self, num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, gmf_dim, hidden_units, dropout_rate):
            super().__init__()
            self.user_embedding = nn.Embedding(num_users, embedding_dim)
            self.item_embedding = nn.Embedding(num_items, embedding_dim)
            self.gmf_user_proj = nn.Linear(embedding_dim + user_feature_dim, gmf_dim)
            self.gmf_item_proj = nn.Linear(embedding_dim + item_feature_dim, gmf_dim)
            input_dim = embedding_dim + user_feature_dim + embedding_dim + item_feature_dim
            layers = []
            for units in hidden_units:
                layers.extend([nn.Linear(input_dim, units), nn.GELU(), nn.Dropout(dropout_rate)])
                input_dim = units
            self.mlp_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(gmf_dim + hidden_units[-1], 1)

        def forward(self, user_input, item_input, user_features, item_features):
            user_emb = self.user_embedding(user_input)
            item_emb = self.item_embedding(item_input)
            user_vec = torch.cat([user_emb, user_features], dim=-1)
            item_vec = torch.cat([item_emb, item_features], dim=-1)
            gmf = self.gmf_user_proj(user_vec) * self.gmf_item_proj(item_vec)
            mlp = self.mlp_layers(torch.cat([user_vec, item_vec], dim=-1))
            return self.output_layer(torch.cat([gmf, mlp], dim=-1)).view(-1)

    return HybridGMF, HybridMLP, HybridNCF


def tensor_dataset(data: pd.DataFrame, labels: np.ndarray, user_features: pd.DataFrame, item_features: pd.DataFrame):
    import torch
    from torch.utils.data import TensorDataset

    user_ids = data["user_id"].to_numpy(dtype=np.int64)
    item_ids = data["item_id"].to_numpy(dtype=np.int64)
    user_names = data["Cancer"].tolist()
    item_names = data["Intervention"].tolist()
    user_feat = torch.tensor(user_features.loc[user_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    item_feat = torch.tensor(item_features.loc[item_names].to_numpy(dtype=np.float32), dtype=torch.float32)
    return TensorDataset(
        torch.tensor(user_ids, dtype=torch.long),
        torch.tensor(item_ids, dtype=torch.long),
        user_feat,
        item_feat,
        torch.tensor(labels.astype(np.float32), dtype=torch.float32),
    )


def train_with_early_stopping(
    model,
    train_loader,
    val_loader,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    device,
):
    import copy
    import torch
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    epochs_ran = 0
    no_improve = 0
    model.to(device)
    for epoch in range(1, epochs + 1):
        epochs_ran = epoch
        model.train()
        for user, item, user_feat, item_feat, rating in train_loader:
            user = user.to(device)
            item = item.to(device)
            user_feat = user_feat.to(device)
            item_feat = item_feat.to(device)
            rating = rating.to(device)
            optimizer.zero_grad()
            loss = criterion(model(user, item, user_feat, item_feat), rating)
            loss.backward()
            optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for user, item, user_feat, item_feat, rating in val_loader:
                user = user.to(device)
                item = item.to(device)
                user_feat = user_feat.to(device)
                item_feat = item_feat.to(device)
                rating = rating.to(device)
                losses.append(criterion(model(user, item, user_feat, item_feat), rating).item())
        val_loss = float(np.mean(losses)) if losses else float("inf")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_loss, best_epoch, epochs_ran


def load_pretrained_weights(ncf_model, gmf_weights, mlp_weights, alpha: float = 0.5):
    import torch

    state = ncf_model.state_dict()
    state["gmf_user_proj.weight"] = gmf_weights["gmf_user_proj.weight"]
    state["gmf_user_proj.bias"] = gmf_weights["gmf_user_proj.bias"]
    state["gmf_item_proj.weight"] = gmf_weights["gmf_item_proj.weight"]
    state["gmf_item_proj.bias"] = gmf_weights["gmf_item_proj.bias"]
    for key in list(state.keys()):
        if "mlp_layers" in key:
            state[key] = mlp_weights[key]
    state["user_embedding.weight"] = (
        gmf_weights["user_embedding.weight"] + mlp_weights["user_embedding.weight"]
    ) / 2.0
    state["item_embedding.weight"] = (
        gmf_weights["item_embedding.weight"] + mlp_weights["item_embedding.weight"]
    ) / 2.0
    state["output_layer.weight"] = alpha * torch.cat(
        [gmf_weights["output_layer.weight"], mlp_weights["output_layer.weight"]],
        dim=1,
    )
    state["output_layer.bias"] = alpha * (
        gmf_weights["output_layer.bias"] + mlp_weights["output_layer.bias"]
    )
    ncf_model.load_state_dict(state)
    return ncf_model


def train_hybrid_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    user_mapping: Dict[str, int],
    item_mapping: Dict[str, int],
    hp: Dict[str, object],
    seed: int,
    args,
):
    import torch
    from torch.utils.data import DataLoader

    set_seeds(seed)
    HybridGMF, HybridMLP, HybridNCF = build_torch_model_classes()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = tensor_dataset(train_df, train_labels, user_features, item_features)
    val_ds = tensor_dataset(val_df, val_labels, user_features, item_features)
    batch_size = int(hp["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    num_users = len(user_mapping)
    num_items = len(item_mapping)
    user_feature_dim = user_features.shape[1]
    item_feature_dim = item_features.shape[1]
    embedding_dim = int(hp["embedding_dim"])
    gmf_dim = int(hp["gmf_dim"])
    hidden_units = [int(unit) for unit in hp["hidden_units"]]
    dropout_rate = float(hp["dropout_rate"])
    learning_rate = float(hp["learning_rate"])
    weight_decay = float(hp["weight_decay"])

    gmf = HybridGMF(num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, gmf_dim)
    gmf, _, gmf_best_epoch, gmf_epochs_ran = train_with_early_stopping(
        gmf,
        train_loader,
        val_loader,
        learning_rate,
        weight_decay,
        args.pretrain_epochs,
        args.pretrain_patience,
        device,
    )
    mlp = HybridMLP(num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, hidden_units, dropout_rate)
    mlp, _, mlp_best_epoch, mlp_epochs_ran = train_with_early_stopping(
        mlp,
        train_loader,
        val_loader,
        learning_rate,
        weight_decay,
        args.pretrain_epochs,
        args.pretrain_patience,
        device,
    )

    ncf = HybridNCF(num_users, num_items, embedding_dim, user_feature_dim, item_feature_dim, gmf_dim, hidden_units, dropout_rate)
    ncf = load_pretrained_weights(ncf, gmf.state_dict(), mlp.state_dict())
    ncf, _, ncf_best_epoch, ncf_epochs_ran = train_with_early_stopping(
        ncf,
        train_loader,
        val_loader,
        learning_rate * args.finetune_lr_scale,
        weight_decay,
        args.finetune_epochs,
        args.finetune_patience,
        device,
    )
    ncf.eval()
    return ncf, device, {
        "gmf_best_epoch": gmf_best_epoch,
        "gmf_epochs_ran": gmf_epochs_ran,
        "mlp_best_epoch": mlp_best_epoch,
        "mlp_epochs_ran": mlp_epochs_ran,
        "early_stopping_epoch": ncf_best_epoch,
        "finetune_epochs_ran": ncf_epochs_ran,
    }


def predict_dataframe(model, data: pd.DataFrame, labels: np.ndarray, user_features, item_features, device, batch_size: int):
    import torch
    from torch.utils.data import DataLoader

    ds = tensor_dataset(data, labels, user_features, item_features)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for user, item, user_feat, item_feat, _ in loader:
            out = model(
                user.to(device),
                item.to(device),
                user_feat.to(device),
                item_feat.to(device),
            )
            preds.extend(out.cpu().numpy().tolist())
    return np.array(preds, dtype=float)


def aligned_feature_tensors(user_mapping, item_mapping, user_features, item_features, device):
    import torch

    inv_user = {v: k for k, v in user_mapping.items()}
    inv_item = {v: k for k, v in item_mapping.items()}
    user_names = [inv_user[idx] for idx in range(len(inv_user))]
    item_names = [inv_item[idx] for idx in range(len(inv_item))]
    aligned_user = torch.tensor(user_features.loc[user_names].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    aligned_item = torch.tensor(item_features.loc[item_names].to_numpy(dtype=np.float32), dtype=torch.float32).to(device)
    return aligned_user, aligned_item


def top_recommendations(
    model,
    target_cancer: str,
    data: pd.DataFrame,
    user_mapping: Dict[str, int],
    item_mapping: Dict[str, int],
    aligned_user_features,
    aligned_item_features,
    device,
    top_n: int,
) -> pd.DataFrame:
    import torch

    if target_cancer not in user_mapping:
        return pd.DataFrame()
    user_id = user_mapping[target_cancer]
    inv_item = {v: k for k, v in item_mapping.items()}
    item_ids = np.array(sorted(inv_item), dtype=np.int64)
    item_tensor = torch.tensor(item_ids, dtype=torch.long).to(device)
    user_tensor = torch.full((len(item_ids),), user_id, dtype=torch.long).to(device)
    user_feat = aligned_user_features[user_tensor]
    item_feat = aligned_item_features[item_tensor]
    model.eval()
    with torch.no_grad():
        pred = model(user_tensor, item_tensor, user_feat, item_feat).cpu().numpy()
    full = pd.DataFrame(
        {
            "Intervention": [inv_item[idx] for idx in item_ids],
            "Intervention_id": item_ids,
            "predicted_score": pred,
        }
    )
    known = set(data.loc[data["Cancer"] == target_cancer, "Intervention"].unique())
    full = full[~full["Intervention"].isin(known)].copy()
    full = full.sort_values("predicted_score", ascending=False).head(top_n).reset_index(drop=True)
    metadata = data.drop_duplicates(subset=["Intervention"]).set_index("Intervention")
    rows = []
    for rank, row in enumerate(full.itertuples(index=False), 1):
        meta = metadata.loc[row.Intervention] if row.Intervention in metadata.index else pd.Series(dtype=object)
        rows.append(
            {
                "rank": rank,
                "intervention": row.Intervention,
                "normalized_intervention_key": normalized_intervention_key(row.Intervention),
                "predicted_score": float(row.predicted_score),
                "source_cancer": meta.get("Cancer", ""),
                "source_id": meta.get("NCT Number", ""),
                "source_id_type": source_id_type(meta) if not meta.empty else "unknown",
                "source_id_is_real_nct": is_real_nct(meta.get("NCT Number", "")) if not meta.empty else False,
                "phase": meta.get("Phases", ""),
                "status": meta.get("Study Status", ""),
                "year": meta.get("Year", ""),
            }
        )
    return pd.DataFrame(rows)


def overlap_at_k(base: Sequence[str], other: Sequence[str], k: int) -> float:
    base_set = {normalized_intervention_key(x) for x in base[:k]}
    other_set = {normalized_intervention_key(x) for x in other[:k]}
    return len(base_set & other_set) / float(k)


def clinical_signal_rows_exact(
    top100_by_variant: Dict[Tuple[str, str], pd.DataFrame],
    variants: Sequence[str],
) -> List[Dict[str, object]]:
    rows = []
    specs = [
        ("non-small cell lung cancer", NSCLC_SIGNAL_INTERVENTIONS),
        ("melanoma: cutaneous", MELANOMA_SIGNAL_INTERVENTIONS),
    ]
    for cancer, signals in specs:
        for signal_name, intervention in signals.items():
            canonical = normalized_intervention_key(intervention)
            for variant in variants:
                recs = top100_by_variant.get((variant, cancer), pd.DataFrame())
                match = recs[recs["intervention"].apply(lambda x: normalized_intervention_key(x) == canonical)]
                if match.empty:
                    rows.append(
                        {
                            "target_cancer": cancer,
                            "signal_name": signal_name,
                            "canonical_intervention": intervention,
                            "variant": variant,
                            "seed": "",
                            "rank_if_in_top100": "",
                            "recovered_top10": False,
                            "recovered_top20": False,
                            "predicted_score": "",
                            "source_cancer": "",
                            "source_id": "",
                            "source_id_type": "",
                            "phase": "",
                            "status": "",
                            "year": "",
                        }
                    )
                    continue
                row = match.iloc[0]
                rank = int(row["rank"])
                rows.append(
                    {
                        "target_cancer": cancer,
                        "signal_name": signal_name,
                        "canonical_intervention": intervention,
                        "variant": variant,
                        "seed": "",
                        "rank_if_in_top100": rank,
                        "recovered_top10": rank <= 10,
                        "recovered_top20": rank <= 20,
                        "predicted_score": row["predicted_score"],
                        "source_cancer": row["source_cancer"],
                        "source_id": row["source_id"],
                        "source_id_type": row["source_id_type"],
                        "phase": row.get("phase", ""),
                        "status": row.get("status", ""),
                        "year": row.get("year", ""),
                    }
                )
    return rows


def clinical_signal_rows_component(
    top100_by_variant: Dict[Tuple[str, str], pd.DataFrame],
    variants: Sequence[str],
) -> List[Dict[str, object]]:
    specs = [
        {
            "target_cancer": "non-small cell lung cancer",
            "signal_name": "ROS1/NTRK component signal",
            "tokens": {"entrectinib", "larotrectinib", "repotrectinib"},
            "min_count": 2,
        },
        {
            "target_cancer": "non-small cell lung cancer",
            "signal_name": "ALK component signal",
            "tokens": {"alectinib", "brigatinib", "ceritinib", "crizotinib", "lorlatinib"},
            "min_count": 2,
        },
        {
            "target_cancer": "non-small cell lung cancer",
            "signal_name": "BRAF/MEK component signal",
            "tokens": {"dabrafenib", "trametinib"},
            "min_count": 2,
        },
        {
            "target_cancer": "non-small cell lung cancer",
            "signal_name": "RET component signal",
            "tokens": {"pralsetinib", "selpercatinib"},
            "min_count": 1,
        },
        {
            "target_cancer": "melanoma: cutaneous",
            "signal_name": "PARP component signal",
            "tokens": {"niraparib", "olaparib", "rucaparib", "talazoparib"},
            "min_count": 1,
        },
    ]
    rows: List[Dict[str, object]] = []
    for spec in specs:
        cancer = spec["target_cancer"]
        target_tokens = spec["tokens"]
        min_count = int(spec["min_count"])
        for variant in variants:
            recs = top100_by_variant.get((variant, cancer), pd.DataFrame())
            best = None
            best_count = 0
            best_tokens: List[str] = []
            for _, rec in recs.iterrows():
                matched = sorted(intervention_token_set(rec["intervention"]) & target_tokens)
                count = len(matched)
                if count > best_count:
                    best = rec
                    best_count = count
                    best_tokens = matched
                if count >= min_count:
                    break
            if best is None or best_count == 0:
                rows.append(
                    {
                        "target_cancer": cancer,
                        "signal_name": spec["signal_name"],
                        "variant": variant,
                        "seed": "",
                        "component_match": False,
                        "matched_component_count": 0,
                        "matched_components": "",
                        "rank_if_in_top100": "",
                        "recovered_top10": False,
                        "recovered_top20": False,
                        "predicted_score": "",
                        "source_cancer": "",
                        "source_id": "",
                        "source_id_type": "",
                        "phase": "",
                        "status": "",
                        "year": "",
                    }
                )
                continue
            rank = int(best["rank"])
            rows.append(
                {
                    "target_cancer": cancer,
                    "signal_name": spec["signal_name"],
                    "variant": variant,
                    "seed": "",
                    "component_match": best_count >= min_count,
                    "matched_component_count": best_count,
                    "matched_components": ", ".join(best_tokens),
                    "rank_if_in_top100": rank,
                    "recovered_top10": rank <= 10,
                    "recovered_top20": rank <= 20,
                    "predicted_score": best["predicted_score"],
                    "source_cancer": best["source_cancer"],
                    "source_id": best["source_id"],
                    "source_id_type": best["source_id_type"],
                    "phase": best.get("phase", ""),
                    "status": best.get("status", ""),
                    "year": best.get("year", ""),
                }
            )
    return rows


def write_report(
    output_dir: Path,
    config: Dict[str, ScoringConfig],
    baseline_check: Dict[str, object],
    score_distributions: pd.DataFrame,
    hybrid_metrics: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    clinical_retention: pd.DataFrame,
    args,
):
    report = output_dir / "sensitivity_report.md"
    lines = [
        "# Scoring Sensitivity Analysis",
        "",
        "## Methods Paragraph",
        "",
        (
            "To assess robustness to heuristic choices in the scoring function, we recomputed "
            "cancer-treatment scores under alternative scoring schemes that perturbed trial "
            "status/outcome penalties, removed the temporal adjustment, reduced the NCCN-derived "
            "treatment score, and compressed phase-based baseline weights. For each variant, we "
            "retrained Hybrid-NCF using the same train-validation-test split, architecture, and "
            "hyperparameters as in the main analysis. We evaluated robustness using RMSE, "
            "normalized RMSE, baseline-variant Top-K overlap, and retention of clinically "
            "interpretable treatment signals in the melanoma and NSCLC case studies."
        ),
        "",
        "## Run Settings",
        "",
        f"- Smoke mode: {args.smoke}",
        f"- Seed: {args.seed}",
        f"- Variants: {', '.join(config.keys())}",
        f"- Baseline score max absolute difference: {baseline_check['max_abs_diff']}",
        f"- Baseline score mismatches: {baseline_check['mismatches']}",
        "",
        "## Hybrid-NCF Sensitivity Metrics",
        "",
        hybrid_metrics.to_markdown(index=False) if not hybrid_metrics.empty else "No metrics produced.",
        "",
        "## Recommendation Stability Summary",
        "",
        overlap_summary.to_markdown(index=False) if not overlap_summary.empty else "No overlap summary produced.",
        "",
        "## NSCLC Clinical Signal Retention",
        "",
        clinical_retention[clinical_retention["target_cancer"] == "non-small cell lung cancer"].to_markdown(index=False)
        if not clinical_retention.empty
        else "No clinical signal retention rows produced.",
        "",
        "## Melanoma Clinical Signal Retention",
        "",
        clinical_retention[clinical_retention["target_cancer"] == "melanoma: cutaneous"].to_markdown(index=False)
        if not clinical_retention.empty
        else "No clinical signal retention rows produced.",
        "",
        "## Score Distributions",
        "",
        score_distributions.to_markdown(index=False) if not score_distributions.empty else "No score distributions produced.",
        "",
        "## Limitations",
        "",
        (
            "- Predicted scores are unconstrained regression outputs and should be interpreted as "
            "ranking scores rather than literal clinical-trial phase numbers."
        ),
        (
            "- Guideline-derived rows are identified using NCCN/First-Second-Line arm metadata, "
            "nonstandard NCT-like identifiers, or PHASE4 guideline markers. These internal IDs "
            "should not be cited as ClinicalTrials.gov NCT identifiers."
        ),
        (
            "- This script focuses on Hybrid-NCF sensitivity. MF/NCF method-comparison rows are "
            "left as documented placeholders unless added in a future extension."
        ),
        (
            "- Top-N recommendations in this report come from fixed-split retraining for the "
            "sensitivity analysis. They are not expected to exactly match the deployed "
            "all-data production checkpoint recommendations."
        ),
    ]
    report.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: Optional[Sequence[str]] = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensitivity"))
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--variants", default="baseline,no_year_adjustment,phase_nccn_neutral,no_status_outcome_adjustment")
    parser.add_argument("--target-cancers", default=",".join(DEFAULT_TARGET_CANCERS))
    parser.add_argument("--smoke", action="store_true", help="Run a tiny quick check with fewer rows, variants, and epochs.")
    parser.add_argument("--smoke-rows", type=int, default=600)
    parser.add_argument("--limit-rows", type=int, default=0, help="Sample this many rows after full preprocessing; 0 disables sampling.")
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.1)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--n-jobs", default="auto", help="Reserved for future parallel execution; accepted for command compatibility.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.variants = "baseline,mild_status_outcome_penalties"
        args.pretrain_epochs = 1
        args.pretrain_patience = 1
        args.finetune_epochs = 1
        args.finetune_patience = 1

    repo_root = args.repo_root.resolve()
    cancer_dir = repo_root / "cancer_rec_system"
    param_dir = cancer_dir / "params" / "cancer-ncf-pretrain"
    output_dir = (repo_root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)

    variants_all = build_scoring_variants()
    variant_names = [name.strip() for name in args.variants.split(",") if name.strip()]
    variants = {name: variants_all[name] for name in variant_names}
    with open(output_dir / "scoring_variants_config.json", "w", encoding="utf-8") as handle:
        json.dump({name: asdict(cfg) for name, cfg in variants.items()}, handle, indent=2)

    full_data = prepare_modeling_data(cancer_dir)
    user_mapping, item_mapping = make_fixed_mappings(full_data, param_dir)
    baseline_recomputed = compute_scores(full_data, variants_all["baseline"])
    score_diff = (baseline_recomputed - full_data["Score"].astype(float)).abs()
    baseline_check = {
        "max_abs_diff": float(score_diff.max()),
        "mismatches": int((score_diff > 1e-8).sum()),
        "n": int(score_diff.shape[0]),
    }
    with open(output_dir / "baseline_score_check.json", "w", encoding="utf-8") as handle:
        json.dump(baseline_check, handle, indent=2)

    data = full_data
    row_limit = args.smoke_rows if args.smoke else args.limit_rows
    if row_limit and len(data) > row_limit:
        data = data.sample(n=row_limit, random_state=args.seed).sort_index().reset_index(drop=True)

    data["user_id"] = data["Cancer"].map(user_mapping)
    data["item_id"] = data["Intervention"].map(item_mapping)
    data = data.dropna(subset=["user_id", "item_id"]).copy()
    data["user_id"] = data["user_id"].astype(int)
    data["item_id"] = data["item_id"].astype(int)

    mutation = load_mutation_module(cancer_dir)
    hp = load_json(param_dir / "best_hyperparameters.json")
    split = fixed_split(len(data), args.seed)
    target_cancers = [cancer.strip() for cancer in args.target_cancers.split(",") if cancer.strip()]
    all_cancers = sorted(data["Cancer"].unique())
    user_features = mutation.cancer_vectors.astype(float)
    item_features = build_intervention_vectors(
        [name for name, _ in sorted(item_mapping.items(), key=lambda kv: kv[1])],
        mutation.drug_vectors.astype(float),
    )

    score_rows = []
    metrics_rows = []
    method_rows = []
    rec_rows = []
    top100_rec_rows = []
    baseline_top_by_cancer: Dict[str, pd.DataFrame] = {}
    top_by_variant_cancer: Dict[Tuple[str, str], pd.DataFrame] = {}
    top100_by_variant_cancer: Dict[Tuple[str, str], pd.DataFrame] = {}

    baseline_test_rmse = None
    baseline_test_nrmse = None

    for variant_name, config in variants.items():
        start = time.time()
        labels = compute_scores(data, config).to_numpy(dtype=float)
        score_rows.append(score_distribution(variant_name, pd.Series(labels)))
        train_idx = split["train"]
        val_idx = split["validation"]
        test_idx = split["test"]
        model, device, training_info = train_hybrid_model(
            data.iloc[train_idx],
            data.iloc[val_idx],
            labels[train_idx],
            labels[val_idx],
            user_features,
            item_features,
            user_mapping,
            item_mapping,
            hp,
            args.seed,
            args,
        )
        if args.save_models:
            import torch

            torch.save(model.state_dict(), output_dir / "models" / f"hybrid_{variant_name}_seed{args.seed}.pth")

        val_pred = predict_dataframe(
            model,
            data.iloc[val_idx],
            labels[val_idx],
            user_features,
            item_features,
            device,
            int(hp["batch_size"]),
        )
        test_pred = predict_dataframe(
            model,
            data.iloc[test_idx],
            labels[test_idx],
            user_features,
            item_features,
            device,
            int(hp["batch_size"]),
        )
        val_rmse = rmse(labels[val_idx], val_pred)
        test_rmse = rmse(labels[test_idx], test_pred)
        val_nrmse = nrmse(labels[val_idx], val_pred)
        test_nrmse = nrmse(labels[test_idx], test_pred)
        if variant_name == "baseline":
            baseline_test_rmse = test_rmse
            baseline_test_nrmse = test_nrmse
            method_rows.append(
                {
                    "method": "Hybrid-NCF",
                    "scoring_variant": "baseline",
                    "seed": args.seed,
                    "split": "test",
                    "rmse": test_rmse,
                    "mae": mae(labels[test_idx], test_pred),
                    "nrmse": test_nrmse,
                    "source": "fixed_split_retraining",
                    "notes": "Retrained by scripts/sensitivity_analysis.py",
                }
            )
            for method in ["MF", "NCF"]:
                method_rows.append(
                    {
                        "method": method,
                        "scoring_variant": "baseline",
                        "seed": "",
                        "split": "test",
                        "rmse": "",
                        "mae": "",
                        "nrmse": "",
                        "source": "not_rerun",
                        "notes": "Not rerun by this standalone Hybrid-NCF sensitivity script.",
                    }
                )

        metrics_rows.append(
            {
                "variant": variant_name,
                "seed": args.seed,
                "validation_rmse": val_rmse,
                "test_rmse": test_rmse,
                "validation_mae": mae(labels[val_idx], val_pred),
                "test_mae": mae(labels[test_idx], test_pred),
                "validation_nrmse": val_nrmse,
                "test_nrmse": test_nrmse,
                "delta_test_rmse_vs_baseline": test_rmse - baseline_test_rmse if baseline_test_rmse is not None else 0.0,
                "delta_test_nrmse_vs_baseline": test_nrmse - baseline_test_nrmse if baseline_test_nrmse is not None else 0.0,
                "early_stopping_epoch": training_info["early_stopping_epoch"],
                "finetune_epochs_ran": training_info["finetune_epochs_ran"],
                "gmf_best_epoch": training_info["gmf_best_epoch"],
                "mlp_best_epoch": training_info["mlp_best_epoch"],
                "runtime_seconds": time.time() - start,
            }
        )

        aligned_user, aligned_item = aligned_feature_tensors(
            user_mapping,
            item_mapping,
            user_features,
            item_features,
            device,
        )
        cancers_for_recs = all_cancers if not args.smoke else target_cancers
        top100_rows_for_variant = []
        for cancer in cancers_for_recs:
            top20 = top_recommendations(
                model,
                cancer,
                data,
                user_mapping,
                item_mapping,
                aligned_user,
                aligned_item,
                device,
                args.top_n,
            )
            top20.insert(0, "seed", args.seed)
            top20.insert(0, "variant", variant_name)
            top20.insert(0, "target_cancer", cancer)
            rec_rows.extend(top20.to_dict("records"))
            top100 = top_recommendations(
                model,
                cancer,
                data,
                user_mapping,
                item_mapping,
                aligned_user,
                aligned_item,
                device,
                100,
            )
            top100.insert(0, "seed", args.seed)
            top100.insert(0, "variant", variant_name)
            top100.insert(0, "target_cancer", cancer)
            top100_rows_for_variant.extend(top100.to_dict("records"))
            if cancer in ["non-small cell lung cancer", "melanoma: cutaneous"]:
                top100_by_variant_cancer[(variant_name, cancer)] = top100
        top100_rec_rows.extend(top100_rows_for_variant)
        cancers_for_overlap = all_cancers if not args.smoke else target_cancers
        for cancer in cancers_for_overlap:
            top20 = top_recommendations(
                model,
                cancer,
                data,
                user_mapping,
                item_mapping,
                aligned_user,
                aligned_item,
                device,
                20,
            )
            top_by_variant_cancer[(variant_name, cancer)] = top20
            if variant_name == "baseline":
                baseline_top_by_cancer[cancer] = top20

    score_df = pd.DataFrame(score_rows)
    if not score_df.empty:
        score_df.insert(1, "seed", args.seed)
    metrics_df = pd.DataFrame(metrics_rows)
    methods_df = pd.DataFrame(method_rows)
    rec_df = pd.DataFrame(rec_rows)
    top100_rec_df = pd.DataFrame(top100_rec_rows if "top100_rec_rows" in locals() else [])

    overlap_rows = []
    for variant_name in variants:
        for cancer in baseline_top_by_cancer:
            base = baseline_top_by_cancer[cancer]["intervention"].tolist()
            other = top_by_variant_cancer[(variant_name, cancer)]["intervention"].tolist()
            overlap_rows.append(
                {
                    "variant": variant_name,
                    "seed": args.seed,
                    "cancer": cancer,
                    "overlap_at_10": overlap_at_k(base, other, 10),
                    "overlap_at_20": overlap_at_k(base, other, 20),
                }
            )
    overlap_df = pd.DataFrame(overlap_rows)
    summary_rows = []
    if not overlap_df.empty:
        for variant_name, group in overlap_df.groupby("variant"):
            nsclc = group[group["cancer"] == "non-small cell lung cancer"]
            melanoma = group[group["cancer"] == "melanoma: cutaneous"]
            summary_rows.append(
                {
                    "variant": variant_name,
                    "seed": args.seed,
                    "avg_overlap_at_10_all_cancers": group["overlap_at_10"].mean(),
                    "avg_overlap_at_20_all_cancers": group["overlap_at_20"].mean(),
                    "nsclc_overlap_at_10": nsclc["overlap_at_10"].iloc[0] if not nsclc.empty else "",
                    "nsclc_overlap_at_20": nsclc["overlap_at_20"].iloc[0] if not nsclc.empty else "",
                    "melanoma_overlap_at_10": melanoma["overlap_at_10"].iloc[0] if not melanoma.empty else "",
                    "melanoma_overlap_at_20": melanoma["overlap_at_20"].iloc[0] if not melanoma.empty else "",
                }
            )
    overlap_summary = pd.DataFrame(summary_rows)
    clinical_exact_df = pd.DataFrame(clinical_signal_rows_exact(top100_by_variant_cancer, list(variants)))
    if not clinical_exact_df.empty:
        clinical_exact_df["seed"] = args.seed
    clinical_component_df = pd.DataFrame(clinical_signal_rows_component(top100_by_variant_cancer, list(variants)))
    if not clinical_component_df.empty:
        clinical_component_df["seed"] = args.seed

    score_df.to_csv(output_dir / "score_distribution_by_variant.csv", index=False)
    methods_df.to_csv(output_dir / "rmse_methods_baseline.csv", index=False)
    metrics_df.to_csv(output_dir / "hybrid_sensitivity_metrics.csv", index=False)
    rec_df.to_csv(output_dir / "recommendation_top20_by_variant.csv", index=False)
    top100_rec_df.to_csv(output_dir / "recommendation_top100_by_variant.csv", index=False)
    overlap_df.to_csv(output_dir / "baseline_variant_overlap.csv", index=False)
    overlap_summary.to_csv(output_dir / "overlap_summary.csv", index=False)
    clinical_exact_df.to_csv(output_dir / "clinical_signal_retention_exact.csv", index=False)
    clinical_component_df.to_csv(output_dir / "clinical_signal_retention_component.csv", index=False)
    write_report(output_dir, variants, baseline_check, score_df, metrics_df, overlap_summary, clinical_exact_df, args)
    print(f"Wrote sensitivity outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
