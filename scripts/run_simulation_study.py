#!/usr/bin/env python3
"""Run a controlled simulation benchmark for oncology recommendation models.

The benchmark is intentionally standalone: it generates synthetic
cancer-intervention scores under manuscript-focused data-generating conditions
and compares MF, NCF, and a feature-fusion Hybrid-NCF analogue. The goal is to
demonstrate when each model should work, and when Hybrid-NCF should not help.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import os
import platform
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import yaml
except ImportError:  # pragma: no cover - PyYAML is available in the target env.
    yaml = None


FEATURE_GAMMAS = {
    "informative": 1.5,
    "moderately_informative": 0.75,
    "uninformative": 0.0,
    "permuted_features": 1.5,
}
FEATURE_GAMMAS_BY_SIGNAL = {"standard": FEATURE_GAMMAS}
SIGNAL_DEFAULTS = {"standard": {"beta_latent": 0.7, "noise_sd": 0.3}}
DESIGN_VERSION = "manuscript_five_condition_v1"
CONDITION_IDS = (
    "linear_latent_warm",
    "nonlinear_latent_warm",
    "global_cold_informative",
    "global_cold_moderately_informative",
    "global_cold_permuted_features",
)
DGP_MODES = ("linear_latent", "nonlinear_latent", "feature_structured_cold")
SPLIT_MODES = ("warm", "global_cold_item")
MODEL_NAMES = ("MF", "NCF", "Hybrid-NCF")
MODEL_SEED_OFFSETS = {
    "MF": 11,
    "NCF": 17,
    "Hybrid-NCF": 23,
}


@dataclass(frozen=True)
class SyntheticData:
    true_scores: np.ndarray
    cancer_features: np.ndarray
    intervention_features: np.ndarray
    latent_signal: np.ndarray
    feature_signal: np.ndarray
    gamma_feature: float
    beta_latent: float
    noise_sd: float
    seed: int
    feature_setting: str
    condition_id: str
    signal_strength: str
    dgp_mode: str
    feature_diagnostics: dict[str, float]
    oracle_id_scores: np.ndarray
    oracle_feature_scores: np.ndarray


@dataclass(frozen=True)
class ObservedSplits:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    density: float
    seed: int
    popularity_biased: bool
    split_mode: str


@dataclass(frozen=True)
class ScenarioCondition:
    condition_id: str
    split_mode: str
    density: float
    dgp_mode: str
    feature_setting: str
    beta_latent: float
    gamma_feature: float
    noise_sd: float
    rho_feature: float = 0.0
    eta_bias: float = 0.0
    residual_item_bias_sd: float = 0.3
    cancer_bias_sd: float = 0.2
    feature_share_target: float | None = None


@dataclass(frozen=True)
class SimulationConfig:
    n_cancers: int = 55
    n_interventions: int = 3000
    feature_dim: int = 100
    latent_dim: int = 16
    conditions: tuple[str, ...] = CONDITION_IDS
    replicates: int = 10
    output_root: Path = Path("results/simulation_study")
    max_parallel_jobs: int = 1
    smoke_test: bool = False
    force: bool = False
    embedding_dim: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 100
    patience: int = 10
    beta_latent: float | None = None
    noise_sd: float | None = None
    score_mean: float = 1.3
    score_sd: float = 0.8
    score_min: float = -1.15
    score_max: float = 4.0
    active_features_per_entity: int = 10
    popularity_biased_sampling: bool = True
    compute_ranking: bool = False
    ranking_k: int = 20
    torch_threads: int = 1
    signal_strength: str = "standard"
    run_dir: Path | None = None


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def standardize_matrix(matrix: np.ndarray) -> np.ndarray:
    mean = float(matrix.mean())
    sd = float(matrix.std())
    if sd <= 1e-12:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((matrix - mean) / sd).astype(np.float32)


def fixed_k_sparse_binary_matrix(
    n_rows: int,
    n_cols: int,
    active_per_row: int,
    rng: np.random.Generator,
) -> np.ndarray:
    active = min(active_per_row, n_cols)
    matrix = np.zeros((n_rows, n_cols), dtype=np.float32)
    if active == 0:
        return matrix
    for row in range(n_rows):
        cols = rng.choice(n_cols, size=active, replace=False)
        matrix[row, cols] = 1.0
    return matrix


def feature_gamma(feature_setting: str, signal_strength: str) -> float:
    if signal_strength not in FEATURE_GAMMAS_BY_SIGNAL:
        raise ValueError(f"Unknown signal strength: {signal_strength}")
    gamma_map = FEATURE_GAMMAS_BY_SIGNAL[signal_strength]
    if feature_setting not in gamma_map:
        raise ValueError(f"Unknown feature setting: {feature_setting}")
    return gamma_map[feature_setting]


def normalize_split_mode(split_mode: str) -> str:
    return split_mode


def signal_default(signal_strength: str, name: str) -> float:
    if signal_strength not in SIGNAL_DEFAULTS:
        raise ValueError(f"Unknown signal strength: {signal_strength}")
    return float(SIGNAL_DEFAULTS[signal_strength][name])


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = left.reshape(-1).astype(np.float64)
    right_flat = right.reshape(-1).astype(np.float64)
    if float(left_flat.std()) <= 1e-12 or float(right_flat.std()) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left_flat, right_flat)[0, 1])


def transform_raw_scores(
    raw_scores: np.ndarray,
    raw_mean: float,
    raw_sd: float,
    score_mean: float,
    score_sd: float,
    score_min: float,
    score_max: float,
) -> np.ndarray:
    if raw_sd > 1e-12:
        scores = (raw_scores - raw_mean) / raw_sd * score_sd + score_mean
    else:
        scores = np.full_like(raw_scores, score_mean, dtype=np.float32)
    return np.clip(scores, score_min, score_max).astype(np.float32)


def feature_variance_share(
    feature_q_component: np.ndarray,
    feature_bias_component: np.ndarray,
    match_component: np.ndarray,
    total_signal: np.ndarray,
) -> float:
    feature_signal = feature_q_component + feature_bias_component + match_component
    denominator = float(np.var(total_signal))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.var(feature_signal) / denominator)


def calibrate_gamma_for_feature_share(
    target_share: float,
    beta_latent_component: np.ndarray,
    feature_q_component: np.ndarray,
    feature_bias_component: np.ndarray,
    feature_match: np.ndarray,
    cancer_bias_component: np.ndarray,
    item_bias_component: np.ndarray,
    noise: np.ndarray,
    initial_gamma: float,
) -> float:
    if target_share <= 0.0:
        return 0.0

    def share_for(gamma: float) -> float:
        match_component = gamma * feature_match
        total = (
            beta_latent_component
            + cancer_bias_component
            + item_bias_component
            + match_component
            + noise
        )
        return feature_variance_share(
            feature_q_component,
            feature_bias_component,
            match_component,
            total,
        )

    high = max(0.05, float(initial_gamma))
    for _ in range(20):
        if share_for(high) >= target_share:
            break
        high *= 1.5
    low = 0.0
    for _ in range(30):
        mid = (low + high) / 2.0
        if share_for(mid) < target_share:
            low = mid
        else:
            high = mid
    return float(high)


def simulation_conditions(config: SimulationConfig) -> tuple[ScenarioCondition, ...]:
    selected = set(config.conditions)
    unknown = selected - set(CONDITION_IDS)
    if unknown:
        raise ValueError(f"Unknown conditions: {sorted(unknown)}")

    conditions: list[ScenarioCondition] = []

    def add(
        condition_id: str,
        split_mode: str,
        density: float,
        dgp_mode: str,
        feature_setting: str,
        beta_latent: float,
        gamma_feature: float,
        noise_sd: float,
        rho_feature: float = 0.0,
        eta_bias: float = 0.0,
        residual_item_bias_sd: float = 0.3,
        cancer_bias_sd: float = 0.2,
        feature_share_target: float | None = None,
    ) -> None:
        if condition_id not in selected:
            return
        conditions.append(
            ScenarioCondition(
                condition_id=condition_id,
                split_mode=normalize_split_mode(split_mode),
                density=float(density),
                dgp_mode=dgp_mode,
                feature_setting=feature_setting,
                beta_latent=float(beta_latent),
                gamma_feature=float(gamma_feature),
                noise_sd=float(noise_sd),
                rho_feature=float(rho_feature),
                eta_bias=float(eta_bias),
                residual_item_bias_sd=float(residual_item_bias_sd),
                cancer_bias_sd=float(cancer_bias_sd),
                feature_share_target=feature_share_target,
            )
        )

    add(
        "linear_latent_warm",
        "warm",
        0.10,
        "linear_latent",
        "uninformative",
        beta_latent=1.0,
        gamma_feature=0.0,
        noise_sd=0.25,
    )
    add(
        "nonlinear_latent_warm",
        "warm",
        0.10,
        "nonlinear_latent",
        "uninformative",
        beta_latent=1.0,
        gamma_feature=0.0,
        noise_sd=0.25,
    )
    for condition_id, feature_setting, rho, eta, gamma, target_share in [
        ("global_cold_informative", "informative", 0.25, 0.05, 1.0, 0.30),
        ("global_cold_moderately_informative", "moderately_informative", 0.15, 0.03, 0.5, 0.16),
        ("global_cold_permuted_features", "permuted_features", 0.25, 0.05, 1.0, 0.30),
    ]:
        add(
            condition_id,
            "global_cold_item",
            0.05,
            "feature_structured_cold",
            feature_setting,
            beta_latent=0.5,
            gamma_feature=gamma,
            noise_sd=0.2,
            rho_feature=rho,
            eta_bias=eta,
            residual_item_bias_sd=0.1,
            cancer_bias_sd=0.2,
            feature_share_target=target_share,
        )
    return tuple(conditions)


def generate_synthetic_data(
    n_cancers: int,
    n_interventions: int,
    latent_dim: int,
    feature_dim: int,
    feature_setting: str,
    seed: int,
    beta_latent: float | None = None,
    noise_sd: float | None = None,
    score_mean: float = 1.3,
    score_sd: float = 0.8,
    score_min: float = -1.15,
    score_max: float = 4.0,
    active_features_per_entity: int = 5,
    signal_strength: str = "standard",
    condition_id: str = "custom",
    dgp_mode: str = "linear_latent",
    gamma_feature_override: float | None = None,
    rho_feature: float = 0.0,
    eta_bias: float = 0.0,
    residual_item_bias_sd: float = 0.3,
    cancer_bias_sd: float = 0.2,
    feature_share_target: float | None = None,
) -> SyntheticData:
    if dgp_mode not in DGP_MODES:
        raise ValueError(f"Unknown DGP mode: {dgp_mode}")
    if beta_latent is None:
        beta_latent = signal_default(signal_strength, "beta_latent")
    if noise_sd is None:
        noise_sd = signal_default(signal_strength, "noise_sd")

    rng = np.random.default_rng(seed)
    cancer_latent = rng.normal(0.0, 1.0, size=(n_cancers, latent_dim)).astype(np.float32)
    cancer_bias = rng.normal(0.0, cancer_bias_sd, size=n_cancers).astype(np.float32)

    true_cancer_features = fixed_k_sparse_binary_matrix(
        n_cancers,
        feature_dim,
        active_features_per_entity,
        rng,
    )
    true_intervention_features = fixed_k_sparse_binary_matrix(
        n_interventions,
        feature_dim,
        active_features_per_entity,
        rng,
    )
    cancer_features = true_cancer_features.copy()
    intervention_features = true_intervention_features.copy()
    if feature_setting == "permuted_features":
        cancer_features = cancer_features[rng.permutation(n_cancers)].copy()
        intervention_features = intervention_features[rng.permutation(n_interventions)].copy()

    if dgp_mode == "feature_structured_cold":
        mapping = rng.normal(
            0.0,
            1.0 / math.sqrt(max(1, latent_dim)),
            size=(feature_dim, latent_dim),
        ).astype(np.float32)
        q_feature = true_intervention_features @ mapping
        q_feature = standardize_matrix(q_feature)
        q_noise = rng.normal(0.0, 1.0, size=(n_interventions, latent_dim)).astype(np.float32)
        rho = float(np.clip(rho_feature, 0.0, 1.0))
        intervention_latent = (
            rho * q_feature
            + math.sqrt(max(0.0, 1.0 - rho**2)) * q_noise
        ).astype(np.float32)
        bias_weights = rng.normal(0.0, 1.0, size=feature_dim).astype(np.float32)
        bias_feature = standardize_matrix(true_intervention_features @ bias_weights).reshape(-1)
        bias_noise = rng.normal(0.0, residual_item_bias_sd, size=n_interventions).astype(np.float32)
        intervention_bias = (eta_bias * bias_feature + bias_noise).astype(np.float32)
    else:
        q_feature = np.zeros((n_interventions, latent_dim), dtype=np.float32)
        q_noise = rng.normal(0.0, 1.0, size=(n_interventions, latent_dim)).astype(np.float32)
        bias_feature = np.zeros(n_interventions, dtype=np.float32)
        intervention_latent = q_noise
        intervention_bias = rng.normal(0.0, residual_item_bias_sd, size=n_interventions).astype(np.float32)

    if dgp_mode == "linear_latent":
        effective_rank = min(8, latent_dim)
        latent_linear = (
            cancer_latent[:, :effective_rank]
            @ intervention_latent[:, :effective_rank].T
            / math.sqrt(effective_rank)
        )
    else:
        latent_linear = cancer_latent @ intervention_latent.T / math.sqrt(latent_dim)
    if dgp_mode == "nonlinear_latent":
        hidden_dim = 32
        pair_features = np.concatenate(
            [
                np.repeat(cancer_latent[:, None, :], n_interventions, axis=1),
                np.repeat(intervention_latent[None, :, :], n_cancers, axis=0),
            ],
            axis=2,
        )
        oracle_w1 = rng.normal(0.0, 1.0 / math.sqrt(2 * latent_dim), size=(2 * latent_dim, hidden_dim))
        oracle_w2 = rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=hidden_dim)
        oracle_hidden = np.tanh(pair_features @ oracle_w1)
        nonlinear_raw = 0.2 * latent_linear + 0.8 * (oracle_hidden @ oracle_w2)
        latent_signal = standardize_matrix(nonlinear_raw)
    else:
        latent_signal = standardize_matrix(latent_linear)
    feature_raw = true_cancer_features @ true_intervention_features.T
    feature_signal = standardize_matrix(feature_raw)
    gamma_feature = (
        float(gamma_feature_override)
        if gamma_feature_override is not None
        else feature_gamma(feature_setting, signal_strength)
    )

    noise = rng.normal(0.0, noise_sd, size=(n_cancers, n_interventions)).astype(np.float32)
    latent_component = beta_latent * latent_signal
    latent_raw_sd = float(latent_linear.std())
    if dgp_mode == "feature_structured_cold" and latent_raw_sd > 1e-12:
        q_feature_raw = cancer_latent @ (rho_feature * q_feature).T / math.sqrt(latent_dim)
        feature_q_component = beta_latent * (q_feature_raw / latent_raw_sd)
    else:
        feature_q_component = np.zeros((n_cancers, n_interventions), dtype=np.float32)
    feature_bias_component = np.tile((eta_bias * bias_feature).reshape(1, -1), (n_cancers, 1)).astype(np.float32)
    cancer_bias_component = cancer_bias[:, None]
    item_bias_component = intervention_bias[None, :]
    if feature_share_target is not None and gamma_feature > 0.0:
        gamma_feature = calibrate_gamma_for_feature_share(
            feature_share_target,
            latent_component,
            feature_q_component,
            feature_bias_component,
            feature_signal,
            cancer_bias_component,
            item_bias_component,
            noise,
            gamma_feature,
        )
    match_component = gamma_feature * feature_signal
    raw_signal = (
        latent_component
        + match_component
        + cancer_bias_component
        + item_bias_component
        + noise
    )
    raw_scores = score_mean + raw_signal
    raw_mean = float(raw_scores.mean())
    raw_sd = float(raw_scores.std())
    scores = transform_raw_scores(raw_scores, raw_mean, raw_sd, score_mean, score_sd, score_min, score_max)
    oracle_id_raw = score_mean + cancer_bias_component + np.zeros_like(raw_signal)
    oracle_feature_raw = score_mean + cancer_bias_component + feature_q_component + feature_bias_component + match_component
    oracle_id_scores = transform_raw_scores(
        oracle_id_raw,
        raw_mean,
        raw_sd,
        score_mean,
        score_sd,
        score_min,
        score_max,
    )
    oracle_feature_scores = transform_raw_scores(
        oracle_feature_raw,
        raw_mean,
        raw_sd,
        score_mean,
        score_sd,
        score_min,
        score_max,
    )
    feature_share = feature_variance_share(
        feature_q_component,
        feature_bias_component,
        match_component,
        raw_signal,
    )
    feature_diagnostics = {
        "var_latent_component": float(np.var(latent_component)),
        "var_feature_q_component": float(np.var(feature_q_component)),
        "var_feature_bias_component": float(np.var(feature_bias_component)),
        "var_match_component": float(np.var(match_component)),
        "var_noise": float(np.var(noise)),
        "total_feature_explainable_variance_share": feature_share,
        "corr_y_feature_match": safe_corr(raw_scores, feature_signal),
        "corr_y_Z_bias": safe_corr(raw_scores, feature_bias_component),
        "fraction_clipped_low": float(np.mean(scores <= score_min + 1e-8)),
        "fraction_clipped_high": float(np.mean(scores >= score_max - 1e-8)),
        "calibrated_gamma_match": float(gamma_feature),
        "rho_feature": float(rho_feature),
        "eta_bias": float(eta_bias),
        "residual_item_bias_sd": float(residual_item_bias_sd),
    }

    return SyntheticData(
        true_scores=scores,
        cancer_features=cancer_features,
        intervention_features=intervention_features,
        latent_signal=latent_signal,
        feature_signal=feature_signal,
        gamma_feature=gamma_feature,
        beta_latent=beta_latent,
        noise_sd=noise_sd,
        seed=seed,
        feature_setting=feature_setting,
        condition_id=condition_id,
        signal_strength=signal_strength,
        dgp_mode=dgp_mode,
        feature_diagnostics=feature_diagnostics,
        oracle_id_scores=oracle_id_scores,
        oracle_feature_scores=oracle_feature_scores,
    )


def sample_observed_pairs(
    n_cancers: int,
    n_interventions: int,
    density: float,
    rng: np.random.Generator,
    popularity_biased: bool,
) -> np.ndarray:
    n_total = n_cancers * n_interventions
    n_observed = max(3, min(n_total, int(n_total * density)))
    if popularity_biased:
        cancer_popularity = rng.normal(0.0, 0.5, size=n_cancers)
        intervention_popularity = rng.normal(0.0, 0.8, size=n_interventions)
        logits = cancer_popularity[:, None] + intervention_popularity[None, :]
        logits = logits - float(logits.max())
        weights = np.exp(logits).reshape(-1)
        probabilities = weights / float(weights.sum())
    else:
        probabilities = None
    flat = rng.choice(n_total, size=n_observed, replace=False, p=probabilities)
    rng.shuffle(flat)
    return np.column_stack((flat // n_interventions, flat % n_interventions)).astype(np.int64)


def split_counts(n_observed: int) -> tuple[int, int, int]:
    n_train = max(1, int(0.70 * n_observed))
    n_validation = max(1, int(0.15 * n_observed))
    if n_train + n_validation >= n_observed:
        n_validation = 1
        n_train = max(1, n_observed - 2)
    return n_train, n_validation, n_observed - n_train - n_validation


def item_train_counts(pairs: np.ndarray, n_interventions: int) -> np.ndarray:
    counts = np.zeros(n_interventions, dtype=np.int64)
    if len(pairs):
        np.add.at(counts, pairs[:, 1], 1)
    return counts


def split_warm_pairs(pairs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train, n_validation, _ = split_counts(len(pairs))
    train = pairs[:n_train]
    validation = pairs[n_train : n_train + n_validation]
    test = pairs[n_train + n_validation :]
    return train, validation, test


def sample_observed_splits(
    data: SyntheticData,
    density: float,
    seed: int,
    popularity_biased: bool = True,
    split_mode: str = "warm",
) -> ObservedSplits:
    if not 0.0 < density <= 1.0:
        raise ValueError(f"Density must be in (0, 1], got {density}")
    split_mode = normalize_split_mode(split_mode)
    if split_mode not in SPLIT_MODES:
        raise ValueError(f"Unknown split mode: {split_mode}")

    rng = np.random.default_rng(seed)
    n_cancers, n_interventions = data.true_scores.shape
    pairs = sample_observed_pairs(n_cancers, n_interventions, density, rng, popularity_biased)

    if split_mode == "global_cold_item":
        n_train, n_validation, n_test = split_counts(len(pairs))
        cancer_popularity = rng.normal(0.0, 0.5, size=n_cancers) if popularity_biased else None
        intervention_popularity = rng.normal(0.0, 0.8, size=n_interventions) if popularity_biased else None
        train, validation, test = sample_item_split_pairs(
            n_cancers=n_cancers,
            n_interventions=n_interventions,
            n_train=n_train,
            n_validation=n_validation,
            n_test=n_test,
            rng=rng,
            popularity_biased=popularity_biased,
            cancer_popularity=cancer_popularity,
            intervention_popularity=intervention_popularity,
            split_mode=split_mode,
        )
    elif split_mode == "warm":
        train, validation, test = split_warm_pairs(pairs)
    else:
        raise ValueError(f"Unsupported split mode in revised benchmark: {split_mode}")

    return ObservedSplits(
        train=train,
        validation=validation,
        test=test,
        density=density,
        seed=seed,
        popularity_biased=popularity_biased,
        split_mode=split_mode,
    )


def sample_pairs_from_item_pool(
    n_cancers: int,
    item_indices: np.ndarray,
    n_pairs: int,
    rng: np.random.Generator,
    popularity_biased: bool,
    cancer_popularity: np.ndarray | None,
    intervention_popularity: np.ndarray | None,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> np.ndarray:
    if n_pairs <= 0 or len(item_indices) == 0:
        return np.empty((0, 2), dtype=np.int64)
    exclude_pairs = exclude_pairs or set()
    users = np.repeat(np.arange(n_cancers, dtype=np.int64), len(item_indices))
    items = np.tile(item_indices.astype(np.int64), n_cancers)
    if exclude_pairs:
        keep = np.array(
            [(int(user), int(item)) not in exclude_pairs for user, item in zip(users, items)],
            dtype=bool,
        )
        users = users[keep]
        items = items[keep]
    capacity = len(users)
    if capacity == 0:
        return np.empty((0, 2), dtype=np.int64)
    n_pairs = min(n_pairs, capacity)
    if popularity_biased:
        logits = cancer_popularity[users] + intervention_popularity[items]
        logits = logits - float(logits.max())
        weights = np.exp(logits)
        probabilities = weights / float(weights.sum())
    else:
        probabilities = None
    selected = rng.choice(capacity, size=n_pairs, replace=False, p=probabilities)
    pairs = np.column_stack((users[selected], items[selected])).astype(np.int64)
    rng.shuffle(pairs)
    return pairs


def sample_item_split_pairs(
    n_cancers: int,
    n_interventions: int,
    n_train: int,
    n_validation: int,
    n_test: int,
    rng: np.random.Generator,
    popularity_biased: bool,
    cancer_popularity: np.ndarray | None,
    intervention_popularity: np.ndarray | None,
    split_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_indices = np.arange(n_interventions, dtype=np.int64)
    rng.shuffle(item_indices)
    n_test_items = max(1, int(round(0.20 * n_interventions)))
    n_validation_items = max(1, int(round(0.10 * n_interventions)))
    if n_test_items + n_validation_items >= n_interventions:
        n_test_items = max(1, n_interventions // 4)
        n_validation_items = max(1, n_interventions // 8)
    test_items = item_indices[:n_test_items]
    validation_items = item_indices[n_test_items : n_test_items + n_validation_items]
    train_items = item_indices[n_test_items + n_validation_items :]

    if split_mode == "global_cold_item":
        train = sample_pairs_from_item_pool(
            n_cancers,
            train_items,
            n_train,
            rng,
            popularity_biased,
            cancer_popularity,
            intervention_popularity,
        )
        validation = sample_pairs_from_item_pool(
            n_cancers,
            validation_items,
            n_validation,
            rng,
            popularity_biased,
            cancer_popularity,
            intervention_popularity,
        )
        test = sample_pairs_from_item_pool(
            n_cancers,
            test_items,
            n_test,
            rng,
            popularity_biased,
            cancer_popularity,
            intervention_popularity,
        )
        return train, validation, test

    raise ValueError(f"Unsupported item split mode: {split_mode}")


def scores_for_pairs(data: SyntheticData, pairs: np.ndarray) -> np.ndarray:
    return data.true_scores[pairs[:, 0], pairs[:, 1]].astype(np.float32)


class MatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, initial_mean: float):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(initial_mean), dtype=torch.float32))
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        cancer_features: torch.Tensor | None = None,
        intervention_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        user = self.user_embedding(user_idx)
        item = self.item_embedding(item_idx)
        dot = (user * item).sum(dim=1)
        return (
            self.global_bias
            + self.user_bias(user_idx).squeeze(1)
            + self.item_bias(item_idx).squeeze(1)
            + dot
        )


class NCF(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, initial_mean: float):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(initial_mean), dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(2 * embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        cancer_features: torch.Tensor | None = None,
        intervention_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        features = torch.cat(
            [self.user_embedding(user_idx), self.item_embedding(item_idx)],
            dim=1,
        )
        return (
            self.global_bias
            + self.user_bias(user_idx).squeeze(1)
            + self.item_bias(item_idx).squeeze(1)
            + self.network(features).squeeze(1)
        )


class HybridNCF(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int,
        feature_dim: int,
        initial_mean: float,
    ):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.tensor(float(initial_mean), dtype=torch.float32))
        input_dim = 2 * embedding_dim + 2 * feature_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.side_interaction = nn.Linear(feature_dim, 1, bias=False)
        self.match_scale = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.zeros_(self.side_interaction.weight)

    def forward(
        self,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
        cancer_features: torch.Tensor | None = None,
        intervention_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cancer_features is None or intervention_features is None:
            raise ValueError("HybridNCF requires cancer and intervention features.")
        features = torch.cat(
            [
                self.user_embedding(user_idx),
                cancer_features,
                self.item_embedding(item_idx),
                intervention_features,
            ],
            dim=1,
        )
        neural_score = self.network(features).squeeze(1)
        feature_cross = cancer_features * intervention_features
        side_score = self.side_interaction(feature_cross).squeeze(1)
        side_score = side_score + self.match_scale * feature_cross.sum(dim=1)
        return (
            self.global_bias
            + self.user_bias(user_idx).squeeze(1)
            + self.item_bias(item_idx).squeeze(1)
            + neural_score
            + side_score
        )


def create_model(
    model_name: str,
    data: SyntheticData,
    embedding_dim: int,
    train_mean: float,
) -> nn.Module:
    n_cancers, n_interventions = data.true_scores.shape
    feature_dim = data.cancer_features.shape[1]
    if model_name == "MF":
        return MatrixFactorization(n_cancers, n_interventions, embedding_dim, train_mean)
    if model_name == "NCF":
        return NCF(n_cancers, n_interventions, embedding_dim, train_mean)
    if model_name == "Hybrid-NCF":
        return HybridNCF(n_cancers, n_interventions, embedding_dim, feature_dim, train_mean)
    raise ValueError(f"Unknown model: {model_name}")


def apply_unseen_item_defaults(model: nn.Module, train_item_counts: np.ndarray) -> None:
    unseen = torch.as_tensor(train_item_counts == 0, dtype=torch.bool)
    with torch.no_grad():
        if hasattr(model, "item_embedding"):
            model.item_embedding.weight[unseen] = 0.0
        if hasattr(model, "item_bias"):
            model.item_bias.weight[unseen] = 0.0


def tensor_dataset_for_pairs(data: SyntheticData, pairs: np.ndarray) -> TensorDataset:
    users = torch.as_tensor(pairs[:, 0], dtype=torch.long)
    items = torch.as_tensor(pairs[:, 1], dtype=torch.long)
    targets = torch.as_tensor(scores_for_pairs(data, pairs), dtype=torch.float32)
    return TensorDataset(users, items, targets)


def predict_pairs(
    model: nn.Module,
    data: SyntheticData,
    pairs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    if len(pairs) == 0:
        return np.array([], dtype=np.float32)
    model.eval()
    cancer_features = torch.as_tensor(data.cancer_features, dtype=torch.float32, device=device)
    intervention_features = torch.as_tensor(data.intervention_features, dtype=torch.float32, device=device)
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start : start + batch_size]
            users = torch.as_tensor(batch[:, 0], dtype=torch.long, device=device)
            items = torch.as_tensor(batch[:, 1], dtype=torch.long, device=device)
            preds = model(
                users,
                items,
                cancer_features[users],
                intervention_features[items],
            )
            predictions.append(preds.detach().cpu().numpy())
    return np.concatenate(predictions).astype(np.float32)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if len(y_true) == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "nrmse": float("nan")}
    residual = y_pred - y_true
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    sd = float(np.std(y_true))
    nrmse = rmse / sd if sd > 1e-12 else float("nan")
    return {"rmse": rmse, "mae": mae, "nrmse": nrmse}


def evaluate_pairs(
    model: nn.Module,
    data: SyntheticData,
    pairs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    y_true = scores_for_pairs(data, pairs)
    y_pred = predict_pairs(model, data, pairs, batch_size, device)
    return regression_metrics(y_true, y_pred)


def item_count_bin(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 10:
        return "4-10"
    return ">10"


def stratified_item_count_metrics(
    model: nn.Module,
    data: SyntheticData,
    test_pairs: np.ndarray,
    train_item_counts: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> list[dict[str, float | int | str]]:
    if len(test_pairs) == 0:
        return []
    y_true = scores_for_pairs(data, test_pairs)
    y_pred = predict_pairs(model, data, test_pairs, batch_size, device)
    bins = np.array([item_count_bin(int(train_item_counts[item])) for item in test_pairs[:, 1]])
    rows = []
    for bin_name in ["0", "1", "2-3", "4-10", ">10"]:
        mask = bins == bin_name
        if not np.any(mask):
            continue
        metrics = regression_metrics(y_true[mask], y_pred[mask])
        rows.append(
            {
                "item_train_count_bin": bin_name,
                "n_test_pairs": int(mask.sum()),
                "test_rmse": metrics["rmse"],
            }
        )
    return rows


def train_one_model(
    model_name: str,
    data: SyntheticData,
    splits: ObservedSplits,
    config: SimulationConfig,
    seed: int,
) -> tuple[dict[str, float | int | str], list[dict[str, float | int | str]]]:
    set_global_seed(seed + MODEL_SEED_OFFSETS[model_name])
    torch.set_num_threads(max(1, int(config.torch_threads)))
    device = torch.device("cpu")
    train_targets = scores_for_pairs(data, splits.train)
    model = create_model(model_name, data, config.embedding_dim, float(train_targets.mean()))
    model.to(device)
    train_item_count = item_train_counts(splits.train, data.true_scores.shape[1])
    if splits.split_mode == "global_cold_item":
        apply_unseen_item_defaults(model, train_item_count)

    train_dataset = tensor_dataset_for_pairs(data, splits.train)
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed + MODEL_SEED_OFFSETS[model_name])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    cancer_features = torch.as_tensor(data.cancer_features, dtype=torch.float32, device=device)
    intervention_features = torch.as_tensor(data.intervention_features, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_fn = nn.MSELoss()
    best_epoch = 0
    best_val_rmse = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for users, items, targets in train_loader:
            users = users.to(device)
            items = items.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(
                users,
                items,
                cancer_features[users],
                intervention_features[items],
            )
            loss = loss_fn(predictions, targets)
            loss.backward()
            optimizer.step()
            if splits.split_mode == "global_cold_item":
                apply_unseen_item_defaults(model, train_item_count)

        val_metrics = evaluate_pairs(model, data, splits.validation, config.batch_size, device)
        if math.isnan(val_metrics["rmse"]) or val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    model.load_state_dict(best_state)
    if splits.split_mode == "global_cold_item":
        apply_unseen_item_defaults(model, train_item_count)
    val_metrics = evaluate_pairs(model, data, splits.validation, config.batch_size, device)
    test_metrics = evaluate_pairs(model, data, splits.test, config.batch_size, device)
    ranking = {"recall_at_20": float("nan"), "ndcg_at_20": float("nan")}
    if config.compute_ranking:
        ranking = compute_ranking_metrics(
            model,
            data,
            splits.train,
            batch_size=config.batch_size,
            device=device,
            k=config.ranking_k,
        )
    runtime = time.perf_counter() - started
    stratified_rows = stratified_item_count_metrics(
        model,
        data,
        splits.test,
        train_item_count,
        config.batch_size,
        device,
    )

    return {
        "model": model_name,
        "validation_rmse": val_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "test_nrmse": test_metrics["nrmse"],
        "test_mae": test_metrics["mae"],
        "best_epoch": best_epoch,
        "runtime_seconds": runtime,
        **ranking,
    }, stratified_rows


def top_k_indices(values: np.ndarray, k: int) -> np.ndarray:
    if len(values) <= k:
        return np.argsort(values)[::-1]
    partial = np.argpartition(values, -k)[-k:]
    return partial[np.argsort(values[partial])[::-1]]


def dcg_at_k(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def compute_ranking_metrics(
    model: nn.Module,
    data: SyntheticData,
    train_pairs: np.ndarray,
    batch_size: int,
    device: torch.device,
    k: int,
) -> dict[str, float]:
    n_cancers, n_interventions = data.true_scores.shape
    train_seen = [set() for _ in range(n_cancers)]
    for user, item in train_pairs:
        train_seen[int(user)].add(int(item))

    all_pairs = np.array(
        [(user, item) for user in range(n_cancers) for item in range(n_interventions)],
        dtype=np.int64,
    )
    predicted = predict_pairs(model, data, all_pairs, batch_size, device).reshape(n_cancers, n_interventions)
    recalls = []
    ndcgs = []
    for user in range(n_cancers):
        candidates = np.array(
            [item for item in range(n_interventions) if item not in train_seen[user]],
            dtype=np.int64,
        )
        if len(candidates) == 0:
            continue
        effective_k = min(k, len(candidates))
        true_scores = data.true_scores[user, candidates]
        pred_scores = predicted[user, candidates]
        true_top_local = top_k_indices(true_scores, effective_k)
        pred_top_local = top_k_indices(pred_scores, effective_k)
        true_top = set(candidates[true_top_local].tolist())
        pred_top = candidates[pred_top_local].tolist()
        hits = np.array([1.0 if item in true_top else 0.0 for item in pred_top], dtype=np.float32)
        recalls.append(float(hits.sum() / effective_k))
        ideal = dcg_at_k(np.ones(effective_k, dtype=np.float32))
        ndcgs.append(dcg_at_k(hits) / ideal if ideal > 0.0 else 0.0)
    return {
        "recall_at_20": float(np.mean(recalls)) if recalls else float("nan"),
        "ndcg_at_20": float(np.mean(ndcgs)) if ndcgs else float("nan"),
    }


def data_summary_row(
    data: SyntheticData,
    splits: ObservedSplits,
    density: float,
    replicate: int,
    seed: int,
    config: SimulationConfig,
) -> dict[str, float | int | str]:
    scores = data.true_scores
    train_item_count = item_train_counts(splits.train, scores.shape[1])
    observed_n = len(splits.train) + len(splits.validation) + len(splits.test)
    test_cancers = set(splits.test[:, 0].tolist()) if len(splits.test) else set()
    return {
        "condition_id": data.condition_id,
        "dgp_mode": data.dgp_mode,
        "split_mode": splits.split_mode,
        "feature_setting": data.feature_setting,
        "density": density,
        "replicate": replicate,
        "seed": seed,
        "signal_strength": data.signal_strength,
        "U": scores.shape[0],
        "I": scores.shape[1],
        "d": config.feature_dim,
        "K": config.latent_dim,
        "n_observed": observed_n,
        "actual_density": observed_n / float(scores.shape[0] * scores.shape[1]),
        "n_train": len(splits.train),
        "n_val": len(splits.validation),
        "n_test": len(splits.test),
        "y_min": float(np.min(scores)),
        "y_q1": float(np.quantile(scores, 0.25)),
        "y_median": float(np.quantile(scores, 0.50)),
        "y_q3": float(np.quantile(scores, 0.75)),
        "y_max": float(np.max(scores)),
        "y_mean": float(np.mean(scores)),
        "y_sd": float(np.std(scores)),
        "X_density": float(data.cancer_features.mean()),
        "Z_density": float(data.intervention_features.mean()),
        "n_cancers_with_test_pairs": len(test_cancers),
        "n_items_in_train": int(np.sum(train_item_count > 0)),
        "n_items_in_val": int(len(set(splits.validation[:, 1].tolist()))) if len(splits.validation) else 0,
        "n_items_in_test": int(len(set(splits.test[:, 1].tolist()))) if len(splits.test) else 0,
        "n_global_cold_items": int(np.sum(train_item_count == 0)),
        "gamma_feature": data.gamma_feature,
        "beta_latent": data.beta_latent,
        "noise_sd": data.noise_sd,
        "n_cancers": scores.shape[0],
        "n_interventions": scores.shape[1],
        "feature_dim": config.feature_dim,
        "latent_dim": config.latent_dim,
        "score_mean": float(scores.mean()),
        "score_sd": float(scores.std()),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "latent_signal_mean": float(data.latent_signal.mean()),
        "latent_signal_sd": float(data.latent_signal.std()),
        "feature_signal_mean": float(data.feature_signal.mean()),
        "feature_signal_sd": float(data.feature_signal.std()),
        "cancer_feature_density": float(data.cancer_features.mean()),
        "intervention_feature_density": float(data.intervention_features.mean()),
    }


def feature_diagnostics_row(
    data: SyntheticData,
    splits: ObservedSplits,
    density: float,
    replicate: int,
    seed: int,
) -> dict[str, float | int | str]:
    train_items = set(splits.train[:, 1].tolist()) if len(splits.train) else set()
    validation_items = set(splits.validation[:, 1].tolist()) if len(splits.validation) else set()
    test_items = set(splits.test[:, 1].tolist()) if len(splits.test) else set()
    row: dict[str, float | int | str] = {
        "condition_id": data.condition_id,
        "dgp_mode": data.dgp_mode,
        "split_mode": splits.split_mode,
        "feature_setting": data.feature_setting,
        "density": density,
        "replicate": replicate,
        "seed": seed,
        "n_train_items": len(train_items),
        "n_validation_items": len(validation_items),
        "n_test_items": len(test_items),
        "n_validation_items_seen_in_train": len(validation_items & train_items),
        "n_test_items_seen_in_train": len(test_items & train_items),
        "n_validation_test_item_overlap": len(validation_items & test_items),
    }
    row.update(data.feature_diagnostics)
    return row


def oracle_scores_for_pairs(scores: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.array([], dtype=np.float32)
    return scores[pairs[:, 0], pairs[:, 1]].astype(np.float32)


def oracle_metrics_rows(
    data: SyntheticData,
    splits: ObservedSplits,
    density: float,
    replicate: int,
    seed: int,
) -> list[dict[str, float | int | str]]:
    y_test = scores_for_pairs(data, splits.test)
    rows = []
    oracle_predictions = {
        "Oracle_ID_only": oracle_scores_for_pairs(data.oracle_id_scores, splits.test),
        "Oracle_feature_aware": oracle_scores_for_pairs(data.oracle_feature_scores, splits.test),
    }
    oracle_metrics = {
        name: regression_metrics(y_test, prediction)
        for name, prediction in oracle_predictions.items()
    }
    id_rmse = oracle_metrics["Oracle_ID_only"]["rmse"]
    feature_rmse = oracle_metrics["Oracle_feature_aware"]["rmse"]
    for name, metrics in oracle_metrics.items():
        rows.append(
            {
                "condition_id": data.condition_id,
                "dgp_mode": data.dgp_mode,
                "split_mode": splits.split_mode,
                "feature_setting": data.feature_setting,
                "density": density,
                "replicate": replicate,
                "seed": seed,
                "oracle": name,
                "test_rmse": metrics["rmse"],
                "test_nrmse": metrics["nrmse"],
                "test_mae": metrics["mae"],
                "oracle_id_rmse_minus_feature_rmse": id_rmse - feature_rmse,
                "percent_oracle_feature_improvement": 100.0 * (id_rmse - feature_rmse) / id_rmse
                if id_rmse > 0
                else float("nan"),
            }
        )
    return rows


def run_single_task(
    task: tuple[int, ScenarioCondition, SimulationConfig],
) -> tuple[
    list[dict[str, float | int | str]],
    dict[str, float | int | str],
    list[dict[str, float | int | str]],
    dict[str, float | int | str],
    list[dict[str, float | int | str]],
]:
    replicate, condition, config = task
    seed = 3000 + replicate
    data = generate_synthetic_data(
        n_cancers=config.n_cancers,
        n_interventions=config.n_interventions,
        latent_dim=config.latent_dim,
        feature_dim=config.feature_dim,
        feature_setting=condition.feature_setting,
        condition_id=condition.condition_id,
        seed=seed,
        beta_latent=config.beta_latent if config.beta_latent is not None else condition.beta_latent,
        noise_sd=config.noise_sd if config.noise_sd is not None else condition.noise_sd,
        score_mean=config.score_mean,
        score_sd=config.score_sd,
        score_min=config.score_min,
        score_max=config.score_max,
        active_features_per_entity=config.active_features_per_entity,
        signal_strength=config.signal_strength,
        dgp_mode=condition.dgp_mode,
        gamma_feature_override=condition.gamma_feature,
        rho_feature=condition.rho_feature,
        eta_bias=condition.eta_bias,
        residual_item_bias_sd=condition.residual_item_bias_sd,
        cancer_bias_sd=condition.cancer_bias_sd,
        feature_share_target=condition.feature_share_target,
    )
    splits = sample_observed_splits(
        data,
        density=condition.density,
        seed=seed + int(round(condition.density * 10000)),
        popularity_biased=config.popularity_biased_sampling,
        split_mode=condition.split_mode,
    )

    rows: list[dict[str, float | int | str]] = []
    stratified_rows: list[dict[str, float | int | str]] = []
    for model_name in MODEL_NAMES:
        metrics, model_stratified = train_one_model(model_name, data, splits, config, seed)
        row = (
            {
                "replicate": replicate,
                "seed": seed,
                "condition_id": condition.condition_id,
                "dgp_mode": condition.dgp_mode,
                "feature_setting": condition.feature_setting,
                "signal_strength": config.signal_strength,
                "split_mode": splits.split_mode,
                "density": condition.density,
                "model": model_name,
                "n_cancers": config.n_cancers,
                "n_interventions": config.n_interventions,
                "feature_dim": config.feature_dim,
                "latent_dim": config.latent_dim,
                "train_n": len(splits.train),
                "validation_n": len(splits.validation),
                "test_n": len(splits.test),
                **metrics,
            }
        )
        rows.append(row)
        for stratified in model_stratified:
            stratified_rows.append(
                {
                    "split_mode": splits.split_mode,
                    "condition_id": condition.condition_id,
                    "dgp_mode": condition.dgp_mode,
                    "feature_setting": condition.feature_setting,
                    "density": condition.density,
                    "replicate": replicate,
                    "seed": seed,
                    "model": model_name,
                    **stratified,
                }
            )

    metrics_by_model = {row["model"]: row for row in rows}
    hybrid = metrics_by_model.get("Hybrid-NCF")
    if hybrid:
        mf = metrics_by_model.get("MF")
        ncf = metrics_by_model.get("NCF")
        hybrid_rmse = float(hybrid["test_rmse"])
        if mf:
            hybrid["mf_minus_hybrid_rmse"] = float(mf["test_rmse"]) - hybrid_rmse
        if ncf:
            ncf_rmse = float(ncf["test_rmse"])
            hybrid["ncf_minus_hybrid_rmse"] = ncf_rmse - hybrid_rmse
            hybrid["percent_improvement_vs_ncf"] = 100.0 * (ncf_rmse - hybrid_rmse) / ncf_rmse if ncf_rmse > 0 else float("nan")
    for row in rows:
        row.setdefault("mf_minus_hybrid_rmse", float("nan"))
        row.setdefault("ncf_minus_hybrid_rmse", float("nan"))
        row.setdefault("percent_improvement_vs_ncf", float("nan"))

    return (
        rows,
        data_summary_row(data, splits, condition.density, replicate, seed, config),
        stratified_rows,
        feature_diagnostics_row(data, splits, condition.density, replicate, seed),
        oracle_metrics_rows(data, splits, condition.density, replicate, seed),
    )


def timestamped_run_dir(output_root: Path) -> Path:
    return output_root / dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def config_to_serializable(config: SimulationConfig) -> dict:
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
        elif isinstance(value, tuple):
            payload[key] = list(value)
    payload["feature_gamma"] = FEATURE_GAMMAS_BY_SIGNAL
    payload["design_version"] = DESIGN_VERSION
    payload["seed_start"] = 3000
    payload["sampling"] = "popularity_biased" if config.popularity_biased_sampling else "uniform"
    payload["signal_strength"] = config.signal_strength
    payload["models"] = list(MODEL_NAMES)
    return payload


def git_metadata() -> dict[str, str | bool | list[str] | None]:
    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    status = run_git(["status", "--short"])
    status_lines = status.splitlines() if status else []
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "branch": run_git(["branch", "--show-current"]),
        "dirty": bool(status_lines),
        "status_short": status_lines,
    }


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ["numpy", "pandas", "torch", "PyYAML"]:
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def manifest_payload(
    config: SimulationConfig,
    conditions: Sequence[ScenarioCondition],
    argv: Sequence[str] | None = None,
) -> dict[str, object]:
    return {
        "design_version": DESIGN_VERSION,
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(argv) if argv else "",
        "argv": list(argv) if argv else [],
        "config": config_to_serializable(config),
        "conditions": [asdict(condition) for condition in conditions],
        "models": list(MODEL_NAMES),
        "model_seed_offsets": dict(MODEL_SEED_OFFSETS),
        "seed_start": 3000,
        "replicate_seed_formula": "3000 + replicate",
        "split_seed_formula": "replicate_seed + round(condition.density * 10000)",
        "environment": environment_payload(),
        "git": git_metadata(),
        "package_versions": package_versions(),
    }


def write_manifest(
    path: Path,
    config: SimulationConfig,
    conditions: Sequence[ScenarioCondition],
    argv: Sequence[str] | None = None,
) -> None:
    payload = manifest_payload(config, conditions, argv)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def config_from_manifest(manifest: dict[str, object]) -> SimulationConfig:
    config_payload = dict(manifest.get("config", {}))
    config_payload["output_root"] = Path(config_payload.get("output_root", "results/simulation_study"))
    run_dir_value = config_payload.get("run_dir")
    config_payload["run_dir"] = Path(run_dir_value) if run_dir_value else None
    config_payload["conditions"] = tuple(config_payload.get("conditions", CONDITION_IDS))
    known_fields = {field.name for field in SimulationConfig.__dataclass_fields__.values()}
    return SimulationConfig(**{key: value for key, value in config_payload.items() if key in known_fields})


def validate_manifest_for_current_design(manifest: dict[str, object]) -> None:
    if manifest.get("design_version") != DESIGN_VERSION:
        raise ValueError(
            f"Manifest design_version {manifest.get('design_version')!r} does not match {DESIGN_VERSION!r}."
        )
    config = config_from_manifest(manifest)
    if tuple(config.conditions) != CONDITION_IDS:
        raise ValueError("Manifest config must declare exactly the current five simulation conditions.")
    expected_conditions = [asdict(condition) for condition in simulation_conditions(config)]
    actual_conditions = manifest.get("conditions")
    if actual_conditions != expected_conditions:
        raise ValueError("Manifest conditions do not match the current expanded simulation design.")
    if manifest.get("models") != list(MODEL_NAMES):
        raise ValueError("Manifest model list does not match the current simulation models.")


def write_config(path: Path, config: SimulationConfig) -> None:
    payload = config_to_serializable(config)
    if yaml is not None:
        path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    else:
        lines = [f"{key}: {value}" for key, value in sorted(payload.items())]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_output_rows_for_current_design(
    frame: pd.DataFrame,
    name: str,
    require_models: bool = False,
    require_conditions: bool = False,
) -> None:
    if "condition_id" in frame.columns:
        actual_conditions = set(frame["condition_id"].dropna().astype(str))
        unexpected_conditions = actual_conditions - set(CONDITION_IDS)
        if unexpected_conditions:
            raise ValueError(f"{name} contains unknown condition_id values: {sorted(unexpected_conditions)}")
        if require_conditions and actual_conditions != set(CONDITION_IDS):
            raise ValueError(f"{name} condition_id values {sorted(actual_conditions)} do not match {list(CONDITION_IDS)}")
    if require_models and "model" in frame.columns:
        actual_models = set(frame["model"].dropna().astype(str))
        if actual_models != set(MODEL_NAMES):
            raise ValueError(f"{name} model values {sorted(actual_models)} do not match {list(MODEL_NAMES)}")


def metrics_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "validation_rmse",
        "test_rmse",
        "test_nrmse",
        "test_mae",
        "runtime_seconds",
        "best_epoch",
        "recall_at_20",
        "ndcg_at_20",
        "mf_minus_hybrid_rmse",
        "ncf_minus_hybrid_rmse",
        "percent_improvement_vs_ncf",
    ]
    rows = []
    group_columns = [
        column
        for column in ["condition_id", "dgp_mode", "split_mode", "feature_setting", "density", "model"]
        if column in metrics.columns
    ]
    for keys, group in metrics.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {column: value for column, value in zip(group_columns, keys)}
        row["n_replicates"] = int(group["replicate"].nunique())
        for column in metric_columns:
            if column in group:
                values = group[column].dropna()
                output_name = "improvement_vs_ncf" if column == "ncf_minus_hybrid_rmse" else column
                row[f"{output_name}_mean"] = float(values.mean()) if len(values) else float("nan")
                row[f"{output_name}_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
        row["interpretation"] = interpretation_for_summary_row(row)
        rows.append(row)
    return pd.DataFrame(rows)


def interpretation_for_summary_row(row: dict[str, float | int | str]) -> str:
    if row.get("model") != "Hybrid-NCF":
        return ""
    condition_id = str(row.get("condition_id", ""))
    split_mode = str(row.get("split_mode", ""))
    feature_setting = str(row.get("feature_setting", ""))
    improvement = row.get("improvement_vs_ncf_mean", float("nan"))
    try:
        improvement_value = float(improvement)
    except (TypeError, ValueError):
        improvement_value = float("nan")
    if condition_id == "linear_latent_warm":
        return "Low-rank linear warm-start setting; MF is expected to perform best or close to best."
    if condition_id == "nonlinear_latent_warm":
        return "Nonlinear warm-start setting; NCF is expected to outperform MF with enough data."
    if feature_setting == "permuted_features":
        return "Permuted-feature control; Hybrid advantage should be small or absent."
    if split_mode == "global_cold_item" and improvement_value > 0:
        return "Global cold-item stress test; globally unseen interventions use default ID embeddings."
    return ""


def stratified_metrics_summary(stratified: pd.DataFrame) -> pd.DataFrame:
    if stratified.empty:
        return pd.DataFrame(
            columns=[
                "condition_id",
                "dgp_mode",
                "split_mode",
                "feature_setting",
                "density",
                "model",
                "item_train_count_bin",
                "n_test_pairs",
                "test_rmse_mean",
                "test_rmse_se",
            ]
        )
    rows = []
    group_columns = [
        "condition_id",
        "dgp_mode",
        "split_mode",
        "feature_setting",
        "density",
        "model",
        "item_train_count_bin",
    ]
    for keys, group in stratified.groupby(group_columns, sort=True):
        values = group["test_rmse"].dropna()
        row = {column: value for column, value in zip(group_columns, keys)}
        row["n_test_pairs"] = int(group["n_test_pairs"].sum())
        row["test_rmse_mean"] = float(values.mean()) if len(values) else float("nan")
        row["test_rmse_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def format_float(value: float, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float) or isinstance(value, np.floating):
                cells.append(format_float(float(value), digits))
            else:
                cells.append(str(value))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *body])


def add_hybrid_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        column
        for column in ["condition_id", "dgp_mode", "split_mode", "feature_setting", "density"]
        if column in summary.columns
    ]
    for keys, group in summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        hybrid = group[group["model"] == "Hybrid-NCF"]
        if hybrid.empty:
            continue
        hybrid_rmse = float(hybrid.iloc[0]["test_rmse_mean"])
        for _, row in group.iterrows():
            if row["model"] == "Hybrid-NCF":
                continue
            baseline_rmse = float(row["test_rmse_mean"])
            delta_row = {column: value for column, value in zip(group_columns, keys)}
            delta_row.update(
                {
                    "comparison": f"Hybrid-NCF vs {row['model']}",
                    "rmse_delta": hybrid_rmse - baseline_rmse,
                    "relative_delta_pct": 100.0 * (hybrid_rmse - baseline_rmse) / baseline_rmse
                    if baseline_rmse > 0.0
                    else float("nan"),
                }
            )
            rows.append(delta_row)
    return pd.DataFrame(rows)


def write_report(
    run_dir: Path,
    config: SimulationConfig,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    data_summary: pd.DataFrame,
    stratified_summary: pd.DataFrame,
    diagnostics: pd.DataFrame | None = None,
    oracle_metrics: pd.DataFrame | None = None,
) -> None:
    diagnostics = diagnostics if diagnostics is not None else pd.DataFrame()
    oracle_metrics = oracle_metrics if oracle_metrics is not None else pd.DataFrame()
    deltas = add_hybrid_deltas(summary)
    compact = summary[
        [
            "condition_id",
            "dgp_mode",
            "split_mode",
            "feature_setting",
            "density",
            "model",
            "n_replicates",
            "test_rmse_mean",
            "test_rmse_se",
            "test_nrmse_mean",
            "test_nrmse_se",
            "test_mae_mean",
            "test_mae_se",
            "validation_rmse_mean",
            "validation_rmse_se",
            "improvement_vs_ncf_mean",
            "percent_improvement_vs_ncf_mean",
            "runtime_seconds_mean",
        ]
    ].copy()
    sort_columns = [
        column
        for column in ["condition_id", "split_mode", "density", "test_rmse_mean"]
        if column in compact.columns
    ]
    compact = compact.sort_values(sort_columns)
    compact_columns = [
        "condition_id",
        "dgp_mode",
        "split_mode",
        "feature_setting",
        "density",
        "model",
        "n_replicates",
        "test_rmse_mean",
        "test_rmse_se",
        "test_nrmse_mean",
        "test_nrmse_se",
        "test_mae_mean",
        "test_mae_se",
        "validation_rmse_mean",
        "validation_rmse_se",
        "improvement_vs_ncf_mean",
        "percent_improvement_vs_ncf_mean",
        "runtime_seconds_mean",
    ]
    delta_columns = [
        *[column for column in ["condition_id", "dgp_mode", "split_mode"] if column in deltas.columns],
        "feature_setting",
        "density",
        "comparison",
        "rmse_delta",
        "relative_delta_pct",
    ]
    data_group_columns = [
        column
        for column in ["condition_id", "dgp_mode", "split_mode", "feature_setting"]
        if column in data_summary.columns
    ]
    data_distribution = data_summary.groupby(data_group_columns, as_index=False).agg(
        score_mean=("y_mean", "mean"),
        score_sd=("y_sd", "mean"),
        score_min=("y_min", "mean"),
        score_max=("y_max", "mean"),
        gamma_feature=("gamma_feature", "mean"),
        beta_latent=("beta_latent", "mean"),
        noise_sd=("noise_sd", "mean"),
    )
    data_distribution_columns = [
        *[column for column in ["condition_id", "dgp_mode", "split_mode"] if column in data_distribution.columns],
        "feature_setting",
        "gamma_feature",
        "beta_latent",
        "noise_sd",
        "score_mean",
        "score_sd",
        "score_min",
        "score_max",
    ]
    diagnostics_columns = [
        "condition_id",
        "split_mode",
        "feature_setting",
        "density",
        "total_feature_explainable_variance_share",
        "var_latent_component",
        "var_feature_q_component",
        "var_feature_bias_component",
        "var_match_component",
        "var_noise",
        "corr_y_feature_match",
        "corr_y_Z_bias",
        "fraction_clipped_low",
        "fraction_clipped_high",
        "n_test_items_seen_in_train",
    ]
    diagnostics_summary = pd.DataFrame()
    if not diagnostics.empty:
        diagnostics_summary = diagnostics.groupby(
            ["condition_id", "split_mode", "feature_setting", "density"],
            as_index=False,
        ).agg(
            total_feature_explainable_variance_share=("total_feature_explainable_variance_share", "mean"),
            var_latent_component=("var_latent_component", "mean"),
            var_feature_q_component=("var_feature_q_component", "mean"),
            var_feature_bias_component=("var_feature_bias_component", "mean"),
            var_match_component=("var_match_component", "mean"),
            var_noise=("var_noise", "mean"),
            corr_y_feature_match=("corr_y_feature_match", "mean"),
            corr_y_Z_bias=("corr_y_Z_bias", "mean"),
            fraction_clipped_low=("fraction_clipped_low", "mean"),
            fraction_clipped_high=("fraction_clipped_high", "mean"),
            n_test_items_seen_in_train=("n_test_items_seen_in_train", "max"),
        )
    oracle_columns = [
        "condition_id",
        "split_mode",
        "feature_setting",
        "density",
        "oracle",
        "test_rmse_mean",
        "test_rmse_se",
        "oracle_id_rmse_minus_feature_rmse_mean",
        "percent_oracle_feature_improvement_mean",
    ]
    oracle_summary = pd.DataFrame()
    if not oracle_metrics.empty:
        oracle_rows = []
        group_columns = ["condition_id", "split_mode", "feature_setting", "density", "oracle"]
        for keys, group in oracle_metrics.groupby(group_columns, sort=True):
            values = group["test_rmse"].dropna()
            row = {column: value for column, value in zip(group_columns, keys)}
            row["test_rmse_mean"] = float(values.mean()) if len(values) else float("nan")
            row["test_rmse_se"] = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else float("nan")
            gap = group["oracle_id_rmse_minus_feature_rmse"].dropna()
            pct = group["percent_oracle_feature_improvement"].dropna()
            row["oracle_id_rmse_minus_feature_rmse_mean"] = float(gap.mean()) if len(gap) else float("nan")
            row["percent_oracle_feature_improvement_mean"] = float(pct.mean()) if len(pct) else float("nan")
            oracle_rows.append(row)
        oracle_summary = pd.DataFrame(oracle_rows)

    lines = [
        "# Simulation Study Report",
        "",
        f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Executive Summary",
        "",
        (
            "This is a controlled methodological benchmark, not clinical validation. "
            "It compares MF, NCF, and a lightweight feature-fusion Hybrid-NCF "
            "analogue under five manuscript-focused simulation conditions, including "
            "a strengthened global cold-item stress test."
        ),
        "",
        (
            "MF should perform well when the true structure is low-rank and "
            "linear. NCF should perform well when nonlinear ID-based interactions "
            "are present and enough warm-start observations are available. "
            "Hybrid-NCF should perform best when side features are informative "
            "under the global cold-item split. Hybrid-NCF "
            "should provide little or no improvement when side-feature alignment "
            "is broken."
        ),
        "",
        "Global cold-item is a stress test; it is intentionally hard for ID-only methods.",
        "",
        (
            "The strengthened global cold-item conditions were designed to represent "
            "a setting in which intervention target features explain a meaningful "
            "portion of the behavior of unseen interventions. This is intentionally "
            "favorable to Hybrid-NCF and complements the warm-start benchmarks. In "
            "the permuted-feature control, the Hybrid advantage should diminish, "
            "demonstrating that the gain depends on meaningful side-feature alignment."
        ),
        "",
        "## Methods",
        "",
        f"- Cancer types: {config.n_cancers}",
        f"- Interventions: {config.n_interventions}",
        f"- Latent dimension: {config.latent_dim}",
        f"- Side-feature dimension: {config.feature_dim}",
        f"- Conditions: {', '.join(config.conditions)}",
        f"- Replicates: {config.replicates}",
        f"- Sampling: {'popularity-biased' if config.popularity_biased_sampling else 'uniform'}",
        f"- Ranking metrics: {'enabled' if config.compute_ranking else 'disabled'}",
        "",
        "The Hybrid-NCF model here is a lightweight analogue of the paper model. "
        "It concatenates cancer ID embeddings, intervention ID embeddings, cancer "
        "side features, and intervention side features, and includes an explicit "
        "side-feature interaction head over elementwise cancer/intervention feature "
        "crosses. It does not implement the full pretrain/fine-tune production architecture.",
        "",
        "For the global cold-item split, unseen item ID embeddings and item "
        "biases are set to default zero values before validation-based model "
        "selection and final evaluation for all models. This avoids letting "
        "random untrained item embeddings determine predictions.",
        "",
        "### Controlled Conditions",
        "",
        "- `linear_latent_warm`: warm-start, density 0.10, low-rank linear latent DGP, uninformative side features. MF should perform best or close to best.",
        "- `nonlinear_latent_warm`: warm-start, density 0.10, nonlinear latent DGP, uninformative side features. NCF should outperform MF when enough warm-start data are available.",
        "- `global_cold_informative`: global cold-item, density 0.05, informative side features. Intervention latent factors and item bias are partly generated from `Z_i`, plus an `X_u^T Z_i` match term.",
        "- `global_cold_moderately_informative`: global cold-item, density 0.05, weaker feature signal than `global_cold_informative`.",
        "- `global_cold_permuted_features`: global cold-item, density 0.05, outcomes are generated from true informative features but observed side features are permuted before training and evaluation.",
        "",
        "The original feature-informed global cold-item gain was modest because ID-only models could still use the global mean and cancer bias, while the cold-item behavior was only weakly side-feature-explainable. In global cold-start, Hybrid can only help when the behavior of unseen interventions is explainable by `Z_i` and by cancer-feature/intervention-feature compatibility. The revised DGP makes this condition explicit by generating part of `q_i` and `b_i` from `Z_i`.",
        "",
        "## RMSE Summary",
        "",
        markdown_table(
            compact,
            compact_columns,
        ),
        "",
        "## Hybrid-NCF Deltas",
        "",
        "Negative RMSE deltas indicate lower RMSE for Hybrid-NCF than the baseline.",
        "",
        markdown_table(
            deltas,
            delta_columns,
        ),
        "",
        "## Simulated Score Distribution",
        "",
        markdown_table(
            data_distribution,
            data_distribution_columns,
        ),
        "",
        "## Feature Signal Diagnostics",
        "",
        markdown_table(
            diagnostics_summary,
            [column for column in diagnostics_columns if column in diagnostics_summary.columns],
        ),
        "",
        "## Oracle Metrics",
        "",
        "The oracle gap shows the maximum possible benefit from side-feature-explainable cold-item behavior under the synthetic DGP. In the permuted-feature control, the feature-aware oracle uses the true latent DGP features and is not the performance of Hybrid-NCF using the permuted observed features.",
        "",
        markdown_table(
            oracle_summary,
            [column for column in oracle_columns if column in oracle_summary.columns],
        ),
        "",
        "## Item Train-Count Stratified RMSE",
        "",
        markdown_table(
            stratified_summary.head(80),
            [
                "condition_id",
                "split_mode",
                "dgp_mode",
                "feature_setting",
                "density",
                "model",
                "item_train_count_bin",
                "n_test_pairs",
                "test_rmse_mean",
                "test_rmse_se",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "MF performs well when the true structure is low-rank and linear. "
            "NCF performs well when nonlinear ID-based interactions are present "
            "and enough warm-start data are available. Hybrid-NCF performs best "
            "when side features are informative under the global cold-item split. "
            "Hybrid-NCF provides little or no improvement when side "
            "features are permuted and feature alignment is broken."
        ),
        "",
        (
            "These simulations should not be interpreted as clinical validation. "
            "They only test whether the algorithms recover known synthetic signal "
            "under controlled assumptions. Hybrid-NCF depends on informative side "
            "features and is not guaranteed to outperform ID-only methods when "
            "features are noisy, irrelevant, or misspecified."
        ),
        "",
        "In warm-start linear latent data, MF is expected to be competitive. "
        "In warm-start nonlinear latent data, NCF is expected to improve over MF. "
        "In global cold-item data with informative side features, ID-only methods "
        "must use default item embeddings and biases for unseen interventions, "
        "while Hybrid-NCF can still use intervention side features. The "
        "permuted-feature control generates outcomes from informative true "
        "features but supplies misaligned features to Hybrid, testing whether "
        "improvement depends on meaningful feature alignment.",
        "",
        "## Limitations",
        "",
        "- Synthetic data are not clinical validation.",
        "- Cancer-level features are coarse and are not patient-level genomics.",
        "- Global cold-item evaluation is a stress test and is intentionally hard for ID-only methods.",
        "- The lightweight Hybrid-NCF simulation model may differ from the full production Hybrid-NCF.",
        "- The explicit interaction head matches the synthetic DGP and should be interpreted as feature-aware recommender behavior, not proof that the production architecture is universally superior.",
        "- The strengthened global cold-item conditions are Hybrid-favorable by design, not a claim that Hybrid always dominates.",
        "",
        "## Manuscript-Ready Paragraph",
        "",
        (
            "We added a controlled methodological benchmark comparing MF, NCF, "
            "and Hybrid-NCF across five synthetic simulation conditions. MF was evaluated in "
            "a low-rank linear warm-start setting, NCF in a nonlinear warm-start "
            "setting, and Hybrid-NCF in a strengthened global cold-item setting "
            "where intervention target features explain a meaningful portion of "
            "unseen intervention behavior. We also included moderately informative "
            "and permuted-feature global cold-item controls. This "
            "design demonstrates that Hybrid-NCF is not universally superior: it "
            "is expected to help most when side features are meaningful and "
            "interventions are evaluated under global cold-item, while MF and NCF remain strong "
            "under their corresponding low-rank linear and nonlinear warm-start "
            "data-generating mechanisms."
        ),
        "",
        "## Rebuttal-Ready Paragraph",
        "",
        (
            "Following the reviewer's suggestion, we revised the simulation "
            "benchmark to separate three mechanisms: low-rank linear structure, "
            "nonlinear ID-based structure, and informative side-feature structure "
            "under global intervention cold-start. We do not use target-cancer "
            "cold-start in this benchmark. MF is expected to perform well in the "
            "linear warm-start condition, NCF in the nonlinear warm-start condition, "
            "and Hybrid-NCF in the strengthened informative global cold-item condition. "
            "The strengthened DGP generates part of the intervention latent factor "
            "and item bias from target features and includes an explicit cancer-feature "
            "by intervention-feature match term. In the permuted-feature control, "
            "Hybrid-NCF should show little or no advantage, demonstrating the limitation "
            "of the hybrid architecture when observed side information is misaligned."
        ),
        "",
        "## Output Files",
        "",
        "- `simulation_config.yaml`",
        "- `simulation_manifest.json`",
        "- `simulation_metrics_by_replicate.csv`",
        "- `simulation_metrics_summary.csv`",
        "- `simulation_feature_signal_diagnostics.csv`",
        "- `simulation_oracle_metrics.csv`",
        "- `simulation_item_count_stratified_metrics.csv`",
        "- `simulated_data_summary.csv`",
        "- `simulation_report.md`",
        "",
    ]
    (run_dir / "simulation_report.md").write_text("\n".join(lines), encoding="utf-8")


def aggregate_outputs(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_path = run_dir / "simulation_metrics_by_replicate.csv"
    data_summary_path = run_dir / "simulated_data_summary.csv"
    stratified_path = run_dir / "simulation_item_count_stratified_metrics.csv"
    diagnostics_path = run_dir / "simulation_feature_signal_diagnostics.csv"
    oracle_path = run_dir / "simulation_oracle_metrics.csv"
    manifest_path = run_dir / "simulation_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest file: {manifest_path}; rerun the simulation with current code.")
    validate_manifest_for_current_design(load_manifest(manifest_path))
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    if not data_summary_path.exists():
        raise FileNotFoundError(f"Missing data summary file: {data_summary_path}")
    metrics = pd.read_csv(metrics_path)
    data_summary = pd.read_csv(data_summary_path)
    validate_output_rows_for_current_design(metrics, metrics_path.name, require_models=True, require_conditions=True)
    validate_output_rows_for_current_design(data_summary, data_summary_path.name, require_conditions=True)
    summary = metrics_summary(metrics)
    summary.to_csv(run_dir / "simulation_metrics_summary.csv", index=False)
    if stratified_path.exists():
        stratified_summary = pd.read_csv(stratified_path)
        validate_output_rows_for_current_design(
            stratified_summary,
            stratified_path.name,
            require_models="model" in stratified_summary.columns,
            require_conditions=True,
        )
    else:
        stratified_summary = pd.DataFrame()
    if "test_rmse" in stratified_summary.columns:
        stratified_summary = stratified_metrics_summary(stratified_summary)
        stratified_summary.to_csv(stratified_path, index=False)
    diagnostics = pd.read_csv(diagnostics_path) if diagnostics_path.exists() else pd.DataFrame()
    validate_output_rows_for_current_design(diagnostics, diagnostics_path.name, require_conditions=True)
    oracle_metrics = pd.read_csv(oracle_path) if oracle_path.exists() else pd.DataFrame()
    validate_output_rows_for_current_design(oracle_metrics, oracle_path.name, require_conditions=True)

    config_path = run_dir / "simulation_config.yaml"
    config = None
    if yaml is not None and config_path.exists():
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload:
            payload = dict(payload)
            payload["output_root"] = Path(payload.get("output_root", "results/simulation_study"))
            run_dir_value = payload.get("run_dir")
            payload["run_dir"] = Path(run_dir_value) if run_dir_value else None
            payload["conditions"] = tuple(payload.get("conditions", CONDITION_IDS))
            known_fields = {field.name for field in SimulationConfig.__dataclass_fields__.values()}
            payload = {key: value for key, value in payload.items() if key in known_fields}
            config = SimulationConfig(**payload)
    if config is None:
        config = SimulationConfig(run_dir=run_dir)
    write_report(
        run_dir,
        config,
        metrics,
        summary,
        data_summary,
        stratified_summary,
        diagnostics,
        oracle_metrics,
    )
    return metrics, summary, data_summary, stratified_summary, diagnostics, oracle_metrics


def environment_payload() -> dict[str, str | int | None]:
    return {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
    }


def prepare_run_dir(config: SimulationConfig) -> Path:
    output_root = Path(config.output_root)
    run_dir = Path(config.run_dir) if config.run_dir else timestamped_run_dir(output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        "simulation_config.yaml",
        "simulation_manifest.json",
        "simulation_metrics_by_replicate.csv",
        "simulation_metrics_summary.csv",
        "simulation_item_count_stratified_metrics.csv",
        "simulation_feature_signal_diagnostics.csv",
        "simulation_oracle_metrics.csv",
        "simulated_data_summary.csv",
        "simulation_report.md",
    ]
    existing = [run_dir / name for name in required_outputs if (run_dir / name).exists()]
    if existing and not config.force:
        existing_names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"{run_dir} already contains outputs ({existing_names}); use --force to overwrite.")
    return run_dir


def run_simulation_study(config: SimulationConfig, argv: Sequence[str] | None = None) -> Path:
    run_dir = prepare_run_dir(config)
    conditions = simulation_conditions(config)
    write_config(run_dir / "simulation_config.yaml", config)
    write_manifest(run_dir / "simulation_manifest.json", config, conditions, argv)
    tasks = [
        (replicate, condition, config)
        for replicate in range(config.replicates)
        for condition in conditions
    ]
    started = time.perf_counter()
    metric_rows: list[dict[str, float | int | str]] = []
    stratified_rows: list[dict[str, float | int | str]] = []
    diagnostic_rows: list[dict[str, float | int | str]] = []
    oracle_rows: list[dict[str, float | int | str]] = []
    data_summary_by_key: dict[tuple[int, str, str, str, float], dict[str, float | int | str]] = {}

    if config.max_parallel_jobs > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=config.max_parallel_jobs) as executor:
            futures = {executor.submit(run_single_task, task): task for task in tasks}
            for future in as_completed(futures):
                rows, data_row, task_stratified_rows, diagnostic_row, task_oracle_rows = future.result()
                metric_rows.extend(rows)
                stratified_rows.extend(task_stratified_rows)
                diagnostic_rows.append(diagnostic_row)
                oracle_rows.extend(task_oracle_rows)
                data_summary_by_key[
                    (
                        int(data_row["replicate"]),
                        str(data_row["condition_id"]),
                        str(data_row["split_mode"]),
                        str(data_row["feature_setting"]),
                        float(data_row["density"]),
                    )
                ] = data_row
    else:
        for task in tasks:
            rows, data_row, task_stratified_rows, diagnostic_row, task_oracle_rows = run_single_task(task)
            metric_rows.extend(rows)
            stratified_rows.extend(task_stratified_rows)
            diagnostic_rows.append(diagnostic_row)
            oracle_rows.extend(task_oracle_rows)
            data_summary_by_key[
                (
                    int(data_row["replicate"]),
                    str(data_row["condition_id"]),
                    str(data_row["split_mode"]),
                    str(data_row["feature_setting"]),
                    float(data_row["density"]),
                )
            ] = data_row

    metrics = pd.DataFrame(metric_rows).sort_values(
        ["condition_id", "split_mode", "density", "replicate", "model"],
        ignore_index=True,
    )
    data_summary = pd.DataFrame(data_summary_by_key.values()).sort_values(
        ["condition_id", "split_mode", "density", "replicate"],
        ignore_index=True,
    )
    stratified = pd.DataFrame(stratified_rows)
    diagnostics = pd.DataFrame(diagnostic_rows).sort_values(
        ["condition_id", "split_mode", "feature_setting", "density", "replicate"],
        ignore_index=True,
    )
    oracle_metrics = pd.DataFrame(oracle_rows).sort_values(
        ["condition_id", "split_mode", "feature_setting", "density", "replicate", "oracle"],
        ignore_index=True,
    )
    metrics["study_runtime_seconds"] = time.perf_counter() - started
    metrics.to_csv(run_dir / "simulation_metrics_by_replicate.csv", index=False)
    data_summary.to_csv(run_dir / "simulated_data_summary.csv", index=False)
    diagnostics.to_csv(run_dir / "simulation_feature_signal_diagnostics.csv", index=False)
    oracle_metrics.to_csv(run_dir / "simulation_oracle_metrics.csv", index=False)
    summary = metrics_summary(metrics)
    summary.to_csv(run_dir / "simulation_metrics_summary.csv", index=False)
    stratified_summary = stratified_metrics_summary(stratified)
    stratified_summary.to_csv(run_dir / "simulation_item_count_stratified_metrics.csv", index=False)
    write_report(run_dir, config, metrics, summary, data_summary, stratified_summary, diagnostics, oracle_metrics)

    return run_dir


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_parallel_jobs(value: str) -> int:
    if str(value).lower() == "auto":
        return max(1, min(4, os.cpu_count() or 1))
    return positive_int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cancers", type=positive_int, default=55)
    parser.add_argument("--n-interventions", type=positive_int, default=3000)
    parser.add_argument("--feature-dim", type=positive_int, default=100)
    parser.add_argument("--latent-dim", type=positive_int, default=16)
    parser.add_argument("--active-features-per-entity", type=positive_int, default=10)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITION_IDS,
        default=list(CONDITION_IDS),
    )
    parser.add_argument("--replicates", type=positive_int, default=10)
    parser.add_argument("--output-root", type=Path, default=Path("results/simulation_study"))
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--max-parallel-jobs", type=parse_parallel_jobs, default=1)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-epochs", type=positive_int, default=100)
    parser.add_argument("--patience", type=positive_int, default=10)
    parser.add_argument("--batch-size", type=positive_int, default=1024)
    parser.add_argument("--embedding-dim", type=positive_int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--beta-latent", type=float, default=None)
    parser.add_argument("--noise-sd", type=float, default=None)
    parser.add_argument("--signal-strength", choices=tuple(SIGNAL_DEFAULTS), default="standard")
    parser.add_argument("--compute-ranking", action="store_true")
    parser.add_argument("--ranking-k", type=positive_int, default=20)
    parser.add_argument("--uniform-sampling", action="store_true")
    parser.add_argument("--torch-threads", type=positive_int, default=1)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    n_cancers = args.n_cancers
    n_interventions = args.n_interventions
    feature_dim = args.feature_dim
    latent_dim = args.latent_dim
    replicates = args.replicates
    max_epochs = args.max_epochs
    patience = args.patience
    batch_size = args.batch_size
    if args.smoke_test:
        n_cancers = min(n_cancers, 20)
        n_interventions = min(n_interventions, 300)
        feature_dim = min(feature_dim, 50)
        latent_dim = min(latent_dim, 8)
        replicates = min(replicates, 1)
        max_epochs = min(max_epochs, 3)
        patience = min(patience, 1)
        batch_size = min(batch_size, 128)

    return SimulationConfig(
        n_cancers=n_cancers,
        n_interventions=n_interventions,
        feature_dim=feature_dim,
        latent_dim=latent_dim,
        active_features_per_entity=args.active_features_per_entity,
        conditions=tuple(args.conditions),
        replicates=replicates,
        output_root=args.output_root,
        max_parallel_jobs=args.max_parallel_jobs,
        smoke_test=args.smoke_test,
        force=args.force,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        beta_latent=args.beta_latent,
        noise_sd=args.noise_sd,
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        compute_ranking=args.compute_ranking,
        ranking_k=args.ranking_k,
        popularity_biased_sampling=not args.uniform_sampling,
        torch_threads=args.torch_threads,
        signal_strength=args.signal_strength,
        run_dir=args.run_dir,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_from_args(args)
    run_dir = run_simulation_study(config, argv=sys.argv if argv is None else [sys.argv[0], *argv])
    print(f"Simulation study outputs written to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
