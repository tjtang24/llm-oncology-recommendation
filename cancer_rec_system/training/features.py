"""Pure data-preparation and mutation-feature functions for model training.

The functions in this module perform no file I/O at import time. The mutation
map is a project feature-engineering input, not clinical guidance.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_CANCER_MUTATIONS: dict[str, list[str]] = {
    "Acute Lymphoblastic Leukemia": ["BCR-ABL1", "JAK2"],
    "Acute Myeloid Leukemia": ["FLT3", "IDH", "IDH1", "IDH2", "BCL-2"],
    "basal cell skin cancer": ["Hh", "EGFR", "PD-1"],
    "squamous cell skin cancer": ["Hh", "EGFR", "PD-1"],
    "bile duct cancer": [
        "FGFR2",
        "IDH1",
        "NTRK",
        "RET",
        "BRAF",
        "MEK",
        "KRAS",
        "HER2",
    ],
    "bladder cancer": [
        "FGFR",
        "Nectin-4",
        "PD-1",
        "PD-L1",
        "FGFR2",
        "FGFR3",
    ],
    "bone cancer": ["IDH1", "IDH2", "RANKL"],
    "Central Nervous System Cancer": [
        "VEGF",
        "IDH",
        "IDH1",
        "IDH2",
        "MTOR",
        "BRAF",
        "NTRK",
    ],
    "breast cancer": ["HER2", "CDK4", "CDK6", "PI3K", "AKT1", "PIK3CA", "PARP"],
    "cervical cancer": ["RET", "NTRK", "VEGF", "PD-1", "PD-L1", "TROP2"],
    "Chronic Lymphocytic Leukemia/Small Lymphocytic Lymphoma": [
        "BTK",
        "BCL-2",
        "PI3K",
    ],
    "Chronic Myeloid Leukemia": [
        "tyrosine kinase inhibitor",
        "BCR-ABL1",
        "ABL-TKI",
        "STAMP",
    ],
    "colorectal cancer": ["VEGF", "EGFR", "BRAF", "HER2", "NTRK", "RET", "KRAS"],
    "Esophageal and Esophagogastric Junction Cancers": [
        "HER2",
        "VEGFR",
        "CLDN18.2",
        "NTRK",
        "VEGF",
        "PD-1",
        "PD-L1",
    ],
    "biliary tract": [
        "HER2",
        "FGFR",
        "FGFR2",
        "IDH1",
        "NTRK",
        "RET",
        "BRAF",
        "KRAS",
        "PD-1",
        "PD-L1",
    ],
    "Gastrointestinal Stromal Tumors": ["PDGFRA", "KIT", "PD-1", "PD-L1"],
    "kidney cancer": ["tyrosine kinases", "HIF", "MTOR", "VEGF", "VEGFR"],
    "head and neck cancer": ["EGFR", "PD-1", "PD-L1", "HER2", "HRAS"],
    "Hepatocellular Cancer": [
        "multi-kinase inhibitor",
        "VEGF",
        "VEGFR",
        "FGFR",
        "PD-1",
        "PD-L1",
    ],
    "melanoma: skin": ["BRAF", "NRAS", "NF1", "KIT"],
    "mesothelioma: pleural": ["VEGF", "PD-1", "CTLA-4"],
    "mesothelioma: peritoneal": ["VEGF", "PD-1", "CTLA-4"],
    "neuroblastoma": ["ALK", "GD2"],
    "Non-Small Cell Lung Cancer": [
        "VEGF",
        "KRAS",
        "EGFR",
        "ALK",
        "ROS1",
        "BRAF",
        "RET",
        "MET",
        "HER2",
        "TRK",
    ],
    "Ovarian Cancer/Fallopian Tube Cancer/Primary Peritoneal Cancer": [
        "VEGF",
        "PARP",
        "NTRK",
        "BRAF",
        "HER2",
        "RET",
        "KRAS",
    ],
    "pancreatic cancer": ["BRAF", "NTRK", "RET", "KRAS", "EGFR", "PARP", "HER2"],
    "Neuroendocrine and Adrenal Tumor": ["MTOR", "HIF", "VEGFR"],
    "prostate cancer": ["PARP"],
    "soft tissue sarcoma": [],
    # The VGEF spelling is retained to preserve the historical feature contract.
    # It should be reviewed scientifically before training a future model.
    "gastric cancer": ["HER2", "VGEF", "NTRK", "CLDN18.2"],
    "thyroid cancer": ["kinase inhibitor", "RET", "NTRK", "BRAF", "MEK", "VEGFR"],
    "uterine cancer": [
        "Kinase inhibitor",
        "NTRK",
        "PARP",
        "PD-1",
        "PD-L1",
        "VEGFR",
        "HER2",
        "MTOR",
    ],
    "vaginal cancer": ["RET", "NTRK", "PD-L1", "EGFR", "PI3K", "MTOR"],
    "Waldenström Macroglobulinemia/Lymphoplasmacytic Lymphoma": [
        "BTK",
        "Proteasome",
        "MTOR",
    ],
}

REPLACED_RAS_GENES = {"nras", "kras", "ras", "hras"}
REQUIRED_SUMMARY_COLUMNS = {
    "Cancer",
    "Intervention",
    "Score",
    "Phases",
    "Year",
}
REQUIRED_TARGET_COLUMNS = {"drug", "mutation"}


def canonicalize_intervention(value: Any) -> str:
    """Sort and normalize a comma-separated intervention combination."""
    drugs = [drug.strip() for drug in str(value).split(",") if drug.strip()]
    return ", ".join(sorted(drugs))


def normalized_mutation_map(
    source: Mapping[str, Sequence[str]] = RAW_CANCER_MUTATIONS,
) -> dict[str, list[str]]:
    """Return the legacy project mutation map with normalized cancer labels."""
    result: dict[str, list[str]] = {}
    for cancer, genes in source.items():
        normalized_genes: list[str] = []
        for gene in genes:
            normalized = "MEK" if gene.lower() in REPLACED_RAS_GENES else gene
            if normalized and normalized not in normalized_genes:
                normalized_genes.append(normalized)
        result[cancer.strip().lower()] = normalized_genes
    return result


DEFAULT_CANCER_MUTATIONS = normalized_mutation_map()


def require_columns(
    frame: pd.DataFrame,
    required: set[str],
    label: str,
) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_source_frames(
    summary_path: str | Path,
    targeted_therapy_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read explicitly supplied local training sources."""
    summary = pd.read_csv(Path(summary_path))
    targeted = pd.read_csv(Path(targeted_therapy_path))
    require_columns(summary, REQUIRED_SUMMARY_COLUMNS, "summary data")
    require_columns(targeted, REQUIRED_TARGET_COLUMNS, "targeted-therapy data")
    return summary, targeted


def prepare_summary_frames(
    summary: pd.DataFrame,
    min_interactions_per_cancer: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create unfiltered metadata rows and legacy-compatible interactions.

    The threshold counts source interaction rows, not distinct interventions.
    This preserves the ID contract used by the packaged production model.
    """
    if min_interactions_per_cancer < 1:
        raise ValueError("min_interactions_per_cancer must be at least 1.")
    require_columns(summary, REQUIRED_SUMMARY_COLUMNS, "summary data")

    metadata = summary.dropna(subset=["Cancer", "Intervention"]).copy()
    metadata["Intervention"] = metadata["Intervention"].map(
        canonicalize_intervention
    )
    metadata["Cancer"] = metadata["Cancer"].astype(str).str.split(",")
    metadata = metadata.explode("Cancer", ignore_index=True)
    metadata["Cancer"] = metadata["Cancer"].str.strip().str.lower()

    interactions = metadata.copy()
    interactions["Score"] = pd.to_numeric(interactions["Score"], errors="coerce")
    interactions["Score"] = interactions["Score"].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    interactions = interactions.dropna(
        subset=["Cancer", "Intervention", "Score"]
    ).reset_index(drop=True)

    treatment_counts = interactions.groupby("Cancer")["Intervention"].count()
    retained_cancers = treatment_counts[
        treatment_counts >= min_interactions_per_cancer
    ].index
    interactions = interactions[
        interactions["Cancer"].isin(retained_cancers)
    ].reset_index(drop=True)
    if interactions.empty:
        raise ValueError("No interactions remain after cancer thresholding.")

    user_mapping = {
        cancer: user_id
        for user_id, cancer in enumerate(interactions["Cancer"].unique())
    }
    item_mapping = {
        intervention: item_id
        for item_id, intervention in enumerate(
            interactions["Intervention"].unique()
        )
    }
    interactions["user_id"] = interactions["Cancer"].map(user_mapping)
    interactions["item_id"] = interactions["Intervention"].map(item_mapping)
    return metadata, interactions


def build_mappings(
    interactions: pd.DataFrame,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build deterministic, contiguous mappings in first-observation order."""
    user_mapping = {
        cancer: user_id
        for user_id, cancer in enumerate(interactions["Cancer"].unique())
    }
    item_mapping = {
        intervention: item_id
        for item_id, intervention in enumerate(
            interactions["Intervention"].unique()
        )
    }
    return user_mapping, item_mapping


def build_feature_frames(
    interactions: pd.DataFrame,
    targeted: pd.DataFrame,
    cancer_mutations: Mapping[str, Sequence[str]] = DEFAULT_CANCER_MUTATIONS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build aligned cancer and individual-drug binary mutation features."""
    require_columns(targeted, REQUIRED_TARGET_COLUMNS, "targeted-therapy data")

    cancers = sorted(interactions["Cancer"].dropna().astype(str).unique())
    drugs = list(
        dict.fromkeys(
            drug.strip()
            for intervention in interactions["Intervention"].astype(str)
            for drug in intervention.split(",")
            if drug.strip()
        )
    )

    targeted_rows = targeted.dropna(subset=["drug", "mutation"]).copy()
    grouped_targets = (
        targeted_rows.groupby("drug", sort=False)["mutation"].apply(list).to_dict()
    )

    cancer_gene_set = {
        gene
        for cancer in cancers
        for gene in cancer_mutations.get(cancer, ())
        if gene
    }
    drug_gene_set = {
        str(gene)
        for drug in drugs
        for gene in grouped_targets.get(drug, ())
        if str(gene)
    }
    feature_names = sorted(cancer_gene_set & drug_gene_set)
    if not feature_names:
        raise ValueError(
            "Cancer and targeted-therapy sources have no shared mutation features."
        )

    feature_index = {name: index for index, name in enumerate(feature_names)}
    cancer_array = np.zeros((len(cancers), len(feature_names)), dtype=np.float32)
    for row, cancer in enumerate(cancers):
        for gene in cancer_mutations.get(cancer, ()):
            if gene in feature_index:
                cancer_array[row, feature_index[gene]] = 1.0

    drug_array = np.zeros((len(drugs), len(feature_names)), dtype=np.float32)
    for row, drug in enumerate(drugs):
        for gene in grouped_targets.get(drug, ()):
            name = str(gene)
            if name in feature_index:
                drug_array[row, feature_index[name]] = 1.0

    cancer_vectors = pd.DataFrame(
        cancer_array,
        index=cancers,
        columns=feature_names,
    )
    drug_vectors = pd.DataFrame(
        drug_array,
        index=drugs,
        columns=feature_names,
    )
    return cancer_vectors, drug_vectors, feature_names


def build_intervention_vectors(
    item_mapping: Mapping[str, int],
    drug_vectors: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate component-drug vectors into combination-item vectors."""
    ordered_items = mapping_names_by_id(item_mapping)
    vectors = np.zeros(
        (len(ordered_items), drug_vectors.shape[1]),
        dtype=np.float32,
    )
    for item_id, intervention in enumerate(ordered_items):
        drugs = [
            drug.strip()
            for drug in intervention.split(",")
            if drug.strip()
        ]
        component_vectors = drug_vectors.reindex(drugs).fillna(0.0).to_numpy()
        if len(component_vectors):
            vectors[item_id] = np.clip(component_vectors.sum(axis=0), 0.0, 1.0)
    return pd.DataFrame(
        vectors,
        index=ordered_items,
        columns=drug_vectors.columns,
    )


def mapping_names_by_id(mapping: Mapping[str, int]) -> list[str]:
    """Validate a mapping and return names in integer-ID order."""
    if not all(isinstance(name, str) for name in mapping):
        raise ValueError("Mapping names must be strings.")
    if not all(
        isinstance(entity_id, int) and not isinstance(entity_id, bool)
        for entity_id in mapping.values()
    ):
        raise ValueError("Mapping IDs must be integers.")
    expected = set(range(len(mapping)))
    actual = set(mapping.values())
    if actual != expected:
        raise ValueError("Mapping IDs must be unique and contiguous from zero.")
    inverse = {entity_id: name for name, entity_id in mapping.items()}
    return [inverse[entity_id] for entity_id in range(len(mapping))]


def build_known_items(
    interactions: pd.DataFrame,
    user_mapping: Mapping[str, int],
) -> dict[str, list[str]]:
    """Return sorted observed interventions for every mapped cancer."""
    known: dict[str, list[str]] = {}
    for cancer, user_id in user_mapping.items():
        values = interactions.loc[
            interactions["Cancer"] == cancer,
            "Intervention",
        ]
        known[str(user_id)] = sorted(set(values.astype(str)))
    return known


def json_scalar(value: Any) -> str | int | float | bool | None:
    """Convert NumPy/Pandas scalars into strict JSON values."""
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def build_metadata_lookup(
    metadata: pd.DataFrame,
) -> dict[str, dict[str, str | int | float | bool | None]]:
    """Keep only the three metadata fields displayed by the Shiny app."""
    result: dict[str, dict[str, str | int | float | bool | None]] = {}
    for row in metadata.itertuples(index=False):
        intervention = str(getattr(row, "Intervention"))
        if intervention in result:
            continue
        result[intervention] = {
            "Cancer": str(getattr(row, "Cancer")),
            "Phases": str(getattr(row, "Phases")),
            "Year": json_scalar(getattr(row, "Year")),
        }
    return result
