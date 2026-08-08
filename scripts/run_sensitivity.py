#!/usr/bin/env python3
"""Run resumable Hybrid-NCF scoring sensitivity jobs and aggregate the results.

This single entry point fans out per-variant, per-seed Hybrid-NCF retraining
jobs and then aggregates the per-job outputs into the final sensitivity tables
and report. It consolidates the previous ``run_sensitivity_overnight.py`` and
``aggregate_sensitivity_results.py`` scripts:

* Running with no ``--aggregate-only`` flag launches the jobs and, once they all
  finish, aggregates the outputs in-process.
* Running with ``--aggregate-only --run-dir <dir>`` skips retraining and only
  (re)builds the aggregate tables and report for an existing run directory.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import sys
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

REPO_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_FOR_IMPORTS))

from scripts.sensitivity_analysis import (
    build_scoring_variants,
    clinical_signal_rows_component,
    clinical_signal_rows_exact,
    normalized_intervention_key,
    overlap_at_k,
)


DEFAULT_VARIANTS = [
    "baseline",
    "no_year_adjustment",
    "phase_nccn_neutral",
    "no_status_outcome_adjustment",
]

NUMERIC_SUMMARY_COLUMNS = [
    "validation_rmse",
    "test_rmse",
    "validation_nrmse",
    "test_nrmse",
    "delta_test_rmse_vs_baseline_same_seed",
    "delta_test_nrmse_vs_baseline_same_seed",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path("results/sensitivity_overnight"))
    parser.add_argument("--run-dir", type=Path, default=None, help="Resume an existing run directory instead of creating a new timestamped one. Required with --aggregate-only.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2000, 2001, 2002])
    parser.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    parser.add_argument("--max-parallel-jobs", default="auto")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run only the first variant/seed with tiny epochs and a row sample.")
    parser.add_argument("--limit-rows", type=int, default=0)
    parser.add_argument("--smoke-rows", type=int, default=600)
    parser.add_argument("--pretrain-epochs", type=int, default=30)
    parser.add_argument("--pretrain-patience", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=20)
    parser.add_argument("--finetune-patience", type=int, default=3)
    parser.add_argument("--finetune-lr-scale", type=float, default=0.1)
    parser.add_argument("--no-save-models", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Skip retraining and only (re)aggregate an existing --run-dir.")
    return parser.parse_args(argv)


def resolve_parallel_jobs(value: str) -> int:
    if value != "auto":
        return max(1, int(value))
    cpu_count = os.cpu_count() or 1
    has_cuda = False
    try:
        import torch

        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_cuda = False
    if has_cuda:
        return 1
    return max(1, min(2, cpu_count // 4 if cpu_count >= 4 else 1))


def atomic_write_json(path: Path, payload: Dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: Dict):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")


def write_yaml_config(path: Path, variants: Sequence[str]):
    configs = build_scoring_variants()
    lines = ["variants:"]
    for variant in variants:
        cfg = asdict(configs[variant])
        lines.append(f"  {variant}:")
        for key, value in cfg.items():
            if isinstance(value, dict):
                lines.append(f"    {key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"      {sub_key}: {sub_value}")
            else:
                lines.append(f"    {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def package_versions() -> Dict[str, str]:
    names = ["numpy", "pandas", "scikit-learn", "torch", "surprise", "pyarrow"]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def environment_payload(repo_root: Path, max_parallel_jobs: int) -> Dict:
    git_hash = ""
    try:
        git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
    except Exception:
        git_hash = "unknown"
    torch_info = {"available": False}
    try:
        import torch

        torch_info = {
            "available": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
            "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        torch_info = {"available": False, "error": str(exc)}
    return {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "git_commit": git_hash,
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "max_parallel_jobs": max_parallel_jobs,
        "thread_limits": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", ""),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", ""),
        },
        "packages": package_versions(),
        "torch": torch_info,
    }


def make_jobs(args: argparse.Namespace, run_dir: Path) -> List[Dict]:
    variants = args.variants
    seeds = args.seeds
    if args.smoke:
        variants = [variants[0]]
        seeds = [seeds[0]]
    jobs = []
    for seed in seeds:
        for variant in variants:
            job_id = f"{variant}__seed{seed}"
            job_dir = run_dir / "jobs" / job_id
            jobs.append(
                {
                    "job_id": job_id,
                    "variant": variant,
                    "seed": seed,
                    "job_dir": str(job_dir),
                    "status": "pending",
                    "returncode": None,
                    "started_at": "",
                    "finished_at": "",
                    "runtime_seconds": None,
                    "log_file": str(run_dir / "logs" / f"{job_id}.log"),
                }
            )
    return jobs


def command_for_job(args: argparse.Namespace, job: Dict) -> List[str]:
    pretrain_epochs = 1 if args.smoke else args.pretrain_epochs
    pretrain_patience = 1 if args.smoke else args.pretrain_patience
    finetune_epochs = 1 if args.smoke else args.finetune_epochs
    finetune_patience = 1 if args.smoke else args.finetune_patience
    limit_rows = args.smoke_rows if args.smoke else args.limit_rows
    cmd = [
        sys.executable,
        str(args.repo_root / "scripts" / "sensitivity_analysis.py"),
        "--repo-root",
        str(args.repo_root),
        "--output-dir",
        job["job_dir"],
        "--seed",
        str(job["seed"]),
        "--variants",
        job["variant"],
        "--pretrain-epochs",
        str(pretrain_epochs),
        "--pretrain-patience",
        str(pretrain_patience),
        "--finetune-epochs",
        str(finetune_epochs),
        "--finetune-patience",
        str(finetune_patience),
        "--finetune-lr-scale",
        str(args.finetune_lr_scale),
    ]
    if limit_rows:
        cmd.extend(["--limit-rows", str(limit_rows)])
    if not args.no_save_models:
        cmd.append("--save-models")
    return cmd


def write_job_tables(run_dir: Path, jobs: Sequence[Dict]):
    table_specs = [
        ({"completed", "skipped"}, "completed_jobs.csv"),
        ({"failed"}, "failed_jobs.csv"),
    ]
    for statuses, filename in table_specs:
        rows = [job for job in jobs if job["status"] in statuses]
        path = run_dir / filename
        if rows:
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("job_id,variant,seed,status,returncode,started_at,finished_at,runtime_seconds,log_file,job_dir\n", encoding="utf-8")


def update_progress(run_dir: Path, jobs: Sequence[Dict], start_time: float):
    counts = {name: sum(1 for job in jobs if job["status"] == name) for name in ["pending", "running", "completed", "failed", "skipped"]}
    finished = counts["completed"] + counts["failed"] + counts["skipped"]
    elapsed = time.time() - start_time
    avg_finished = elapsed / finished if finished else None
    pendingish = counts["pending"] + counts["running"]
    eta = avg_finished * pendingish if avg_finished else None
    payload = {
        "run_dir": str(run_dir),
        "start_time": dt.datetime.fromtimestamp(start_time).isoformat(timespec="seconds"),
        "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "counts": counts,
        "jobs": list(jobs),
    }
    atomic_write_json(run_dir / "progress.json", payload)
    write_job_tables(run_dir, jobs)


def write_heartbeat(run_dir: Path):
    (run_dir / "heartbeat.txt").write_text(dt.datetime.now().isoformat(timespec="seconds") + "\n", encoding="utf-8")


def job_complete(job: Dict) -> bool:
    return (Path(job["job_dir"]) / "hybrid_sensitivity_metrics.csv").exists()


def prepare_run(args: argparse.Namespace, max_parallel_jobs: int) -> Path:
    repo_root = args.repo_root.resolve()
    args.repo_root = repo_root
    if args.run_dir:
        run_dir = args.run_dir.resolve()
    else:
        output_root = args.output_root if args.output_root.is_absolute() else repo_root / args.output_root
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = output_root / timestamp
    for subdir in ["configs", "jobs", "logs"]:
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    write_yaml_config(run_dir / "configs" / "scoring_variants_config.yaml", args.variants)
    atomic_write_json(run_dir / "environment.json", environment_payload(repo_root, max_parallel_jobs))
    raw_check = {}
    exploded_check = {}
    try:
        import pandas as pd

        from scripts.sensitivity_analysis import compute_scores, prepare_modeling_data

        cancer_dir = repo_root / "cancer_rec_system"
        raw = pd.read_csv(repo_root / "private_data" / "summary_w_score.csv")
        baseline = build_scoring_variants()["baseline"]
        raw_scores = compute_scores(raw, baseline)
        raw_diff = (raw_scores - raw["Score"].astype(float)).abs()
        raw_check = {
            "dataset": "raw_summary_w_score",
            "n": int(raw_diff.shape[0]),
            "max_abs_diff": float(raw_diff.max()),
            "mismatches": int((raw_diff > 1e-8).sum()),
        }
        prepared = prepare_modeling_data(cancer_dir)
        prepared_scores = compute_scores(prepared, baseline)
        prepared_diff = (prepared_scores - prepared["Score"].astype(float)).abs()
        exploded_check = {
            "dataset": "exploded_filtered_modeling_data",
            "n": int(prepared_diff.shape[0]),
            "max_abs_diff": float(prepared_diff.max()),
            "mismatches": int((prepared_diff > 1e-8).sum()),
        }
    except Exception as exc:
        raw_check = {"dataset": "raw_summary_w_score", "error": str(exc)}
        exploded_check = {"dataset": "exploded_filtered_modeling_data", "error": str(exc)}
    baseline_check_path = run_dir / "baseline_score_recompute_check.csv"
    with baseline_check_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(set(raw_check) | set(exploded_check)))
        writer.writeheader()
        writer.writerow(raw_check)
        writer.writerow(exploded_check)
    (run_dir / "baseline_reproduction_report.md").write_text(
        "\n".join(
            [
                "# Baseline Reproduction Report",
                "",
                "The current repository contains the Hybrid-NCF production checkpoint and training code, but no saved manuscript training logs. Current inspection indicates the manuscript RMSE values were produced under a cross-validation protocol, while this overnight run uses fixed-split retraining for sensitivity unless the CV path is explicitly rerun.",
                "",
                "Known reconciliation points:",
                "- Baseline score recomputation matches the existing score column in prior diagnostics.",
                "- The production Hybrid-NCF checkpoint evaluated on all observed rows has RMSE about 0.496.",
                "- Current MF/NCF artifacts do not fully reproduce manuscript values from the current code/artifact state.",
            ]
        ),
        encoding="utf-8",
    )
    return run_dir


def read_csvs(job_dirs: Sequence[Path], filename: str) -> pd.DataFrame:
    frames = []
    for job_dir in job_dirs:
        path = job_dir / filename
        if path.exists():
            try:
                frames.append(pd.read_csv(path))
            except EmptyDataError:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_csv(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)


def recompute_metric_deltas(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    metrics = metrics.copy()
    metrics = metrics.rename(
        columns={
            "delta_test_rmse_vs_baseline": "delta_test_rmse_vs_baseline_same_seed",
            "delta_test_nrmse_vs_baseline": "delta_test_nrmse_vs_baseline_same_seed",
        }
    )
    for seed, idx in metrics.groupby("seed").groups.items():
        seed_df = metrics.loc[idx]
        baseline = seed_df[seed_df["variant"] == "baseline"]
        if baseline.empty:
            continue
        baseline_rmse = float(baseline["test_rmse"].iloc[0])
        baseline_nrmse = float(baseline["test_nrmse"].iloc[0])
        metrics.loc[idx, "delta_test_rmse_vs_baseline_same_seed"] = (
            metrics.loc[idx, "test_rmse"].astype(float) - baseline_rmse
        )
        metrics.loc[idx, "delta_test_nrmse_vs_baseline_same_seed"] = (
            metrics.loc[idx, "test_nrmse"].astype(float) - baseline_nrmse
        )
    return metrics


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for variant, group in metrics.groupby("variant", sort=False):
        row = {"variant": variant, "n_seeds": int(group["seed"].nunique())}
        for col in NUMERIC_SUMMARY_COLUMNS:
            if col in group:
                values = pd.to_numeric(group[col], errors="coerce")
                summary_col = col.removesuffix("_same_seed")
                row[f"{summary_col}_mean"] = float(values.mean())
                row[f"{summary_col}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def compute_overlap(top20: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if top20.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows = []
    for (seed, cancer), cancer_df in top20.groupby(["seed", "target_cancer"]):
        base = cancer_df[cancer_df["variant"] == "baseline"].sort_values("rank")
        if base.empty:
            continue
        base_items = base["intervention"].tolist()
        for variant, variant_df in cancer_df.groupby("variant", sort=False):
            variant_items = variant_df.sort_values("rank")["intervention"].tolist()
            rows.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "cancer": cancer,
                    "overlap_at_10": overlap_at_k(base_items, variant_items, 10),
                    "overlap_at_20": overlap_at_k(base_items, variant_items, 20),
                }
            )
    by_cancer = pd.DataFrame(rows)
    if by_cancer.empty:
        return by_cancer, pd.DataFrame(), pd.DataFrame()
    seed_rows = []
    for (variant, seed), group in by_cancer.groupby(["variant", "seed"], sort=False):
        nsclc = group[group["cancer"] == "non-small cell lung cancer"]
        melanoma = group[group["cancer"] == "melanoma: cutaneous"]
        seed_rows.append(
            {
                "variant": variant,
                "seed": seed,
                "avg_overlap_at_10_all_cancers": group["overlap_at_10"].mean(),
                "avg_overlap_at_20_all_cancers": group["overlap_at_20"].mean(),
                "nsclc_overlap_at_10": nsclc["overlap_at_10"].iloc[0] if not nsclc.empty else np.nan,
                "nsclc_overlap_at_20": nsclc["overlap_at_20"].iloc[0] if not nsclc.empty else np.nan,
                "melanoma_overlap_at_10": melanoma["overlap_at_10"].iloc[0] if not melanoma.empty else np.nan,
                "melanoma_overlap_at_20": melanoma["overlap_at_20"].iloc[0] if not melanoma.empty else np.nan,
            }
        )
    by_seed = pd.DataFrame(seed_rows)
    across_rows = []
    for variant, group in by_seed.groupby("variant", sort=False):
        row = {"variant": variant, "n_seeds": int(group["seed"].nunique())}
        for col in [
            "avg_overlap_at_10_all_cancers",
            "avg_overlap_at_20_all_cancers",
            "nsclc_overlap_at_10",
            "nsclc_overlap_at_20",
            "melanoma_overlap_at_10",
            "melanoma_overlap_at_20",
        ]:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_sd"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
        across_rows.append(row)
    return by_cancer, by_seed, pd.DataFrame(across_rows)


def compute_signal_retention(top100: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_rows: List[Dict[str, object]] = []
    component_rows: List[Dict[str, object]] = []
    if top100.empty:
        return pd.DataFrame(), pd.DataFrame()
    for seed, seed_df in top100.groupby("seed"):
        variants = list(seed_df["variant"].drop_duplicates())
        lookup = {}
        for (variant, cancer), group in seed_df.groupby(["variant", "target_cancer"]):
            lookup[(variant, cancer)] = group.sort_values("rank").reset_index(drop=True)
        exact = pd.DataFrame(clinical_signal_rows_exact(lookup, variants))
        component = pd.DataFrame(clinical_signal_rows_component(lookup, variants))
        if not exact.empty:
            exact["seed"] = seed
            exact_rows.extend(exact.to_dict("records"))
        if not component.empty:
            component["seed"] = seed
            component_rows.extend(component.to_dict("records"))
    return pd.DataFrame(exact_rows), pd.DataFrame(component_rows)


def make_side_by_side(run_dir: Path, top20: pd.DataFrame, target_cancer: str) -> pd.DataFrame:
    rows = []
    if not top20.empty:
        baseline = top20[
            (top20["variant"] == "baseline") & (top20["target_cancer"] == target_cancer)
        ].copy()
        if not baseline.empty:
            first_seed = sorted(baseline["seed"].unique())[0]
            baseline = baseline[baseline["seed"] == first_seed].sort_values("rank").head(20)
            for row in baseline.to_dict("records"):
                rows.append(
                    {
                        "target_cancer": target_cancer,
                        "source_run_type": "fixed_split_sensitivity",
                        "rank": row.get("rank"),
                        "intervention": row.get("intervention"),
                        "predicted_score": row.get("predicted_score"),
                        "source_cancer": row.get("source_cancer"),
                        "source_id": row.get("source_id"),
                        "source_id_type": row.get("source_id_type"),
                    }
                )
    environment_path = run_dir / "environment.json"
    repo_root = run_dir.parents[2] if len(run_dir.parents) >= 3 else run_dir
    if environment_path.exists():
        try:
            repo_root = Path(json.loads(environment_path.read_text(encoding="utf-8")).get("repo_root", repo_root))
        except json.JSONDecodeError:
            pass
    production_path = repo_root / "cancer_rec_system" / "data" / "recommendations" / "cancer-ncf-pretrain_recommendations.csv"
    if production_path.exists():
        try:
            production = pd.read_csv(production_path).head(20)
            for rank, row in enumerate(production.to_dict("records"), 1):
                rows.append(
                    {
                        "target_cancer": target_cancer,
                        "source_run_type": "manuscript_production",
                        "rank": row.get("rank", rank),
                        "intervention": row.get("intervention", row.get("Intervention", "")),
                        "predicted_score": row.get("predicted_score", row.get("Model score", "")),
                        "source_cancer": row.get("source_cancer", row.get("Cancer", "")),
                        "source_id": row.get("source_id", row.get("NCT Number", "")),
                        "source_id_type": row.get("source_id_type", ""),
                    }
                )
        except Exception:
            pass
    return pd.DataFrame(rows)


def aggregate(run_dir: Path) -> Dict[str, object]:
    run_dir = run_dir.resolve()
    job_dirs = sorted((run_dir / "jobs").glob("*")) if (run_dir / "jobs").exists() else []
    metrics = recompute_metric_deltas(read_csvs(job_dirs, "hybrid_sensitivity_metrics.csv"))
    score_dist = read_csvs(job_dirs, "score_distribution_by_variant.csv")
    methods = read_csvs(job_dirs, "rmse_methods_baseline.csv")
    top20 = read_csvs(job_dirs, "recommendation_top20_by_variant.csv")
    top100 = read_csvs(job_dirs, "recommendation_top100_by_variant.csv")
    if not top20.empty and "normalized_intervention_key" not in top20:
        top20["normalized_intervention_key"] = top20["intervention"].map(normalized_intervention_key)
    if not top100.empty and "normalized_intervention_key" not in top100:
        top100["normalized_intervention_key"] = top100["intervention"].map(normalized_intervention_key)

    baseline_checks = []
    existing_baseline_check = run_dir / "baseline_score_recompute_check.csv"
    if existing_baseline_check.exists():
        try:
            existing_df = pd.read_csv(existing_baseline_check)
            if "job" in existing_df.columns:
                existing_df = existing_df[existing_df["job"].isna()]
            baseline_checks.extend(existing_df.to_dict("records"))
        except Exception:
            pass
    for job_dir in job_dirs:
        path = job_dir / "baseline_score_check.json"
        if path.exists():
            payload = json.loads(path.read_text())
            payload["dataset"] = "job_exploded_filtered_modeling_data"
            payload["job"] = job_dir.name
            baseline_checks.append(payload)
    baseline_check_df = pd.DataFrame(baseline_checks)
    if not baseline_check_df.empty:
        dedupe_cols = [col for col in ["dataset", "job"] if col in baseline_check_df.columns]
        if dedupe_cols:
            baseline_check_df = baseline_check_df.drop_duplicates(subset=dedupe_cols, keep="last")

    by_cancer, overlap_by_seed, overlap_across = compute_overlap(top20)
    exact, component = compute_signal_retention(top100)
    summary = summarize_metrics(metrics)

    write_csv(score_dist, run_dir / "score_distribution_by_variant.csv")
    write_csv(baseline_check_df, run_dir / "baseline_score_recompute_check.csv")
    write_csv(methods, run_dir / "rmse_methods_baseline.csv")
    write_csv(metrics, run_dir / "hybrid_sensitivity_metrics_by_seed.csv")
    write_csv(summary, run_dir / "hybrid_sensitivity_metrics_summary.csv")
    write_csv(top20, run_dir / "recommendation_top20_by_variant.csv")
    write_csv(top100, run_dir / "recommendation_top100_by_variant.csv")
    write_csv(by_cancer, run_dir / "baseline_variant_overlap_by_cancer.csv")
    write_csv(overlap_by_seed, run_dir / "overlap_summary_by_seed.csv")
    write_csv(overlap_across, run_dir / "overlap_summary_across_seeds.csv")
    write_csv(exact, run_dir / "clinical_signal_retention_exact.csv")
    write_csv(component, run_dir / "clinical_signal_retention_component.csv")
    write_csv(make_side_by_side(run_dir, top20, "non-small cell lung cancer"), run_dir / "nsclc_top20_side_by_side.csv")
    write_csv(make_side_by_side(run_dir, top20, "melanoma: cutaneous"), run_dir / "melanoma_top20_side_by_side.csv")

    write_report(run_dir, metrics, summary, overlap_across, exact, component, baseline_check_df)
    return {
        "run_dir": str(run_dir),
        "jobs_seen": len(job_dirs),
        "metrics_rows": len(metrics),
        "top20_rows": len(top20),
        "top100_rows": len(top100),
    }


def write_report(
    run_dir: Path,
    metrics: pd.DataFrame,
    summary: pd.DataFrame,
    overlap: pd.DataFrame,
    exact: pd.DataFrame,
    component: pd.DataFrame,
    baseline_checks: pd.DataFrame,
):
    baseline_ok = False
    if not baseline_checks.empty and "mismatches" in baseline_checks:
        baseline_ok = pd.to_numeric(baseline_checks["mismatches"], errors="coerce").fillna(1).eq(0).all()
    baseline_metric = metrics[metrics["variant"] == "baseline"] if not metrics.empty else pd.DataFrame()
    baseline_rmse_note = "Baseline fixed-split RMSE was not available."
    if not baseline_metric.empty:
        baseline_val = pd.to_numeric(baseline_metric["validation_rmse"], errors="coerce").mean()
        baseline_test = pd.to_numeric(baseline_metric["test_rmse"], errors="coerce").mean()
        baseline_rmse_note = (
            f"Across the three fixed-split retraining seeds, baseline Hybrid-NCF mean validation RMSE was "
            f"{baseline_val:.3f} and mean test RMSE was {baseline_test:.3f}; this does not reproduce the "
            "manuscript-reported Hybrid-NCF validation RMSE of about 0.433."
        )
    variant_table = pd.DataFrame(
        [
            {
                "variant": "baseline",
                "rule change": "Original phase, status/outcome, year, and NCCN scoring rule.",
                "analysis role": "reference",
            },
            {
                "variant": "mild_status_outcome_penalties",
                "rule change": "Bad status and non-promising penalties reduced to 0.5; unknown outcome penalty reduced to 0.25.",
                "analysis role": "main sensitivity",
            },
            {
                "variant": "strong_status_outcome_penalties",
                "rule change": "Bad status and non-promising penalties increased to 1.5; unknown outcome penalty increased to 0.75.",
                "analysis role": "main sensitivity",
            },
            {
                "variant": "no_year_adjustment",
                "rule change": "Temporal penalty removed; phase, status/outcome, and NCCN rules unchanged.",
                "analysis role": "main sensitivity",
            },
            {
                "variant": "phase_nccn_neutral",
                "rule change": "Phase and NCCN score information removed by assigning I/II/III/IV trials and NCCN rows the same 2.5 score; status/outcome and year rules unchanged.",
                "analysis role": "stress-test ablation",
            },
            {
                "variant": "no_status_outcome_adjustment",
                "rule change": "Status/outcome penalties removed; phase, year, and NCCN rules unchanged.",
                "analysis role": "stress-test ablation",
            },
        ]
    )
    lines = [
        "# Scoring Sensitivity Analysis",
        "",
        "## Executive Summary",
        "",
        f"- Baseline score recomputation matched existing scores: {baseline_ok}.",
        f"- {baseline_rmse_note}",
        "- This run retrains Hybrid-NCF for each scoring variant and seed using a fixed split.",
        "- Manuscript CV reproduction remains documented separately; these results should be described as fixed-split diagnostic sensitivity unless the manuscript CV protocol is rerun.",
        "",
        "## Methods Paragraph",
        "",
        (
            "To assess sensitivity to heuristic choices in the scoring function, we recomputed "
            "cancer-treatment scores under alternative scoring schemes that perturbed trial "
            "status/outcome penalties, removed the temporal adjustment, reduced the NCCN-derived "
            "score, and compressed phase-based baseline weights. For each variant, we retrained "
            "Hybrid-NCF using the same data split, architecture, hyperparameters, and candidate "
            "filtering rule as in the baseline analysis. We evaluated robustness using RMSE, "
            "normalized RMSE, baseline-variant Top-K overlap, and retention of clinically "
            "interpretable treatment signals in the melanoma and NSCLC case studies."
        ),
        "",
        "## Baseline Reproduction",
        "",
        "Scripts used: `scripts/run_sensitivity.py` launched per-job calls to `scripts/sensitivity_analysis.py` and produced the final tables in-process. The Hybrid-NCF architecture, mutation side features, saved user/item mappings, and baseline hyperparameters were loaded from `cancer_rec_system/params/cancer-ncf-pretrain/`. Each seed used a deterministic 70/15/15 row split generated after the same preprocessing and filtering.",
        "",
        "The current repository does not contain saved training logs for the manuscript RMSE values. Prior inspection found that current MF/NCF artifacts do not fully match current model code, while the Hybrid production checkpoint evaluated all observed rows at RMSE about 0.496. The retraining sensitivity analysis therefore uses fixed splits and reports raw RMSE plus nRMSE rather than claiming exact reproduction of the manuscript CV RMSE.",
        "",
        "## Scoring Variants",
        "",
        variant_table.to_markdown(index=False),
        "",
        "## Prediction Robustness",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No completed metric rows.",
        "",
        "## Recommendation Stability",
        "",
        overlap.to_markdown(index=False) if not overlap.empty else "No overlap rows.",
        "",
        "## Clinical Signal Retention: Exact",
        "",
        exact.to_markdown(index=False) if not exact.empty else "No exact signal rows.",
        "",
        "## Clinical Signal Retention: Component",
        "",
        component.to_markdown(index=False) if not component.empty else "No component signal rows.",
        "",
        "## Interpretation",
        "",
        "Ranking stability and score prediction stability should be interpreted separately. A variant can have similar RMSE but different Top-N rankings, especially when guideline-derived NCCN rows or phase compression alter the high-score tail. Hybrid-NCF uses an unconstrained regression output, so predicted scores are ranking scores rather than literal clinical-trial phase numbers.",
        "",
        "## Rebuttal-Ready Paragraph",
        "",
        "We added a scoring sensitivity analysis in which cancer-treatment scores were recomputed under alternative clinically motivated scoring rules and Hybrid-NCF was retrained for each variant using identical splits, architecture, and hyperparameters. Across variants we report RMSE, normalized RMSE to account for changed score distributions, Top-K ranking overlap with the baseline model, and recovery of NSCLC and melanoma treatment signals. This analysis directly addresses whether the main conclusions depend on a single heuristic scoring choice.",
        "",
        "## Limitations",
        "",
        "- Fixed-split sensitivity is not identical to manuscript cross-validation unless the manuscript CV path is explicitly rerun.",
        "- Seed variability is summarized, but additional seeds may still change tail rankings.",
        "- Exact-match signal tracking is conservative for multi-drug treatment classes; component-level tracking is included for clinical interpretability.",
        "- Internal NCCN IDs are not real ClinicalTrials.gov NCT identifiers and should be reported separately.",
    ]
    (run_dir / "sensitivity_report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    max_parallel_jobs = resolve_parallel_jobs(str(args.max_parallel_jobs))
    run_dir = prepare_run(args, max_parallel_jobs)
    jobs = make_jobs(args, run_dir)
    for job in jobs:
        Path(job["job_dir"]).mkdir(parents=True, exist_ok=True)
        if job_complete(job) and not args.force:
            job["status"] = "skipped"
            job["returncode"] = 0
            append_jsonl(run_dir / "progress.jsonl", {"event": "job_skipped", "job_id": job["job_id"], "time": dt.datetime.now().isoformat(timespec="seconds")})

    start_time = time.time()
    running: Dict[str, Dict] = {}
    update_progress(run_dir, jobs, start_time)
    write_heartbeat(run_dir)
    append_jsonl(run_dir / "progress.jsonl", {"event": "run_started", "run_dir": str(run_dir), "time": dt.datetime.now().isoformat(timespec="seconds")})

    interrupted = False
    try:
        while True:
            for job in jobs:
                if len(running) >= max_parallel_jobs:
                    break
                if job["status"] != "pending":
                    continue
                cmd = command_for_job(args, job)
                log_path = Path(job["log_file"])
                log_handle = log_path.open("w", encoding="utf-8")
                env = os.environ.copy()
                for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
                    env.setdefault(key, "1")
                proc = subprocess.Popen(cmd, cwd=args.repo_root, stdout=log_handle, stderr=subprocess.STDOUT, env=env)
                job["status"] = "running"
                job["started_at"] = dt.datetime.now().isoformat(timespec="seconds")
                job["_start_time"] = time.time()
                running[job["job_id"]] = {"process": proc, "log_handle": log_handle, "job": job}
                append_jsonl(run_dir / "progress.jsonl", {"event": "job_started", "job_id": job["job_id"], "cmd": cmd, "time": job["started_at"]})
                update_progress(run_dir, jobs, start_time)

            finished_ids = []
            for job_id, payload in running.items():
                proc = payload["process"]
                ret = proc.poll()
                if ret is None:
                    continue
                payload["log_handle"].close()
                job = payload["job"]
                job["returncode"] = ret
                job["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
                job["runtime_seconds"] = time.time() - float(job.get("_start_time", time.time()))
                job.pop("_start_time", None)
                job["status"] = "completed" if ret == 0 and job_complete(job) else "failed"
                (Path(job["job_dir"]) / "job_status.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
                append_jsonl(run_dir / "progress.jsonl", {"event": f"job_{job['status']}", "job_id": job_id, "returncode": ret, "time": job["finished_at"]})
                finished_ids.append(job_id)
                update_progress(run_dir, jobs, start_time)
            for job_id in finished_ids:
                del running[job_id]

            write_heartbeat(run_dir)
            if all(job["status"] in {"completed", "failed", "skipped"} for job in jobs):
                break
            time.sleep(5)
    except KeyboardInterrupt:
        interrupted = True
        append_jsonl(run_dir / "progress.jsonl", {"event": "keyboard_interrupt", "time": dt.datetime.now().isoformat(timespec="seconds")})
        for payload in running.values():
            payload["process"].terminate()
            payload["log_handle"].close()
        for job in jobs:
            if job["status"] == "running":
                job["status"] = "failed"
                job["finished_at"] = dt.datetime.now().isoformat(timespec="seconds")
        update_progress(run_dir, jobs, start_time)

    if not interrupted:
        try:
            agg_result = aggregate(run_dir)
            agg_ret = 0
        except Exception as exc:  # pragma: no cover - aggregation guard
            agg_result = {"error": str(exc)}
            agg_ret = 1
        append_jsonl(run_dir / "progress.jsonl", {"event": "aggregation_finished", "returncode": agg_ret, "result": agg_result, "time": dt.datetime.now().isoformat(timespec="seconds")})
    update_progress(run_dir, jobs, start_time)
    print(f"RUN_DIR={run_dir}")
    return 130 if interrupted else (1 if any(job["status"] == "failed" for job in jobs) else 0)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.aggregate_only:
        if args.run_dir is None:
            raise SystemExit("--aggregate-only requires --run-dir")
        result = aggregate(args.run_dir.resolve())
        print(json.dumps(result, indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
