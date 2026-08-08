# Oncology Evidence Recommendation System

[Live Shiny demo](https://tjtang.shinyapps.io/recommendation_system/)

This repository contains the **code-only reproduction materials** for the
recommendation-system component of the conference submission. The upstream LLM pipeline, prompts, API calls, and LLM
outputs are intentionally outside this code-release scope.

> **Disclaimer:** This code supports the STAI-X 2026 conference submission.
> It is not a medical device and must not be used for clinical decision-making.
> Model scores are uncalibrated ranking outputs—not treatment efficacy, approval
> status, or estimates of clinical benefit.

In this project users represent cancer types and items represent interventions.
The Hybrid-NCF model combines neural collaborative filtering with cancer and
drug-mutation side features and ranks clinical-trial interventions that were not
observed for a selected cancer type in the training data.

## Repository layout

| Path | Purpose |
| --- | --- |
| `cancer_rec_system/model.py` | Hybrid NCF model definition |
| `cancer_rec_system/training/` | Data preparation, mutation features, maintained Hybrid-NCF training|
| `cancer_rec_system/reproducibility/` | Code-only fixed-configuration MF and ID-only NCF refits, historical CV runners |
| `scripts/run_sensitivity.py` | Scoring-rule sensitivity analysis reproducing supplementary appendix Table 2 |
| `scripts/run_simulation_study.py` | Synthetic simulation benchmark reproducing supplementary appendix Section 4 (Table 3) |

## Reproducibility

- **Baselines refit exactly.** Given the private snapshots
  recorded by SHA-256 in the data manifests, the MF and ID-only NCF baselines
  refit bit-for-bit in the pinned Python 3.9 CPU environment: two independent
  refits matched the models, all 721,600 catalog scores, and the ordered Top-20
  for every cancer.

## Installation

The maintained training path and the Hybrid-NCF reproduction runner target
Python 3.12.13:

```bash
python -m pip install -r requirements.txt
```

The historical MF and ID-only NCF exact-refit audit uses a separate, pinned
Python 3.9.18 CPU environment (pandas 1.5.3 and a historical PyTorch nightly).
Its packages and the exact PyTorch artifact are recorded in
`cancer_rec_system/requirements-reproducibility.txt` and
`cancer_rec_system/reproducibility/environment-macos-arm64.yml`. Do not mix the
two environments.

## Train the recommendation model

Place `summary_w_score.csv`
and `targeted_therapy.csv` in the repo-root `private_data/` directory (the
default input location), then run from the repository root:

```bash
python -m pip install -r cancer_rec_system/requirements.txt
python -m cancer_rec_system.training.train_hybrid_ncf
```

## Reproduce the MF, NCF, and Hybrid-NCF baselines

The manuscript MF and ID-only pretrained NCF baselines have a separate,
historically pinned reproduction path. It records the successful Python 3.9 CPU
audit environment, exact fixed model configurations, private input hashes, and
reference hashes for model parameters, all 721,600 catalog scores, and the
ordered Top-20 for all 55 cancers.

The published refit runners rebuild the frozen configurations directly;
historical model selection and validation RMSE are separate audit questions. Two
independent fixed refits of both models were bitwise exact in the recorded
environment. The source-faithful historical CV runners ship in-tree: `cv_ncf.py`
reproduces the reported ID-only NCF mean RMSE 0.494728 (`0.4947`) and
`cv_hybrid.py` reproduces the reported Hybrid-NCF mean RMSE 0.464292 (`0.4643`),
each verifiable against a frozen per-fold reference with `verify_cv.py`. Both
carry a single global tuner across folds. 

See
[`cancer_rec_system/reproducibility/README.md`](cancer_rec_system/reproducibility/README.md). 

## Simulations in the supplementary materials.


```bash
python scripts/run_sensitivity.py \
  --variants baseline mild_status_outcome_penalties strong_status_outcome_penalties \
             no_year_adjustment nccn_score_3_0 nccn_score_3_5 phase_nccn_neutral \
             no_status_outcome_adjustment \
  --seeds 2000 2001 2002
python scripts/run_simulation_study.py --feature-dim 30
```


## Data 

The research dataset was assembled from ClinicalTrials.gov records, RxNorm,
PharmGKB, large language model outputs, mutation-source data,
and manually abstracted guideline evidence. 


## License and data terms

No open-source license has been assigned yet, and no permission for reuse or
redistribution is granted by this repository. The research data are not included
in this repository and remain subject to their original providers' terms
(ClinicalTrials.gov, RxNorm, PharmGKB/ClinPGx, and NCCN guideline-derived
evidence). The full data-source review accompanies the private data package and
is not part of this code-only release.

## Citation

```bibtex
@inproceedings{
tang2026bridging,
title={Bridging Data Gaps in Oncology: Integrating Large Language Models and Recommendation Systems for Evidence Synthesis},
author={Tengjie Tang and Angkai Li and Xuanyu Bai and Xingye Tan and Qingli Ji and Michael Ulis and Lu Si and Le Bao},
booktitle={The First Conference on Statistics and Trustworthy AI for Cross (X)-Domain Acceleration},
year={2026},
url={https://openreview.net/forum?id=or67MercQB}
}
```
