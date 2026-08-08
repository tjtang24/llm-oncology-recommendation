# Recommendation-System Training and Reproduction Source

[Back to the repository overview](../README.md)

This directory contains the hybrid neural collaborative filtering (Hybrid-NCF)
model definition, the maintainable training and bundle-export source, and the
historically pinned MF / ID-only NCF / Hybrid-NCF baseline reproduction runners.

> **Research prototype:** The recommendations are generated for research use
> only and are not intended for patient-level or clinical decision-making.
> Predicted scores are uncalibrated ranking outputs, not efficacy probabilities
> or estimates of clinical benefit.

## Recommendation-system source

```text
model.py
training/
├── features.py
├── train_hybrid_ncf.py
├── export_bundle.py
└── data_manifest.json
```

The training input schemas are:

- `summary_w_score.csv`: `Cancer`, `Intervention`, `Score`, `Phases`, `Year`;
- `targeted_therapy.csv`: `drug`, `mutation`.

Both CSVs are intentionally git-ignored; place them in the repo-root
`private_data/` directory, which is the default input location for the trainer
and every reproduction runner. Their locally verified row counts and SHA-256
fingerprints are recorded in `training/data_manifest.json` without publishing the
data. The historical preprocessing threshold counts source
interaction rows per cancer—not distinct interventions—to preserve the current
55-user and 12,049-item ID contract.

From the repository root:

```bash
python -m pip install -r cancer_rec_system/requirements.txt

# Train with the checked-in fixed hyperparameters and export a new bundle.
python -m cancer_rec_system.training.train_hybrid_ncf
```

The default output is the git-ignored `params/cancer-ncf-pretrain-retrained/`
directory. The exporter stages and validates all eight assets before publishing
them and always refuses to overwrite an existing path. Use a new versioned
output directory for each run, then review it before any separate promotion.

The CLI verifies input bytes, row counts, and SHA-256 hashes against
`training/data_manifest.json` by default. A deliberately updated dataset requires
`--allow-unverified-inputs`; its actual fingerprints and the training
configuration are then recorded inside the output hyperparameter file. Run
`python -m cancer_rec_system.training.train_hybrid_ncf --help` to set input
paths, epochs, device, seed, or output directory.

The maintained trainer uses a pair-grouped validation split so duplicate
cancer–intervention rows cannot occur in both train and validation sets. It does
not reproduce the historical random-search/CV procedure or claim to reproduce
metrics reported outside this repository.

## MF and ID-only NCF baseline reproduction

Code-only fixed-configuration refit runners, the source-faithful historical CV
runners, the Python 3.9 CPU environment record, private input fingerprints, and
expected scientific-output hashes are provided in
[`reproducibility/`](reproducibility/README.md).

Their default locations and the documented examples use new git-ignored
`params/mf-retrained*/` and `params/ncf-retrained*/` directories. `--output` may
instead name any new writable directory, so review a custom location before
staging files. No baseline source data or model weights are committed.

## Maintainer

Tengjie Tang — [tft5323@psu.edu](mailto:tft5323@psu.edu)
