# Reproducing the Reported Results

This directory contains the code and frozen configurations used to investigate code reproducibility.
The successful audit used macOS arm64, Python 3.9.18, and CPU execution. Full
details are in `environment.json` and `environment-macos-arm64.yml`. Place `summary_w_score.csv` and `targeted_therapy.csv` in the repository-root `private_data/` directory.


## Results covered

| Experiment | Recorded result |
| --- | ---: |
| MF, historical 10-fold CV | RMSE `0.68` |
| ID-only NCF, historical 10-fold CV | RMSE `0.4947` |
| Hybrid-NCF, historical 10-fold CV | RMSE `0.4643` |




## Replay the historical CV results

ID-only NCF:

```bash
python -m cancer_rec_system.reproducibility.cv_ncf \
  --protocol faithful --run-id faithful_10x50 \
  --data /path/to/private/summary_w_score.csv

python -m cancer_rec_system.reproducibility.verify_cv ncf \
  cancer_rec_system/params/cv-ncf-runs/faithful_10x50
```

Hybrid-NCF:

```bash
python -m cancer_rec_system.reproducibility.cv_hybrid \
  --protocol faithful --run-id faithful_10x50 \
  --data /path/to/private/summary_w_score.csv \
  --targeted /path/to/private/targeted_therapy.csv

python -m cancer_rec_system.reproducibility.verify_cv hybrid \
  cancer_rec_system/params/cv-hybrid-runs/faithful_10x50
```

