# Spam and Malware Detection Platform

A coursework-ready machine learning toolkit for building, evaluating, and reporting on dual pipelines that detect spam messages and malicious executables. The repository pairs automated data cleaning with supervised, unsupervised, and regression-style models, plus visual reporting suitable for academic submissions.

## Project Overview
- Unified Python package layout for spam (text) and malware (tabular) domains
- Repeatable data cleaning that writes canonical artefacts under `data/processed/`
- Ready-to-run training scripts covering classification, regression, and clustering tasks
- Automated metric aggregation and publication-quality plots for reports
- Re-usable utilities for file paths, evaluation figures, and text processing (NLTK-backed)

## Quick Start
### Prerequisites
- Python 3.10 or newer
- `pip` plus the packages in `requirements.txt`
- Raw datasets placed under `data/raw/spam/` and `data/raw/malware/` (see below)

> The first run that touches `nltk.corpus.stopwords` downloads the corpus automatically; internet access is only needed once.

### Environment Setup
```bash
# 1. Clone your fork
git clone https://github.com/SajanJayalath/spam-malware-detection
cd spam-malware-detection

# 2. Create an isolated environment
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 3. Install Python packages
pip install --upgrade pip
pip install -r requirements.txt
```

### Dataset Placement
```
data/
    raw/
        spam/
            spam_ds1.csv
            spam_ds2.csv
            spam_ds3.csv
        malware/
            malware_ds1.csv
            malware_ds2.csv
    processed/             # Populated by the processing scripts
```

## Project Structure
```
Spam-and-Malware-Detection/
    src/
        common/            # Shared helpers (paths, evaluation plots, text utilities)
        spam_detection/    # Spam data prep, modelling, and visualisation modules
        malware_detection/ # Malware data prep, modelling, and visualisation modules
    scripts/               # CLI entry points for pipelines, training, and reports
    data/                  # Raw (input) and processed (generated) datasets
    models/                # Serialised joblib models produced by training scripts
    reports/               # Metrics JSON, CSV summaries, charts, and figure assets
    notebooks/             # Exploratory analysis notebooks
    docs/                  # Coursework documentation artefacts
    requirements.txt       # Python dependency list
    README.md              # This file
```

## Data Processing Pipelines
Run these once after placing raw datasets. Artefacts land in `data/processed/<domain>/` and diagnostic figures in `reports/<domain>/`.

```bash
# Clean and profile all spam datasets
python scripts/process_spam.py

# Clean, standardise labels, and visualise malware datasets
python scripts/process_malware.py
```

Each script performs deduplication, feature casting, and dataset-specific cleaning before writing canonical CSVs and exploratory graphics (label balance, token usage, section sizes, PCA projections, etc.).

## Model Training & Analysis
### Spam models
```bash
# Multinomial Naive Bayes text classifier (saves model + metrics)
python scripts/train_spam_nb.py

# ElasticNet regression (dense TF-IDF + TruncatedSVD features, thresholded at 0.5)
python scripts/train_spam_regression.py

# KMeans clustering over dense text embeddings
python scripts/cluster_spam_kmeans.py
```
Artefacts: models stored in `models/spam/`, metrics JSON in `reports/spam/metrics/`, cluster assignments under `reports/spam/kmeans_assignments.csv`.

### Malware models
```bash
# RandomForest classifier on dataset 1
python scripts/train_malware_ds1_rf.py

# Histogram-based Gradient Boosting classifier on dataset 2
python scripts/train_malware_ds2_hist_gbdt.py

# Unsupervised clustering for dataset 2
python scripts/cluster_malware_ds2_kmeans.py
```
Artefacts: models saved to `models/malware/`, metrics JSON and assignments under `reports/malware/`.

## Evaluation Workflow
1. Aggregate metrics across domains:
   ```bash
   python scripts/summarise_metrics.py
   ```
   Produces `reports/metrics_summary.csv` and prints a tabular overview.
2. Regenerate publication-quality ROC, PR, confusion, residual, and bar-chart figures:
   ```bash
   python scripts/generate_evaluation_plots.py
   ```
   Exports figures into `reports/<domain>/evaluation/` using the persisted models and test splits.

## Model Performance Snapshot
### Spam classification/regression
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Multinomial Naive Bayes | 97.99% | 97.70% | 93.26% | 95.42% | 99.69% |
| ElasticNet (threshold 0.5) | 95.49% | 96.42% | 82.99% | 89.20% | 99.04% |

Additional ElasticNet regression metrics: RMSE 0.217, MAE 0.156, R^2 0.729.

### Spam clustering
| Metric | Value |
| --- | --- |
| Silhouette score | 0.010 |
| Adjusted Rand Index | -0.021 |
| Mapped accuracy | 77.55% |
| Cluster sizes | 9,577 vs 5,611 |

### Malware classification
| Dataset | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | --- | --- | --- | --- | --- | --- |
| malware_ds1 | Random Forest | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| malware_ds2 | HistGradientBoosting | 92.59% | 98.47% | 93.32% | 95.82% | 96.20% |

### Malware clustering
| Metric | Value |
| --- | --- |
| Silhouette score | 0.987 |
| Adjusted Rand Index | 0.0027 |
| Mapped accuracy | 91.11% |
| Cluster sizes | 20,848 vs 3 |

> Exact values come from the JSON files in `reports/<domain>/metrics/` and are also visible in `reports/metrics_summary.csv`.

## Artefacts & Reporting
- Clean data: `data/processed/<domain>/*.csv`
- Persisted models: `models/<domain>/*.joblib`
- Metrics (JSON) + summary CSV: `reports/<domain>/metrics/`, `reports/metrics_summary.csv`
- Visualisations: `reports/<domain>/*.png` and `reports/<domain>/evaluation/*.png`
- Cluster assignments: `reports/<domain>/*_kmeans_assignments.csv`

These outputs are designed to drop straight into written reports or presentation decks.

## Notebooks
Exploratory notebooks under `notebooks/` can leverage the processed datasets for feature analysis or alternative modelling experiments. Ensure you point them to the artefacts generated by the CLI scripts to keep results consistent.

## Troubleshooting
- **Missing NLTK stopwords**: run `python -m nltk.downloader stopwords` (normally handled automatically).
- **Dataset not found**: confirm filenames match the expected casing and live beneath `data/raw/<domain>/`.
- **Metrics missing**: rerun the relevant training script, then `scripts/summarise_metrics.py`.
- **Plots not updating**: delete stale figures in `reports/<domain>/evaluation/` if you want a clean rebuild before running `generate_evaluation_plots.py`.

## Next Steps
- Extend the spam pipeline with transformer-based encoders or ensemble methods.
- Introduce automated unit tests (e.g., `pytest`) for dataset cleaners and model wrappers.
- Containerise the workflows for fully reproducible execution in CI/CD environments.

