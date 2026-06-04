# How to Run — Neonatal ADR Severity Classification Pipeline

This document describes how to reproduce all results reported in the manuscript  
**"Multi-Method Statistical Signal Aggregation with Machine Learning for Severity Classification of Neonatal Adverse Drug Reactions"**

---

## Repository Structure

```
neonatal-adr-severity/
│
├── adr_severity_classification_v3_fixed.py   # Main pipeline (signal detection + ablation)
├── section_4_8_imbalance.py                  # Section 4.8 Critical class imbalance experiments
├── confirm_table4.py                         # Signal detection N-value verification script
├── HOW_TO_RUN.md                             # This file
├── requirements.txt                          # Python dependencies
│
├── data/                                     # Place dataset files here
│   ├── Neonatal_ADR_Expanded_Dataset.csv     # Main expanded dataset (826,516 rows)
│   └── signals_all.csv                       # Pharmacovigilance signal scores (34,698 pairs)
│
└── results/                                  # Created automatically on first run
    ├── cv_results_binary.csv
    ├── cv_results_multiclass.csv
    ├── test_results.csv
    └── figures/
```

---

## Data Availability

The datasets required to run this pipeline are publicly available:

| File | Source | DOI |
|---|---|---|
| `Neonatal_ADR_Expanded_Dataset.csv` | Mendeley Data | [10.17632/ppd2c7sz8j.2](https://doi.org/10.17632/ppd2c7sz8j.2) |
| Raw FAERS quarterly data | FDA FAERS Public Dashboard | [FDA FAERS](https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers/fda-adverse-event-reporting-system-faers-public-dashboard) |

Download both files and place them in the `data/` directory before running.

---

## Requirements

### System requirements

| Component | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.11 |
| RAM | 16 GB | 32 GB |
| CPU | 4 cores | 8+ cores |
| Disk | 5 GB free | 10 GB free |

### Install dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`**
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openpyxl>=3.1.0
scipy>=1.10.0
tqdm>=4.65.0
```

---

## Step 1 — Verify Signal Detection Figures (Table 4)

Before running the main pipeline, confirm that the signal file figures match the manuscript.

```bash
python confirm_table4.py \
    --sig  data/signals_all.csv \
    --csv  data/Neonatal_ADR_Expanded_Dataset.csv
```

**Expected output:**
```
Total candidate pairs (signals_all.csv rows) : 34,698
Unique pairs in expanded dataset             : 124,841
Flagged by ≥ 1 method                        : 26,409
Flagged by all 4 methods                     : 13,506
```

---

## Step 2 — Run the Main Classification Pipeline

This script runs the complete ablation study:
- Feature engineering (FS1, FS2, FS3)
- 5-fold stratified cross-validation
- 6 classifiers × 3 feature sets × 2 tasks (binary + multiclass)
- ROC and PR curve figures
- Confusion matrices

### Full dataset (recommended)

```bash
python adr_severity_classification_v3_fixed.py \
    --csv    data/Neonatal_ADR_Expanded_Dataset.csv \
    --sig    data/signals_all.csv \
    --sample 0 \
    --folds  5 \
    --out    results/
```

### Quick test run (50,000 sample)

```bash
python adr_severity_classification_v3_fixed.py \
    --csv    data/Neonatal_ADR_Expanded_Dataset.csv \
    --sig    data/signals_all.csv \
    --sample 50000 \
    --folds  5 \
    --out    results_sample/
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | `Neonatal_ADR_Expanded_Dataset.csv` | Path to expanded dataset |
| `--sig` | `signals_all.csv` | Path to signal scores file |
| `--sample` | `50000` | Stratified sample size (0 = full dataset) |
| `--folds` | `5` | Number of cross-validation folds |
| `--out` | `severity_v3` | Output directory for results and figures |

### Expected runtime (full dataset, 8-core CPU)

| Step | Approximate time |
|---|---|
| Data loading and feature engineering | ~3 min |
| 5-fold CV × 6 models × 3 feature sets × 2 tasks | ~45–90 min |
| Test set evaluation and figures | ~5 min |
| **Total** | **~60–100 min** |

### Key outputs

| File | Description |
|---|---|
| `cv_results_binary.csv` | 5-fold CV results for binary task (Tables 9) |
| `cv_results_multiclass.csv` | 5-fold CV results for multiclass task (Table 10) |
| `test_results.csv` | Held-out test set results |
| `fig_roc_binary.png` | ROC curves — binary task (Fig 5) |
| `fig_roc_multiclass_A.png` | ROC curves — multiclass, best model per FS (Fig 8) |
| `fig_roc_multiclass_B.png` | ROC curves — multiclass, all models FS3 (Fig 9) |
| `fig_cm_binary.png` | Confusion matrices — binary task (Fig 7) |
| `fig_cm_multiclass.png` | Confusion matrices — multiclass task (Fig 10) |

---

## Step 3 — Run Section 4.8 Critical Class Imbalance Experiments

This script reproduces the five imbalance-handling strategies reported in Section 4.8 and Table 11.

```bash
python section_4_8_imbalance.py \
    --csv    data/Neonatal_ADR_Expanded_Dataset.csv \
    --sig    data/signals_all.csv \
    --sample 0 \
    --seed   42 \
    --out    imbalance_results/
```

> **Important:** Use `--seed 42` to match the train/validation/test split used in the main pipeline.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--csv` | `Neonatal_ADR_Expanded_Dataset.csv` | Path to expanded dataset |
| `--sig` | `signals_all.csv` | Path to signal scores file |
| `--sample` | `0` | Sample size (0 = full dataset) |
| `--seed` | `42` | Random seed — must match main pipeline |
| `--out` | `imbalance_results` | Output directory |

### Expected runtime (full dataset)

| Strategy | Approximate time |
|---|---|
| S0 Baseline | ~15 s |
| S1 Cost-Sensitive | ~12 s |
| S2 SMOTE+ENN | ~150 s |
| S3 Threshold Tuning | ~10 s |
| S4 Cascade | ~10 s |
| **Total** | **~4 min** |

### Key outputs

| File | Description |
|---|---|
| `table11_overall_metrics.csv` | Overall metrics per strategy (Table 11a) |
| `table11_imbalance_critical.csv` | Per-class metrics per strategy (Table 11b) |
| `Table11_Section48.xlsx` | Full Excel workbook with all Table 11 sheets |
| `fig_s48_critical_focus.png` | Critical precision-recall scatter (Fig S1) |
| `fig_s48_perclass_f1.png` | Per-class F1 grouped bar chart (Fig S2) |
| `fig_s48_confusion_matrices.png` | Confusion matrices all strategies (Fig S3) |
| `fig_s48_delta_critical.png` | Delta vs baseline bar chart (Fig S4) |

---

## Reproducing Key Manuscript Figures

| Figure | Script | Output file |
|---|---|---|
| Fig 1 (Pipeline diagram) | — | `Figure1_revised.png` (provided) |
| Fig 5 (Binary ROC) | `adr_severity_classification_v3_fixed.py` | `fig_roc_binary.png` |
| Fig 8 (Multiclass ROC) | `adr_severity_classification_v3_fixed.py` | `fig_roc_multiclass_A.png` |
| Fig 9 (Multiclass ROC per class) | `adr_severity_classification_v3_fixed.py` | `fig_roc_multiclass_B.png` |
| Table 11 figures | `section_4_8_imbalance.py` | `fig_s48_*.png` |

---

## Reproducing Table 4 (Dataset Statistics)

```bash
python confirm_table4.py \
    --sig  data/signals_all.csv \
    --csv  data/Neonatal_ADR_Expanded_Dataset.csv
```

All figures in Table 4 are derived directly from the dataset files and require no additional computation.

---

## Notes on Reproducibility

- All scripts use `random_state=42` by default. Pass `--seed 42` explicitly where required to ensure identical train/validation/test splits across scripts.
- Target encoding of drug and ADR features is performed **within each training fold** to prevent data leakage. This is the default behaviour in `adr_severity_classification_v3_fixed.py`.
- Signal features for drug-ADR pairs not present in `signals_all.csv` (pairs with n < 3 co-occurrences) are zero-imputed: `log_PRR = 0`, `log_ROR = 0`, `IC = 0`, `log_EBGM = 0`, `n_methods = 0`, `sig_all4 = 0`.
- The SMOTE implementation in `section_4_8_imbalance.py` does not require `imbalanced-learn` and is implemented using `sklearn.neighbors.NearestNeighbors`.

---

## Citation

If you use this code or dataset, please cite:

```
Ronalda M, Roohie Naaz Mir.
Multi-Method Statistical Signal Aggregation with Machine Learning
for Severity Classification of Neonatal Adverse Drug Reactions.
Naunyn-Schmiedeberg's Archives of Pharmacology, 2026.
Submission ID: 9eb63d1e-5df7-43b4-b152-e3b2057b5ddd
```

Dataset:
```
Ronalda M, Roohie Naaz Mir.
Neonatal Drug ADR Association Dataset. Mendeley Data, V2, 2025.
DOI: 10.17632/ppd2c7sz8j.2
```

---

## Contact

For questions regarding the code or data, please contact:

**Ronalda M** (Corresponding Author)  
Department of Computer Science and Engineering  
National Institute of Technology Srinagar, J&K – 190006, India  
ronalda_2022phacse005@nitsri.ac.in
