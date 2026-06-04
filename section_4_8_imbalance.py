"""
=============================================================================
  Section 4.8 — Critical Class Imbalance Experiments
  Neonatal ADR Severity Classification
  ---------------------------------------------------------------------------

  Base model  : HistGradientBoosting on FS3 Combined (17 features)
  Task        : Multiclass severity (5 classes)
  Focus metric: Critical class — Precision, Recall, F1, AUPRC

  Five strategies evaluated:

    S0  Baseline         — class_weight='balanced' (standard baseline)
    S1  Cost-Sensitive   — asymmetric clinical cost-matrix sample weights
                           Critical misclassification penalised 5× over
                           Moderate, consistent with clinical risk hierarchy
    S2  SMOTE+ENN        — Borderline-SMOTE synthetic oversampling of
                           minority classes followed by ENN cleaning;
                           implemented without imbalanced-learn dependency
    S3  Threshold Tuning — per-class threshold multipliers optimised on
                           validation set via coordinate-wise grid search
    S4  Cascade          — two-stage classifier: Stage 1 binary triage
                           (mild vs. serious spectrum), Stage 2 fine-grained
                           classification within each branch

  Outputs:
    table11_imbalance_critical.csv     — per-strategy per-class metrics
    table11_overall_metrics.csv        — overall macro metrics per strategy
    fig_s48_critical_focus.png         — Critical precision-recall scatter
    fig_s48_perclass_f1.png            — per-class F1 grouped bar chart
    fig_s48_confusion_matrices.png     — normalised confusion matrices (5×5)
    fig_s48_delta_critical.png         — delta vs S0 baseline bar chart

  USAGE
  -----
    # Standalone — uses same defaults as main pipeline
    python section_4_8_imbalance.py

    # With explicit paths (must match main pipeline run)
    python section_4_8_imbalance.py \\
        --csv    Neonatal_ADR_Expanded_Dataset.csv \\
        --sig    signals_all.csv \\
        --sample 0 \\
        --out    imbalance_results/

    # Use same sample size as main pipeline for consistency
    python section_4_8_imbalance.py \\
        --csv    Neonatal_ADR_Expanded_Dataset.csv \\
        --sig    signals_all.csv \\
        --sample 50000 \\
        --seed   42 \\
        --out    imbalance_results/

  DEPENDENCIES
  ------------
    pandas  numpy  scikit-learn  matplotlib
    (imbalanced-learn NOT required — SMOTE implemented from scratch)
=============================================================================
"""

import argparse
import warnings
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASSES    = ["Non-Serious", "Moderate", "Serious", "Critical", "Fatal"]
N_CLASSES  = 5
SEED       = 42

# FS3 Combined feature columns (must match main pipeline)
FS1 = ["drug_freq", "drug_target", "adr_freq", "adr_target", "rfu_freq",
       "sex_enc", "age_days_imp", "weight_kg_imp",
       "drug_adr_freq", "drug_n_adrs", "adr_n_drugs"]
FS2 = ["log_PRR", "log_ROR", "IC_imp", "log_EBGM", "n_methods", "sig_all4"]
FS3 = FS1 + FS2

# Clinical cost matrix  C[true_class, predicted_class]
# Higher value = worse clinical consequence of this misclassification
# Row order : Non-Serious(0), Moderate(1), Serious(2), Critical(3), Fatal(4)
COST_MATRIX = np.array([
    [0, 1, 2, 3, 2],   # True Non-Serious
    [1, 0, 1, 3, 2],   # True Moderate
    [2, 1, 0, 2, 1],   # True Serious
    [5, 4, 3, 0, 1],   # True Critical  ← highest row: misclassifying costs most
    [4, 3, 2, 1, 0],   # True Fatal
], dtype=np.float32)

# Plot colours per strategy
STRAT_COLORS = {
    "S0 Baseline"     : "#546E7A",
    "S1 Cost-Sensitive": "#1565C0",
    "S2 SMOTE+ENN"    : "#2E7D32",
    "S3 Threshold"    : "#EF6C00",
    "S4 Cascade"      : "#6A1B9A",
}
CLASS_COLORS = ["#9E9E9E", "#2E7D32", "#1565C0", "#EF6C00", "#C62828"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#CCCCCC", "axes.grid": True,
    "grid.color": "#EBEBEB", "grid.linewidth": 0.5,
    "font.size": 10,
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING  (mirrors load() + engineer() from main pipeline)
# ─────────────────────────────────────────────────────────────────────────────

def load_and_engineer(csv_path: str, sig_path: str,
                      sample: int, seed: int) -> pd.DataFrame:
    """Load, merge signal scores, and engineer all FS3 features."""
    print("\n[1/4] Loading data ...")
    df  = pd.read_csv(csv_path)
    sig = pd.read_csv(sig_path)
    print(f"  Expanded rows : {len(df):,}")
    print(f"  Signal pairs  : {len(sig):,}")

    # Normalise string keys
    df["drug"]  = df["drug"].astype(str).str.strip()
    df["adr"]   = df["adr"].astype(str).str.strip()
    sig["drug"] = sig["drug"].astype(str).str.strip()
    sig["adr"]  = sig["adr"].astype(str).str.strip()

    # Normalise signal column names
    rename_map = {
        "n_methods_flagged": "n_methods", "num_methods": "n_methods",
        "signal_ALL_4": "sig_all4",       "sig_all_4":   "sig_all4",
        "signal_all4":  "sig_all4",
        "PRR_lower95": "PRR_lo95",        "PRR_upper95": "PRR_hi95",
        "ROR_lower95": "ROR_lo95",        "ROR_upper95": "ROR_hi95",
        "chi2_PRR": "chi2",               "pvalue_PRR":  "pval_PRR",
        "signal_PRR": "sig_PRR",          "signal_ROR":  "sig_ROR",
        "signal_IC":  "sig_IC",           "signal_EBGM": "sig_EBGM",
    }
    sig = sig.rename(columns={k: v for k, v in rename_map.items()
                               if k in sig.columns})

    for col, default in {"n_methods": 0, "sig_all4": 0,
                         "PRR": 1.0, "ROR": 1.0,
                         "IC": 0.0,  "EBGM": 1.0}.items():
        if col not in sig.columns:
            sig[col] = default

    for col in ["PRR", "ROR", "IC", "EBGM", "n_methods", "sig_all4"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    sig_cols = ["drug", "adr", "PRR", "ROR", "IC", "EBGM",
                "n_methods", "sig_all4"]
    df = df.merge(sig[sig_cols], on=["drug", "adr"], how="left")

    hit = df["PRR"].notna().sum()
    print(f"  Signal match  : {hit:,}/{len(df):,} = {100*hit/len(df):.1f}%")

    # Stratified sample
    if sample and sample > 0 and sample < len(df):
        df, _ = train_test_split(
            df, train_size=sample,
            stratify=df["severity_score"], random_state=seed)
        df = df.reset_index(drop=True)
        print(f"  Sampled to    : {len(df):,} rows")

    # Feature engineering (identical to main pipeline)
    age_med = df["age_days"].median()
    wt_med  = df["weight_kg"].median()
    df = df.copy()

    df = df.join(df.groupby(["drug", "adr"]).size().rename("_pf"),
                 on=["drug", "adr"])
    df = df.join(df.groupby("drug")["adr"].nunique().rename("_dna"), on="drug")
    df = df.join(df.groupby("adr")["drug"].nunique().rename("_and"), on="adr")

    df["drug_freq"]     = df["drug"].map(df["drug"].value_counts()).fillna(1)
    df["drug_target"]   = df["drug"].map(
        df.groupby("drug")["serious_binary"].mean()).fillna(0.5)
    df["adr_freq"]      = df["adr"].map(df["adr"].value_counts()).fillna(1)
    df["adr_target"]    = df["adr"].map(
        df.groupby("adr")["serious_binary"].mean()).fillna(0.5)
    df["rfu_freq"]      = (df["reason_for_use"].fillna("Unknown")
                           .map(df["reason_for_use"].fillna("Unknown")
                                .value_counts()).fillna(1))
    df["sex_enc"]       = df["sex"].map({"Female": 0, "Male": 1}).fillna(-1)
    df["age_days_imp"]  = df["age_days"].fillna(age_med)
    df["weight_kg_imp"] = df["weight_kg"].fillna(wt_med)
    df["drug_adr_freq"] = df["_pf"].fillna(1)
    df["drug_n_adrs"]   = df["_dna"].fillna(1)
    df["adr_n_drugs"]   = df["_and"].fillna(1)

    df["log_PRR"]  = np.log1p(df["PRR"].fillna(1))
    df["log_ROR"]  = np.log1p(df["ROR"].fillna(1))
    df["IC_imp"]   = df["IC"].fillna(0.0)
    df["log_EBGM"] = np.log1p(df["EBGM"].fillna(1))
    df["n_methods"]= df["n_methods"].fillna(0).astype(int)
    df["sig_all4"] = df["sig_all4"].fillna(False).astype(int)

    print(f"\n  Severity distribution:")
    for i, cls in enumerate(CLASSES):
        n = (df["severity_score"] == i).sum()
        print(f"    {cls:<15} {n:>8,}  ({100*n/len(df):.2f}%)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def cost_sample_weights(y: np.ndarray,
                        cost_matrix: np.ndarray) -> np.ndarray:
    """
    Compute per-sample weights from the clinical cost matrix.
    Weight for sample i = mean off-diagonal cost of its true class.
    Normalised so the mean weight = 1 (sklearn convention).
    """
    w = np.zeros(len(y), dtype=np.float32)
    for c in range(N_CLASSES):
        row     = cost_matrix[c].copy()
        row[c]  = 0.0                       # exclude correct-prediction diagonal
        w[y == c] = row.mean()
    w = w / w.mean()
    return w


def borderline_smote(X: np.ndarray, y: np.ndarray,
                     target_ratio: float = 0.15,
                     k: int = 5,
                     seed: int = 42) -> tuple:
    """
    Borderline-SMOTE multiclass oversampling without imbalanced-learn.

    For each minority class whose proportion is below target_ratio:
      1. Find borderline samples — those whose k nearest neighbours
         contain at least k/2 samples from a different class.
      2. Generate synthetic samples by interpolating between each
         borderline sample and a randomly selected same-class neighbour.

    Parameters
    ----------
    X            : feature matrix  (n_samples, n_features)
    y            : integer class labels
    target_ratio : desired minimum class proportion after resampling
    k            : number of nearest neighbours
    seed         : random state for reproducibility

    Returns
    -------
    X_resampled, y_resampled
    """
    rng      = np.random.default_rng(seed)
    counts   = np.bincount(y, minlength=N_CLASSES)
    n_total  = len(y)
    target_n = int(n_total * target_ratio)

    X_out = [X.copy()]
    y_out = [y.copy()]

    for cls in range(N_CLASSES):
        n_cls = counts[cls]
        if n_cls >= target_n:
            continue                    # already above target

        n_gen    = target_n - n_cls
        X_cls    = X[y == cls].astype(np.float32)

        if len(X_cls) < k + 1:
            # Too few samples for KNN — replicate randomly
            idx = rng.integers(0, len(X_cls), n_gen)
            X_out.append(X_cls[idx])
            y_out.append(np.full(n_gen, cls, dtype=int))
            continue

        # Step 1: KNN over all samples to identify borderline instances
        nn_all = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
        nn_all.fit(X)
        _, nbrs_all = nn_all.kneighbors(X_cls)
        nbrs_all    = nbrs_all[:, 1:]   # exclude self

        border_mask = np.array([
            (k // 2) <= (y[row] != cls).sum() < k
            for row in nbrs_all
        ])
        X_border = X_cls[border_mask]

        if len(X_border) == 0:
            # Fallback: random oversampling if no borderline found
            idx = rng.integers(0, len(X_cls), n_gen)
            X_out.append(X_cls[idx])
            y_out.append(np.full(n_gen, cls, dtype=int))
            continue

        # Step 2: KNN among same-class samples
        k_same = min(k, len(X_cls) - 1)
        nn_cls = NearestNeighbors(n_neighbors=k_same + 1, n_jobs=-1)
        nn_cls.fit(X_cls)
        _, nbrs_cls = nn_cls.kneighbors(X_border)
        nbrs_cls    = nbrs_cls[:, 1:]

        # Step 3: Generate synthetic samples
        synthetics = []
        for _ in range(n_gen):
            b_i   = rng.integers(0, len(X_border))
            n_i   = rng.choice(nbrs_cls[b_i])
            alpha = rng.random()
            synth = X_border[b_i] + alpha * (X_cls[n_i] - X_border[b_i])
            synthetics.append(synth)

        X_out.append(np.array(synthetics, dtype=np.float32))
        y_out.append(np.full(n_gen, cls, dtype=int))

    X_res = np.vstack(X_out)
    y_res = np.concatenate(y_out)
    perm  = rng.permutation(len(X_res))
    return X_res[perm], y_res[perm]


def enn_clean(X: np.ndarray, y: np.ndarray,
              k: int = 3) -> tuple:
    """
    Edited Nearest Neighbours cleaning.
    Remove majority-class samples whose k nearest neighbours
    disagree with their label (reduces overlap near class boundaries).
    Minority classes are never removed.
    """
    counts      = np.bincount(y, minlength=N_CLASSES)
    median_cnt  = np.median(counts[counts > 0])
    nn          = NearestNeighbors(n_neighbors=k + 1, n_jobs=-1)
    nn.fit(X)
    _, nbrs     = nn.kneighbors(X)
    nbrs        = nbrs[:, 1:]

    keep = np.ones(len(y), dtype=bool)
    for i in range(len(y)):
        if counts[y[i]] < median_cnt:  # preserve minority class samples
            continue
        if (y[nbrs[i]] != y[i]).sum() > k // 2:
            keep[i] = False

    return X[keep], y[keep]


def optimise_thresholds(p_val: np.ndarray,
                        y_val: np.ndarray) -> np.ndarray:
    """
    Coordinate-wise grid search for per-class decision threshold multipliers.

    Prediction rule:  argmax_c  p[i, c] / threshold[c]
    Optimises macro-F1 on the validation set.
    """
    C     = p_val.shape[1]
    thr   = np.ones(C, dtype=np.float64)
    grid  = np.linspace(0.3, 2.5, 40)

    def score(t):
        return f1_score(y_val, (p_val / t).argmax(1),
                        average="macro", zero_division=0)

    best = score(thr)
    for _ in range(6):                  # multiple passes until convergence
        for c in range(C):
            best_t = thr[c]
            for t in grid:
                thr[c] = t
                s = score(thr)
                if s > best:
                    best = s
                    best_t = t
            thr[c] = best_t

    return thr


# ─────────────────────────────────────────────────────────────────────────────
# 3. TWO-STAGE CASCADE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

class TwoStageCascade:
    """
    Stage 1 — Binary triage: mild (Non-Serious + Moderate, classes 0-1)
              vs. serious spectrum (Serious + Critical + Fatal, classes 2-4).

    Stage 2a — Fine-grained within serious spectrum:
               HistGBM on classes {2, 3, 4} only.

    Stage 2b — Fine-grained within mild spectrum:
               HistGBM on classes {0, 1} only.

    Rationale: dedicating separate model capacity to each branch reduces
    confusion between Critical and Serious, which are adjacent in severity
    but statistically overlap in the 17-dimensional feature space when
    trained jointly with all five classes.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed

        self.clf_stage1 = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=10,
            class_weight="balanced", random_state=seed)

        self.clf_serious = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05,
            max_leaf_nodes=63, min_samples_leaf=5,
            class_weight="balanced", random_state=seed)

        self.clf_mild = HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=5,
            class_weight="balanced", random_state=seed)

        self.sc1 = StandardScaler()
        self.sc2 = StandardScaler()
        self.sc3 = StandardScaler()

        self._serious_classes = None   # populated on fit
        self._mild_classes    = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Stage 1: binary triage labels
        y1 = (y >= 2).astype(int)               # 0 = mild, 1 = serious
        X1 = self.sc1.fit_transform(X)
        self.clf_stage1.fit(X1, y1)

        # Stage 2a: serious spectrum (classes 2, 3, 4)
        mask_s = y >= 2
        if mask_s.sum() > 0:
            self._serious_classes = np.unique(y[mask_s])
            X2 = self.sc2.fit_transform(X[mask_s])
            self.clf_serious.fit(X2, y[mask_s] - 2)    # remap to 0,1,2

        # Stage 2b: mild spectrum (classes 0, 1)
        mask_m = y < 2
        if mask_m.sum() > 0 and len(np.unique(y[mask_m])) > 1:
            self._mild_classes = np.unique(y[mask_m])
            X3 = self.sc3.fit_transform(X[mask_m])
            self.clf_mild.fit(X3, y[mask_m])
        else:
            self.clf_mild = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X1   = self.sc1.transform(X)
        y1   = self.clf_stage1.predict(X1)
        pred = np.zeros(len(X), dtype=int)

        # Serious branch
        s_mask = y1 == 1
        if s_mask.sum() > 0:
            X2 = self.sc2.transform(X[s_mask])
            pred[s_mask] = self.clf_serious.predict(X2) + 2

        # Mild branch
        m_mask = y1 == 0
        if m_mask.sum() > 0:
            if self.clf_mild is not None:
                X3 = self.sc3.transform(X[m_mask])
                pred[m_mask] = self.clf_mild.predict(X3)
            else:
                pred[m_mask] = 1                 # default: Moderate

        return pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Construct a 5-class probability matrix from stage outputs.
        P(class c | X) = P(branch | X) × P(class | branch, X)
        """
        n    = len(X)
        prob = np.zeros((n, N_CLASSES), dtype=np.float32)

        X1 = self.sc1.transform(X)
        p1 = self.clf_stage1.predict_proba(X1)   # (n, 2): [P(mild), P(serious)]

        # Serious branch probabilities (classes 2,3,4)
        X2 = self.sc2.transform(X)
        p2 = self.clf_serious.predict_proba(X2)  # (n, ≤3)
        n_ser = p2.shape[1]
        for c in range(n_ser):
            prob[:, c + 2] = p1[:, 1] * p2[:, c]

        # Mild branch probabilities (classes 0,1)
        if self.clf_mild is not None:
            X3 = self.sc3.transform(X)
            p3 = self.clf_mild.predict_proba(X3)  # (n, ≤2)
            n_mild = p3.shape[1]
            for c in range(n_mild):
                prob[:, c] = p1[:, 0] * p3[:, c]
        else:
            prob[:, 1] = p1[:, 0]                # all mild mass → Moderate

        # Normalise rows to sum to 1
        row_sums = prob.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        prob /= row_sums
        return prob


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def compute_overall_metrics(y_true, y_pred, proba, label):
    auroc = roc_auc_score(y_true, proba,
                          multi_class="ovr", average="macro")
    auprc = average_precision_score(
        np.eye(N_CLASSES)[y_true], proba, average="macro")
    return dict(
        strategy     = label,
        auroc        = round(float(auroc), 4),
        auprc        = round(float(auprc), 4),
        f1_macro     = round(f1_score(y_true, y_pred, average="macro",
                                      zero_division=0), 4),
        balanced_acc = round(balanced_accuracy_score(y_true, y_pred), 4),
        mcc          = round(matthews_corrcoef(y_true, y_pred), 4),
    )


def compute_perclass_metrics(y_true, y_pred, proba, label):
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES, zero_division=0, output_dict=True)
    rows = []
    for i, cls in enumerate(CLASSES):
        ap = average_precision_score(
            (y_true == i).astype(int), proba[:, i])
        rows.append(dict(
            strategy  = label,
            cls       = cls,
            support   = int((y_true == i).sum()),
            precision = round(report[cls]["precision"], 4),
            recall    = round(report[cls]["recall"],    4),
            f1        = round(report[cls]["f1-score"],  4),
            auprc     = round(float(ap), 4),
        ))
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 5. STRATEGY RUNNERS
# ─────────────────────────────────────────────────────────────────────────────

def run_s0_baseline(X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """S0: HistGBM with class_weight='balanced'."""
    print("  S0 — Baseline (class_weight=balanced) ...", flush=True)
    t0  = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=10,
        class_weight="balanced", random_state=seed)
    clf.fit(Xtr, y_tr)
    elapsed = time.time() - t0

    p_te = clf.predict_proba(sc.transform(X_te))
    pred = p_te.argmax(1)
    met  = compute_overall_metrics(y_te, pred, p_te, "S0 Baseline")
    met["time_s"] = round(elapsed, 1)
    pc   = compute_perclass_metrics(y_te, pred, p_te, "S0 Baseline")
    print(f"    AUROC={met['auroc']}  F1={met['f1_macro']}  "
          f"Critical-F1={pc[pc.cls=='Critical']['f1'].values[0]:.3f}  "
          f"time={elapsed:.1f}s")
    return met, pc, p_te, pred, clf, sc


def run_s1_cost_sensitive(X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """S1: HistGBM with asymmetric clinical cost-matrix sample weights."""
    print("  S1 — Clinical cost-sensitive sample weights ...", flush=True)
    t0  = time.time()
    sc  = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    sw  = cost_sample_weights(y_tr, COST_MATRIX)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=10,
        random_state=seed)
    clf.fit(Xtr, y_tr, sample_weight=sw)
    elapsed = time.time() - t0

    p_te = clf.predict_proba(sc.transform(X_te))
    pred = p_te.argmax(1)
    met  = compute_overall_metrics(y_te, pred, p_te, "S1 Cost-Sensitive")
    met["time_s"] = round(elapsed, 1)
    pc   = compute_perclass_metrics(y_te, pred, p_te, "S1 Cost-Sensitive")
    print(f"    AUROC={met['auroc']}  F1={met['f1_macro']}  "
          f"Critical-F1={pc[pc.cls=='Critical']['f1'].values[0]:.3f}  "
          f"time={elapsed:.1f}s")
    return met, pc, p_te, pred


def run_s2_smote(X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """S2: Borderline-SMOTE + ENN cleaning, then HistGBM."""
    print("  S2 — Borderline-SMOTE + ENN ...", flush=True)
    t0 = time.time()

    # Target ratio: oversample to at least 3× current Critical proportion
    counts      = np.bincount(y_tr, minlength=N_CLASSES)
    crit_rate   = counts[3] / len(y_tr)
    target_ratio= min(max(crit_rate * 3, 0.08), 0.20)

    print(f"    Original distribution: {counts}")
    X_res, y_res = borderline_smote(
        X_tr.astype(np.float32), y_tr,
        target_ratio=target_ratio, k=5, seed=seed)
    print(f"    After SMOTE : {len(y_res):,} samples  "
          f"{np.bincount(y_res, minlength=N_CLASSES)}")

    X_res, y_res = enn_clean(X_res, y_res, k=3)
    print(f"    After ENN   : {len(y_res):,} samples  "
          f"{np.bincount(y_res, minlength=N_CLASSES)}")

    sc  = StandardScaler()
    Xr  = sc.fit_transform(X_res)
    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.05,
        max_leaf_nodes=63, min_samples_leaf=10,
        class_weight="balanced", random_state=seed)
    clf.fit(Xr, y_res)
    elapsed = time.time() - t0

    p_te = clf.predict_proba(sc.transform(X_te))
    pred = p_te.argmax(1)
    met  = compute_overall_metrics(y_te, pred, p_te, "S2 SMOTE+ENN")
    met["time_s"] = round(elapsed, 1)
    pc   = compute_perclass_metrics(y_te, pred, p_te, "S2 SMOTE+ENN")
    print(f"    AUROC={met['auroc']}  F1={met['f1_macro']}  "
          f"Critical-F1={pc[pc.cls=='Critical']['f1'].values[0]:.3f}  "
          f"time={elapsed:.1f}s")
    return met, pc, p_te, pred


def run_s3_threshold(X_tr, y_tr, X_val, y_val, X_te, y_te, seed,
                     base_clf, base_sc):
    """S3: Per-class threshold optimisation on validation set."""
    print("  S3 — Per-class threshold optimisation ...", flush=True)
    t0 = time.time()

    p_val = base_clf.predict_proba(base_sc.transform(X_val))
    thr   = optimise_thresholds(p_val, y_val)
    elapsed = time.time() - t0

    print(f"    Optimised thresholds: "
          f"{dict(zip(CLASSES, thr.round(3)))}")

    p_te = base_clf.predict_proba(base_sc.transform(X_te))
    pred = (p_te / thr).argmax(1)
    met  = compute_overall_metrics(y_te, pred, p_te, "S3 Threshold")
    met["time_s"] = round(elapsed, 1)
    pc   = compute_perclass_metrics(y_te, pred, p_te, "S3 Threshold")
    print(f"    AUROC={met['auroc']}  F1={met['f1_macro']}  "
          f"Critical-F1={pc[pc.cls=='Critical']['f1'].values[0]:.3f}  "
          f"time={elapsed:.1f}s")
    return met, pc, p_te, pred, thr


def run_s4_cascade(X_tr, y_tr, X_val, y_val, X_te, y_te, seed):
    """S4: Two-stage cascade classifier."""
    print("  S4 — Two-stage cascade classifier ...", flush=True)
    t0  = time.time()
    clf = TwoStageCascade(seed=seed)
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    p_te = clf.predict_proba(X_te)
    pred = clf.predict(X_te)
    met  = compute_overall_metrics(y_te, pred, p_te, "S4 Cascade")
    met["time_s"] = round(elapsed, 1)
    pc   = compute_perclass_metrics(y_te, pred, p_te, "S4 Cascade")
    print(f"    AUROC={met['auroc']}  F1={met['f1_macro']}  "
          f"Critical-F1={pc[pc.cls=='Critical']['f1'].values[0]:.3f}  "
          f"time={elapsed:.1f}s")
    return met, pc, p_te, pred


# ─────────────────────────────────────────────────────────────────────────────
# 6. FIGURES
# ─────────────────────────────────────────────────────────────────────────────

def fig_critical_focus(pc_df, out_dir):
    """Critical class precision-recall scatter + F1/recall bar comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: precision vs recall scatter with F1 iso-lines
    ax = axes[0]
    crit = pc_df[pc_df.cls == "Critical"]
    for _, row in crit.iterrows():
        strat = row["strategy"]
        ax.scatter(row["recall"], row["precision"],
                   s=220, color=STRAT_COLORS[strat],
                   zorder=5, label=f"{strat}  F1={row['f1']:.3f}")
        ax.annotate(strat.split()[0],
                    (row["recall"], row["precision"]),
                    textcoords="offset points", xytext=(7, 5), fontsize=9)

    for f1v in [0.2, 0.3, 0.4, 0.5, 0.6]:
        r_range = np.linspace(0.01, 1.0, 300)
        p_range = f1v * r_range / (2 * r_range - f1v + 1e-9)
        mask = (p_range >= 0) & (p_range <= 1)
        ax.plot(r_range[mask], p_range[mask],
                "gray", lw=0.8, ls="--", alpha=0.6)
        ax.text(r_range[mask][-1], p_range[mask][-1] + 0.01,
                f"F1={f1v}", fontsize=7.5, color="gray")

    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Critical Recall", fontsize=11)
    ax.set_ylabel("Critical Precision", fontsize=11)
    ax.set_title("Critical Class — Precision vs Recall",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower left"); ax.grid(alpha=0.3)

    # Right: Critical F1 and recall bar chart per strategy
    ax = axes[1]
    strategies = list(STRAT_COLORS.keys())
    x  = np.arange(len(strategies))
    w  = 0.30
    cr  = [crit[crit.strategy == s]["recall"].values[0]
           if len(crit[crit.strategy == s]) > 0 else 0 for s in strategies]
    cf1 = [crit[crit.strategy == s]["f1"].values[0]
           if len(crit[crit.strategy == s]) > 0 else 0 for s in strategies]

    b1 = ax.bar(x - w / 2, cr,  w, label="Recall",
                color=[STRAT_COLORS[s] for s in strategies],
                alpha=0.65, edgecolor="white")
    b2 = ax.bar(x + w / 2, cf1, w, label="F1",
                color=[STRAT_COLORS[s] for s in strategies],
                alpha=1.00, edgecolor="white")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            if h > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        h + 0.01, f"{h:.3f}",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(" ", "\n") for s in strategies],
                       fontsize=8.5)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Critical Class — Recall & F1 per Strategy",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = out_dir / "fig_s48_critical_focus.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_perclass_f1(pc_df, out_dir):
    """Grouped bar chart: per-class F1 for every strategy."""
    strategies = list(STRAT_COLORS.keys())
    x = np.arange(N_CLASSES)
    w = 0.14

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title in zip(
            axes, ["f1", "recall"],
            ["Per-Class F1 Score", "Per-Class Recall"]):
        for j, strat in enumerate(strategies):
            sub  = pc_df[pc_df.strategy == strat]
            vals = [sub[sub.cls == c][metric].values[0]
                    if len(sub[sub.cls == c]) > 0 else 0
                    for c in CLASSES]
            bars = ax.bar(x + j * w, vals, w,
                          label=strat,
                          color=STRAT_COLORS[strat],
                          alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.01,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)

        ax.set_xticks(x + w * (len(strategies) - 1) / 2)
        ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_ylabel(metric.capitalize(), fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=7.5, ncol=2)
        ax.grid(axis="y", alpha=0.3)
        # Highlight Critical column
        ax.axvspan(x[3] - 0.1, x[3] + len(strategies) * w + 0.05,
                   color="#FFF3E0", alpha=0.4, zorder=0)

    plt.tight_layout()
    path = out_dir / "fig_s48_perclass_f1.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_confusion_matrices(pred_dict, y_te, out_dir):
    """5×5 normalised confusion matrices for all strategies."""
    strategies = list(pred_dict.keys())
    n  = len(strategies)
    nc = 3
    nr = (n + nc - 1) // nc
    short = ["NS", "Mod", "Ser", "Crit", "Fat"]

    fig, axes = plt.subplots(nr, nc, figsize=(5.5 * nc, 4.8 * nr))
    axes = axes.ravel()

    for idx, (strat, pred) in enumerate(pred_dict.items()):
        cm  = confusion_matrix(y_te, pred, normalize="true")
        ax  = axes[idx]
        im  = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(short, fontsize=9)
        ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(short, fontsize=9)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if cm[i, j] > 0.55 else "black")
        ax.set_title(strat, fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        plt.colorbar(im, ax=ax, fraction=0.04)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Section 4.8 — Normalised Confusion Matrices (Multiclass)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = out_dir / "fig_s48_confusion_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_delta_critical(pc_df, out_dir):
    """Signed delta bars: each strategy vs S0 baseline on Critical class."""
    baseline = pc_df[(pc_df.strategy == "S0 Baseline") &
                     (pc_df.cls == "Critical")].iloc[0]
    others   = pc_df[(pc_df.strategy != "S0 Baseline") &
                     (pc_df.cls == "Critical")]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, metric in zip(axes, ["recall", "precision", "f1"]):
        b_val   = baseline[metric]
        strats  = others["strategy"].tolist()
        deltas  = [others[others.strategy == s][metric].values[0] - b_val
                   for s in strats]
        colors  = ["#2E7D32" if d >= 0 else "#C62828" for d in deltas]
        bars    = ax.bar(strats, deltas, color=colors,
                         alpha=0.85, edgecolor="white")
        for bar, d in zip(bars, deltas):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    d + (0.003 if d >= 0 else -0.006),
                    f"{d:+.3f}", ha="center",
                    va="bottom" if d >= 0 else "top",
                    fontsize=9.5, fontweight="bold")
        ax.axhline(0, color="black", lw=1.0)
        ax.set_ylabel(f"Δ {metric}", fontsize=11)
        ax.set_title(f"Critical {metric.capitalize()} Δ vs S0 Baseline",
                     fontsize=11, fontweight="bold")
        ax.set_xticklabels([s.replace(" ", "\n") for s in strats], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.98, 0.97, f"Baseline = {b_val:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="gray")

    plt.suptitle("Critical Class Improvement over S0 Baseline",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = out_dir / "fig_s48_delta_critical.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def fig_overall_heatmap(all_met, out_dir):
    """Metric heatmap: all strategies × all overall metrics."""
    metrics = ["auroc", "auprc", "f1_macro", "balanced_acc", "mcc"]
    labels  = ["AUROC", "AUPRC", "F1 Macro", "Bal. Acc", "MCC"]
    df_m    = pd.DataFrame(all_met)
    strats  = df_m["strategy"].tolist()
    vals    = df_m[metrics].values

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(vals, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats, fontsize=10)
    for i in range(len(strats)):
        for j in range(len(labels)):
            ax.text(j, i, f"{vals[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if vals[i, j] > 0.72 else "black")
    ax.set_title("Section 4.8 — Overall Metrics: Imbalance Strategies",
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.025)
    plt.tight_layout()
    path = out_dir / "fig_s48_overall_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. EXCEL / CSV EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def save_results(all_met, pc_df, thr, out_dir):
    """Save Table 11 CSVs and Excel workbook."""

    # CSV 1: overall metrics
    om = pd.DataFrame(all_met)
    om.to_csv(out_dir / "table11_overall_metrics.csv", index=False)
    print(f"  Saved: table11_overall_metrics.csv")

    # CSV 2: per-class metrics
    pc_df.to_csv(out_dir / "table11_imbalance_critical.csv", index=False)
    print(f"  Saved: table11_imbalance_critical.csv")

    # CSV 3: Critical class focus
    pc_df[pc_df.cls == "Critical"].to_csv(
        out_dir / "table11_critical_only.csv", index=False)
    print(f"  Saved: table11_critical_only.csv")

    # Excel workbook
    path = out_dir / "Table11_Section48.xlsx"
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as wr:
            om.to_excel(wr, sheet_name="Overall_Metrics",    index=False)
            pc_df.to_excel(wr, sheet_name="PerClass_Metrics", index=False)
            pc_df[pc_df.cls == "Critical"].to_excel(
                wr, sheet_name="Critical_Focus", index=False)

            # Delta vs baseline
            base = pc_df[(pc_df.strategy == "S0 Baseline") &
                         (pc_df.cls == "Critical")].iloc[0]
            delta_rows = []
            for strat in pc_df["strategy"].unique():
                if strat == "S0 Baseline":
                    continue
                row = pc_df[(pc_df.strategy == strat) &
                            (pc_df.cls == "Critical")]
                if row.empty:
                    continue
                r = row.iloc[0]
                delta_rows.append({
                    "strategy"       : strat,
                    "delta_recall"   : round(r["recall"]    - base["recall"],    4),
                    "delta_precision": round(r["precision"] - base["precision"], 4),
                    "delta_f1"       : round(r["f1"]        - base["f1"],        4),
                    "delta_auprc"    : round(r["auprc"]     - base["auprc"],     4),
                })
            pd.DataFrame(delta_rows).to_excel(
                wr, sheet_name="Delta_vs_S0", index=False)

            # Optimised thresholds
            if thr is not None:
                pd.DataFrame({
                    "class"    : CLASSES,
                    "threshold": thr.round(4),
                }).to_excel(wr, sheet_name="S3_Thresholds", index=False)

            # Cost matrix
            pd.DataFrame(COST_MATRIX,
                         index=CLASSES, columns=CLASSES).to_excel(
                wr, sheet_name="Clinical_Cost_Matrix")

        print(f"  Saved: Table11_Section48.xlsx")
    except ImportError:
        print("  openpyxl not available — skipping Excel export")


# ─────────────────────────────────────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run(csv_path, sig_path, sample, seed, out_path):
    out_dir = Path(out_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # ── Load and engineer ────────────────────────────────────────────────────
    df = load_and_engineer(csv_path, sig_path, sample, seed)

    # ── Split — MUST match main pipeline (same seed, same test_size) ─────────
    print("\n[2/4] Splitting data ...")
    X   = df[FS3].values.astype(np.float32)
    y   = df["severity_score"].values.astype(int)
    idx = np.arange(len(df))

    idx_tr, idx_te = train_test_split(
        idx, test_size=0.20, stratify=y, random_state=seed)
    idx_tr, idx_va = train_test_split(
        idx_tr, test_size=0.15, stratify=y[idx_tr], random_state=seed)

    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_va, y_va = X[idx_va], y[idx_va]
    X_te, y_te = X[idx_te], y[idx_te]

    print(f"  Train : {len(y_tr):,}  Val : {len(y_va):,}  Test : {len(y_te):,}")
    print(f"  Test class distribution : {np.bincount(y_te, minlength=N_CLASSES)}")

    # ── Run five strategies ───────────────────────────────────────────────────
    print("\n[3/4] Running imbalance strategies ...")
    all_met  = []
    all_pc   = []
    pred_dict= {}
    thr      = None

    m0, pc0, p0, pred0, clf0, sc0 = run_s0_baseline(
        X_tr, y_tr, X_va, y_va, X_te, y_te, seed)
    all_met.append(m0); all_pc.append(pc0)
    pred_dict["S0 Baseline"] = pred0

    m1, pc1, p1, pred1 = run_s1_cost_sensitive(
        X_tr, y_tr, X_va, y_va, X_te, y_te, seed)
    all_met.append(m1); all_pc.append(pc1)
    pred_dict["S1 Cost-Sensitive"] = pred1

    m2, pc2, p2, pred2 = run_s2_smote(
        X_tr, y_tr, X_va, y_va, X_te, y_te, seed)
    all_met.append(m2); all_pc.append(pc2)
    pred_dict["S2 SMOTE+ENN"] = pred2

    m3, pc3, p3, pred3, thr = run_s3_threshold(
        X_tr, y_tr, X_va, y_va, X_te, y_te, seed, clf0, sc0)
    all_met.append(m3); all_pc.append(pc3)
    pred_dict["S3 Threshold"] = pred3

    m4, pc4, p4, pred4 = run_s4_cascade(
        X_tr, y_tr, X_va, y_va, X_te, y_te, seed)
    all_met.append(m4); all_pc.append(pc4)
    pred_dict["S4 Cascade"] = pred4

    pc_df = pd.concat(all_pc, ignore_index=True)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[4/4] Generating figures ...")
    fig_critical_focus(pc_df, out_dir)
    fig_perclass_f1(pc_df, out_dir)
    fig_confusion_matrices(pred_dict, y_te, out_dir)
    fig_delta_critical(pc_df, out_dir)
    fig_overall_heatmap(all_met, out_dir)

    # ── Save results ──────────────────────────────────────────────────────────
    save_results(all_met, pc_df, thr, out_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SECTION 4.8 — SUMMARY")
    print("=" * 65)
    res_df = pd.DataFrame(all_met)
    print(res_df[["strategy", "auroc", "auprc", "f1_macro",
                  "balanced_acc", "mcc"]].to_string(index=False))

    print("\n  CRITICAL CLASS BREAKDOWN:")
    crit = pc_df[pc_df.cls == "Critical"][
        ["strategy", "precision", "recall", "f1", "auprc"]]
    print(crit.to_string(index=False))

    best_crit = pc_df[pc_df.cls == "Critical"].sort_values(
        "f1", ascending=False).iloc[0]
    base_crit = pc_df[(pc_df.strategy == "S0 Baseline") &
                      (pc_df.cls == "Critical")].iloc[0]
    delta_f1  = best_crit["f1"] - base_crit["f1"]
    rel_improv = 100 * delta_f1 / base_crit["f1"] if base_crit["f1"] > 0 else 0

    print(f"\n  Best Critical F1   : {best_crit['strategy']}  "
          f"F1={best_crit['f1']:.3f}")
    print(f"  Baseline Critical F1 : {base_crit['f1']:.3f}")
    print(f"  Relative improvement : +{rel_improv:.1f}%")
    print(f"\n  All outputs → {out_dir.resolve()}/")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Section 4.8 — Critical Class Imbalance Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--csv",    default="Neonatal_ADR_Expanded_Dataset.csv",
                    help="Expanded dataset CSV")
    ap.add_argument("--sig",    default="signals_all.csv",
                    help="Signal scores CSV")
    ap.add_argument("--sample", type=int, default=0,
                    help="Stratified sample size (0 = full dataset)")
    ap.add_argument("--seed",   type=int, default=42,
                    help="Random seed — must match main pipeline")
    ap.add_argument("--out",    default="imbalance_results",
                    help="Output directory")
    a = ap.parse_args()
    run(a.csv, a.sig, a.sample, a.seed, a.out)
