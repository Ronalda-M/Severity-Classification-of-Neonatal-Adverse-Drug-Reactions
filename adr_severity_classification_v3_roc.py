"""
=============================================================================
  ADR Severity Classification — Neonatal Dataset
  Signal-Enriched Feature Ablation Study
  6 Models × Binary + Multiclass × 3 Feature Sets
  ---------------------------------------------------------------------------

  CONTRIBUTIONS
  -------------
  C1  Signal-Enriched Feature Pipeline
      Pharmacovigilance disproportionality scores (PRR, ROR, IC, EBGM,
      n_methods consensus) — computed in the preceding signal detection
      phase — are used as input features to the severity classifier.
      Prior ADR severity works use molecular features or MedDRA codes;
      using computed signal scores is novel in this context.

  C2  Feature Ablation Study (this script)
      Three feature sets are evaluated systematically:
        FS1  Base      — demographic + co-occurrence context features
        FS2  Signal    — pharmacovigilance signal scores only
        FS3  Combined  — FS1 ∪ FS2  (proposed full feature set)
      The ablation quantifies the marginal contribution of signal scores
      over base features, and the interaction between the two groups.

  C3  Neonatal-Specific Signal-Severity Characterisation
      The joined dataset lets us ask: do pairs flagged by all 4 methods
      carry a different severity distribution than weak/no-signal pairs?
      Results are reported as a severity-by-consensus breakdown figure.

  FEATURE SETS
  ------------
  FS1 Base (11 features)
    drug_freq        frequency of drug in dataset (proxy for reporting volume)
    drug_target      mean serious_binary for drug (prior severity rate)
    adr_freq         frequency of ADR in dataset
    adr_target       mean serious_binary for ADR (prior severity rate)
    rfu_freq         frequency of reason_for_use
    sex              Female=0 / Male=1 / missing=-1
    age_days         patient age normalised to days (median-imputed)
    weight_kg        patient weight in kg (median-imputed)
    drug_adr_freq    co-occurrence count of this (drug, adr) pair
    drug_n_adrs      number of distinct ADRs linked to this drug
    adr_n_drugs      number of distinct drugs linked to this ADR

  FS2 Signal (6 features from signals_all.csv)
    log_PRR          log(1 + PRR)   — frequentist disproportionality
    log_ROR          log(1 + ROR)   — odds-ratio disproportionality
    IC               Information Component (Bayesian mutual information)
    log_EBGM         log(1 + EBGM)  — Empirical Bayes geometric mean
    n_methods        0-4: number of methods flagging this pair as signal
    sig_all4         binary: 1 if all 4 methods agree (strong signal)
    NOTE: pairs not in signals_all.csv (n<3 reports) imputed as
          PRR=ROR=EBGM=1, IC=0, n_methods=0, sig_all4=0

  FS3 Combined = FS1 + FS2 (17 features)

  MODELS  (6)
  -----------
  LR        Logistic Regression  (saga, L2, class_weight=balanced)
  RF        Random Forest        (200 trees, balanced_subsample)
  ET        Extra Trees          (200 trees, balanced_subsample)
  HistGBM   Histogram GBM        (sklearn, native NaN support)
  AdaBoost  AdaBoost             (100 estimators)
  MLP       Multi-Layer Perceptron (256-128, early stopping)

  EVALUATION
  ----------
  Stratified 5-fold CV  + held-out 20% test set
  Metrics: AUROC, AUPRC, F1-macro, Precision-macro, Recall-macro, Accuracy

  USAGE
  -----
    python adr_severity_classification_v3.py
    python adr_severity_classification_v3.py \
        --csv  Neonatal_ADR_Expanded_Dataset.csv \
        --sig  signals_all.csv \
        --sample 50000  --folds 5  --out results/

  DEPENDENCIES
  ------------
    pandas  numpy  scikit-learn  matplotlib  seaborn
=============================================================================
"""

import argparse, warnings
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd, seaborn as sns
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

# ── Aesthetics ────────────────────────────────────────────────────────────────
MODEL_COLOURS = {"LR":"#1F4E79","RF":"#2E75B6","ET":"#70AD47",
                 "HistGBM":"#E36C09","AdaBoost":"#7030A0","MLP":"#C00000"}
FS_COLOURS    = {"FS1 Base":"#BDD7EE","FS2 Signal":"#FFE699","FS3 Combined":"#C6EFCE"}
FS_EDGE       = {"FS1 Base":"#2E75B6","FS2 Signal":"#E36C09","FS3 Combined":"#375623"}
SEV_COLOURS   = {"Fatal":"#C00000","Critical":"#E36C09","Serious":"#2E75B6",
                 "Moderate":"#70AD47","Non-Serious":"#888888"}
SCORE_LBL     = {4:"Fatal",3:"Critical",2:"Serious",1:"Moderate",0:"Non-Serious"}
MODELS_ORDER  = ["LR","RF","ET","HistGBM","AdaBoost","MLP"]
FS_ORDER      = ["FS1 Base","FS2 Signal","FS3 Combined"]
DIV = "─"*92

plt.rcParams.update({"figure.facecolor":"white","axes.facecolor":"white",
    "axes.edgecolor":"#CCCCCC","axes.grid":True,"grid.color":"#EBEBEB",
    "grid.linewidth":0.5,"font.size":10,"axes.titlesize":11,"axes.labelsize":10})


# =============================================================================
# 1 ── LOAD & MERGE
# =============================================================================

def load(csv_path, sig_path, sample=None, rs=42):
    print(f"\n[1/6] Loading data ...")
    df  = pd.read_csv(csv_path)
    sig = pd.read_csv(sig_path)
    print(f"      Expanded rows : {len(df):,}")
    print(f"      Signal pairs  : {len(sig):,}")

    # Normalise string keys
    df["drug"]  = df["drug"].astype(str).str.strip()
    df["adr"]   = df["adr"].astype(str).str.strip()
    sig["drug"] = sig["drug"].astype(str).str.strip()
    sig["adr"]  = sig["adr"].astype(str).str.strip()

    # Normalise column names across pipeline versions
    sig = sig.rename(columns={
        "n_methods_flagged": "n_methods", "num_methods": "n_methods",
        "signal_ALL_4": "sig_all4",       "sig_all_4":   "sig_all4",
        "signal_all4":  "sig_all4",       "all4":        "sig_all4",
    })
    # Drop overlapping columns from df to prevent _x/_y suffixes
    for col in ["PRR","ROR","IC","EBGM","n_methods","sig_all4"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if sample and sample < len(df):
        df, _ = train_test_split(df, train_size=sample,
                                 stratify=df["severity_score"], random_state=rs)
        df = df.reset_index(drop=True)
        print(f"      Sampled to    : {len(df):,} (stratified)")

    # Merge signal scores
    sig_cols = ["drug","adr","PRR","ROR","IC","EBGM","n_methods","sig_all4"]
    df = df.merge(sig[sig_cols], on=["drug","adr"], how="left")
    hit = df["PRR"].notna().sum()
    print(f"      Signal join   : {hit:,}/{len(df):,} = {100*hit/len(df):.1f}% pairs matched")
    print(f"\n      Severity distribution:")
    for lbl, cnt in df["severity_label"].value_counts().items():
        print(f"        {lbl:<15} {cnt:>8,}  ({100*cnt/len(df):.1f}%)")
    return df


# =============================================================================
# 2 ── FEATURE ENGINEERING
# =============================================================================

def engineer(df):
    """Build all three feature sets. Returns df with all feature columns."""
    print("[2/6] Engineering features ...")
    age_med = df["age_days"].median()
    wt_med  = df["weight_kg"].median()

    # ── FS1: Base features ────────────────────────────────────────────────
    df = df.copy()
    df = df.join(df.groupby(["drug","adr"]).size().rename("_pf"), on=["drug","adr"])
    df = df.join(df.groupby("drug")["adr"].nunique().rename("_dna"), on="drug")
    df = df.join(df.groupby("adr")["drug"].nunique().rename("_and"), on="adr")

    df["drug_freq"]      = df["drug"].map(df["drug"].value_counts()).fillna(1)
    df["drug_target"]    = df["drug"].map(df.groupby("drug")["serious_binary"].mean()).fillna(0.5)
    df["adr_freq"]       = df["adr"].map(df["adr"].value_counts()).fillna(1)
    df["adr_target"]     = df["adr"].map(df.groupby("adr")["serious_binary"].mean()).fillna(0.5)
    df["rfu_freq"]       = (df["reason_for_use"].fillna("Unknown")
                            .map(df["reason_for_use"].fillna("Unknown").value_counts()).fillna(1))
    df["sex_enc"]        = df["sex"].map({"Female":0,"Male":1}).fillna(-1)
    df["age_days_imp"]   = df["age_days"].fillna(age_med)
    df["weight_kg_imp"]  = df["weight_kg"].fillna(wt_med)
    df["drug_adr_freq"]  = df["_pf"].fillna(1)
    df["drug_n_adrs"]    = df["_dna"].fillna(1)
    df["adr_n_drugs"]    = df["_and"].fillna(1)

    # ── FS2: Signal features (impute missing = no signal) ─────────────────
    df["log_PRR"]   = np.log1p(df["PRR"].fillna(1))
    df["log_ROR"]   = np.log1p(df["ROR"].fillna(1))
    df["IC_imp"]    = df["IC"].fillna(0.0)
    df["log_EBGM"]  = np.log1p(df["EBGM"].fillna(1))
    df["n_methods"] = df["n_methods"].fillna(0).astype(int)
    df["sig_all4"]  = df["sig_all4"].fillna(False).astype(int)

    FS1 = ["drug_freq","drug_target","adr_freq","adr_target","rfu_freq",
           "sex_enc","age_days_imp","weight_kg_imp",
           "drug_adr_freq","drug_n_adrs","adr_n_drugs"]
    FS2 = ["log_PRR","log_ROR","IC_imp","log_EBGM","n_methods","sig_all4"]
    FS3 = FS1 + FS2

    feature_sets = {"FS1 Base": FS1, "FS2 Signal": FS2, "FS3 Combined": FS3}
    print(f"      FS1 Base    : {len(FS1)} features")
    print(f"      FS2 Signal  : {len(FS2)} features")
    print(f"      FS3 Combined: {len(FS3)} features")
    return df, feature_sets


# =============================================================================
# 3 ── MODELS
# =============================================================================

def get_models(rs=42):
    return {
        "LR":      Pipeline([("s",StandardScaler()),("c",LogisticRegression(
                       max_iter=500,C=1.0,class_weight="balanced",
                       solver="saga",n_jobs=-1,random_state=rs))]),
        "RF":      Pipeline([("s",StandardScaler()),("c",RandomForestClassifier(
                       n_estimators=200,min_samples_leaf=3,
                       class_weight="balanced_subsample",n_jobs=-1,random_state=rs))]),
        "ET":      Pipeline([("s",StandardScaler()),("c",ExtraTreesClassifier(
                       n_estimators=200,min_samples_leaf=3,
                       class_weight="balanced_subsample",n_jobs=-1,random_state=rs))]),
        "HistGBM": Pipeline([("s",StandardScaler()),("c",HistGradientBoostingClassifier(
                       max_iter=200,learning_rate=0.05,max_leaf_nodes=63,
                       min_samples_leaf=10,random_state=rs))]),
        "AdaBoost":Pipeline([("s",StandardScaler()),("c",AdaBoostClassifier(
                       n_estimators=100,learning_rate=0.5,random_state=rs))]),
        "MLP":     Pipeline([("s",StandardScaler()),("c",MLPClassifier(
                       hidden_layer_sizes=(256,128),activation="relu",solver="adam",
                       max_iter=200,early_stopping=True,validation_fraction=0.1,
                       n_iter_no_change=10,random_state=rs))]),
    }


# =============================================================================
# 4 ── METRICS
# =============================================================================

def compute_metrics(model, Xtr, ytr, Xvl, yvl, task):
    model.fit(Xtr, ytr)
    yp  = model.predict(Xvl)
    ypr = model.predict_proba(Xvl)
    avg = "binary" if task=="binary" else "macro"
    is_bin = task=="binary"
    if is_bin:
        auroc = roc_auc_score(yvl, ypr[:,1])
        auprc = average_precision_score(yvl, ypr[:,1])
    else:
        auroc = roc_auc_score(yvl,ypr,multi_class="ovr",
                              average="macro",labels=np.unique(ytr))
        avs   = [average_precision_score((yvl==c).astype(int),ypr[:,i])
                 for i,c in enumerate(np.unique(ytr)) if (yvl==c).sum()>0]
        auprc = float(np.mean(avs)) if avs else np.nan
    return {"auroc":auroc,"auprc":auprc,
            "f1":f1_score(yvl,yp,average=avg,zero_division=0),
            "precision":precision_score(yvl,yp,average=avg,zero_division=0),
            "recall":recall_score(yvl,yp,average=avg,zero_division=0),
            "accuracy":accuracy_score(yvl,yp),
            "y_true":yvl,"y_pred":yp,"y_proba":ypr}

def summarise(folds):
    keys = [k for k in folds[0] if k not in ("y_true","y_pred","y_proba")]
    return {k:{"mean":float(np.mean([f[k] for f in folds])),
               "ci":float(1.96*np.std([f[k] for f in folds])/np.sqrt(len(folds))),
               "folds":[f[k] for f in folds]} for k in keys}


# =============================================================================
# 5 ── CROSS-VALIDATION
# =============================================================================

def run_cv(Xs, y, feature_sets, task, n_splits=5, rs=42):
    """
    Xs: dict {fs_name: X_array}
    Returns nested dict: {fs_name: {model_name: summarised_cv_metrics}}
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rs)
    results = {fs:{m:[] for m in MODELS_ORDER} for fs in FS_ORDER}

    for fi, (ti, vi) in enumerate(skf.split(Xs[FS_ORDER[0]], y)):
        print(f"    Fold {fi+1}/{n_splits} ", end="", flush=True)
        ytr, yvl = y[ti], y[vi]
        for fs_name in FS_ORDER:
            Xtr, Xvl = Xs[fs_name][ti], Xs[fs_name][vi]
            for mname in MODELS_ORDER:
                m = get_models(rs)[mname]
                results[fs_name][mname].append(
                    compute_metrics(m, Xtr, ytr, Xvl, yvl, task))
                print(".", end="", flush=True)
        print(" done")

    return {fs:{m:summarise(results[fs][m]) for m in MODELS_ORDER} for fs in FS_ORDER}


def run_test(Xs_tr, Xs_te, ytr, yte, task, rs=42):
    results = {fs:{m:{} for m in MODELS_ORDER} for fs in FS_ORDER}
    for fs_name in FS_ORDER:
        for mname in MODELS_ORDER:
            m = get_models(rs)[mname]
            results[fs_name][mname] = compute_metrics(
                m, Xs_tr[fs_name], ytr, Xs_te[fs_name], yte, task)
    return results


# =============================================================================
# 6 ── CONSOLE OUTPUT
# =============================================================================

METRIC_LBLS = ["auroc","auprc","f1","precision","recall","accuracy"]

def print_table(cv, task, fs_name, n_splits):
    tlbl = "Binary" if task=="binary" else "Multiclass"
    print(f"\n{DIV}")
    print(f"  {tlbl} Severity — {fs_name} — {n_splits}-fold CV")
    print(DIV)
    print(f"  {'Model':<10}  {'AUROC':>12}  {'AUPRC':>12}  {'F1-macro':>12}  "
          f"{'Precision':>12}  {'Recall':>12}  {'Accuracy':>12}")
    print(DIV)
    for mname in MODELS_ORDER:
        res = cv[fs_name][mname]
        row = [f"  {mname:<10}"]
        for m in METRIC_LBLS:
            row.append(f"  {res[m]['mean']:.3f}±{res[m]['ci']:.3f}")
        print("".join(row))
    print(DIV)
    print("  mean ± 95% CI across CV folds\n")

def save_csv(cv, out, fname):
    rows = []
    for fs in FS_ORDER:
        for mname in MODELS_ORDER:
            row = {"feature_set":fs, "model":mname}
            for m,s in cv[fs][mname].items():
                row[f"{m}_mean"] = round(s["mean"],4)
                row[f"{m}_ci95"] = round(s["ci"],4)
                for fi,v in enumerate(s["folds"]): row[f"{m}_fold{fi+1}"] = round(v,4)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out/fname, index=False)
    print(f"      → {fname}")


# =============================================================================
# 7 ── FIGURES
# =============================================================================

def fig_signal_severity(df, out):
    """
    C3: Severity distribution broken down by pharmacovigilance consensus
    (n_methods: 0 = no signal detected … 4 = all methods agree).
    """
    df2 = df.copy()
    df2["consensus"] = df2["n_methods"].fillna(0).astype(int)
    order_sev = ["Fatal","Critical","Serious","Moderate","Non-Serious"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: stacked bar — severity % by consensus level
    ax = axes[0]
    groups = df2.groupby(["consensus","severity_label"]).size().unstack(fill_value=0)
    groups = groups.reindex(columns=[s for s in order_sev if s in groups.columns])
    groups_pct = groups.div(groups.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(groups_pct))
    xs = groups_pct.index.tolist()
    for sev in groups_pct.columns:
        vals = groups_pct[sev].values
        ax.bar(xs, vals, bottom=bottom, label=sev,
               color=SEV_COLOURS.get(sev,"#888"), edgecolor="white",
               linewidth=0.7, alpha=0.88)
        bottom += vals
    ax.set_xticks(xs)
    ax.set_xticklabels([f"n_methods={x}\n({groups.loc[x].sum():,} pairs)" for x in xs],
                       fontsize=8.5)
    ax.set_xlabel("Pharmacovigilance Consensus (# methods flagging pair)")
    ax.set_ylabel("Percentage of Drug-ADR Pairs")
    ax.set_title("Severity Distribution by Multi-Method\nPharmacovigilance Consensus (C3)",
                 fontweight="bold")
    ax.set_ylim(0, 103)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # Right: Fatal+Critical rate by consensus
    ax2 = axes[1]
    fatal_crit = (df2.groupby("consensus")
                  .apply(lambda g: 100*(g["severity_label"].isin(["Fatal","Critical"])).mean())
                  .reset_index(name="fatal_crit_pct"))
    counts = df2.groupby("consensus").size().reset_index(name="n")
    fatal_crit = fatal_crit.merge(counts, on="consensus")
    clrs = [MODEL_COLOURS.get("HistGBM","#E36C09")] * len(fatal_crit)
    bars = ax2.bar(fatal_crit["consensus"], fatal_crit["fatal_crit_pct"],
                   color=clrs, edgecolor="white", linewidth=1.1,
                   alpha=0.88, width=0.6)
    for bar, row in zip(bars, fatal_crit.itertuples()):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f"{row.fatal_crit_pct:.1f}%\n(n={row.n:,})",
                 ha="center", fontsize=8.5)
    ax2.set_xlabel("Pharmacovigilance Consensus (# methods flagging pair)")
    ax2.set_ylabel("Fatal + Critical Rate (%)")
    ax2.set_title("Fatal + Critical Rate by\nPharmacovigilance Consensus (C3)",
                  fontweight="bold")
    ax2.set_xticks(fatal_crit["consensus"])
    ax2.set_ylim(0, fatal_crit["fatal_crit_pct"].max() * 1.22)

    fig.suptitle("Signal Strength → Severity Relationship — Neonatal ADR Dataset\n"
                 "Contribution C3: Pairs flagged by more methods are not uniformly more severe,\n"
                 "validating that signal detection and severity prediction are complementary tasks",
                 fontweight="bold", fontsize=10, y=1.03)
    plt.tight_layout()
    fig.savefig(out/"fig_signal_severity.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓  fig_signal_severity.png")


def fig_ablation_heatmap(cv_results, task, metric, out):
    """
    Heatmap: rows = models, cols = feature sets, cells = mean metric.
    Best cell per row highlighted.
    """
    tlbl = "Binary" if task=="binary" else "Multiclass"
    data = pd.DataFrame(
        {fs: {m: cv_results[fs][m][metric]["mean"] for m in MODELS_ORDER}
         for fs in FS_ORDER}
    ).reindex(MODELS_ORDER)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data.values, cmap="YlGnBu",
                   vmin=data.values.min()-0.02, vmax=data.values.max()+0.01,
                   aspect="auto")
    plt.colorbar(im, ax=ax, label=metric.upper())
    ax.set_xticks(range(len(FS_ORDER)));  ax.set_xticklabels(FS_ORDER, fontsize=10)
    ax.set_yticks(range(len(MODELS_ORDER))); ax.set_yticklabels(MODELS_ORDER, fontsize=10)

    for i, m in enumerate(MODELS_ORDER):
        best_col = int(np.argmax([data.loc[m,fs] for fs in FS_ORDER]))
        for j, fs in enumerate(FS_ORDER):
            v   = data.loc[m, fs]
            ci  = cv_results[fs][m][metric]["ci"]
            bold= (j == best_col)
            clr = "white" if v > (data.values.min() + 0.6*(data.values.max()-data.values.min())) else "black"
            ax.text(j, i, f"{'★ ' if bold else ''}{v:.3f}\n±{ci:.3f}",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold" if bold else "normal",
                    color=clr)

    ax.set_title(f"{metric.upper()} — {tlbl} Severity — Feature Set Ablation\n"
                 "★ = best feature set per model  |  values: mean ± 95% CI",
                 fontweight="bold")
    ax.set_xlabel("Feature Set"); ax.set_ylabel("Model")
    plt.tight_layout()
    fig.savefig(out/f"fig_ablation_{metric}_{task}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓  fig_ablation_{metric}_{task}.png")


def fig_grouped_bar(cv_results, task, metric, out):
    """
    Grouped bars: x=models, groups=feature sets, with CI error bars.
    """
    tlbl = "Binary" if task=="binary" else "Multiclass"
    x    = np.arange(len(MODELS_ORDER))
    n    = len(FS_ORDER); w = 0.26

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, fs in enumerate(FS_ORDER):
        means = [cv_results[fs][m][metric]["mean"] for m in MODELS_ORDER]
        cis   = [cv_results[fs][m][metric]["ci"]   for m in MODELS_ORDER]
        offset = (i - (n-1)/2) * w
        ax.bar(x + offset, means, w, yerr=cis, capsize=4,
               label=fs, color=FS_COLOURS[fs], edgecolor=FS_EDGE[fs],
               linewidth=1.2, alpha=0.9,
               error_kw={"elinewidth":1.2,"ecolor":"#333333"})
        for xi, (mv, ci) in enumerate(zip(means, cis)):
            ax.text(xi + offset, mv + ci + 0.002, f"{mv:.3f}",
                    ha="center", fontsize=7, rotation=55)

    ax.set_xticks(x); ax.set_xticklabels(MODELS_ORDER, fontsize=10)
    mlbl = metric.upper()
    ax.set_title(f"{mlbl} — {tlbl} Severity — Feature Set Comparison\n"
                 "(error bars = 95% CI across CV folds)",
                 fontweight="bold")
    ax.set_ylabel(mlbl)
    all_means = [cv_results[fs][m][metric]["mean"]
                 for fs in FS_ORDER for m in MODELS_ORDER]
    ax.set_ylim(max(0, min(all_means)-0.08), min(1.05, max(all_means)+0.10))
    ax.legend(loc="lower right", framealpha=0.92)
    plt.tight_layout()
    fig.savefig(out/f"fig_bar_{metric}_{task}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓  fig_bar_{metric}_{task}.png")


def fig_roc(test_res, task, out):
    if task != "binary": return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, fs in zip(axes, FS_ORDER):
        for mname in MODELS_ORDER:
            r = test_res[fs][mname]
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_proba"][:,1])
            ax.plot(fpr, tpr, lw=2, color=MODEL_COLOURS[mname],
                    label=f"{mname} ({r['auroc']:.3f})")
        ax.plot([0,1],[0,1],"k--",lw=1)
        ax.set_title(f"{fs}", fontweight="bold")
        ax.set_xlabel("FPR")
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("TPR")
    fig.suptitle("ROC Curves — Binary Severity | One panel per Feature Set",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out/"fig_roc_binary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓  fig_roc_binary.png")


def fig_roc_multiclass(test_res, out):
    """
    Two-figure multiclass ROC (One-vs-Rest).

    Figure A — fig_roc_multiclass_A.png
        3 panels (one per feature set).
        Each panel: OvR ROC curve for every class using the best model
        (highest macro-AUROC on that feature set), plus micro- and
        macro-average curves.

    Figure B — fig_roc_multiclass_B.png
        5 panels (one per severity class) for FS3 Combined only.
        Each panel: OvR ROC curve for every model, so the reader can
        compare model performance on the hardest classes (Critical, Fatal).
    """
    from sklearn.preprocessing import label_binarize

    # Colour palette: one colour per severity class
    N_CLASSES  = 5
    CLASS_LBLS = ["Non-Serious", "Moderate", "Serious", "Critical", "Fatal"]
    # Clinically motivated colours: grey → green → blue → orange → red
    CLASS_CLRS = ["#757575", "#2E7D32", "#1565C0", "#EF6C00", "#C62828"]

    # ── FIGURE A: best model per FS, all classes ──────────────────────────
    fig_a, axes_a = plt.subplots(1, 3, figsize=(17, 5.8), sharey=True)

    for ax, fs in zip(axes_a, FS_ORDER):
        # Identify best model for this feature set (highest macro-AUROC)
        best_m = max(MODELS_ORDER,
                     key=lambda m: test_res[fs][m]["auroc"])
        r       = test_res[fs][best_m]
        y_true  = r["y_true"]
        y_proba = r["y_proba"]          # (n_test, n_classes)
        classes = np.unique(y_true)

        # Binarise true labels  (n_test, n_classes)
        Y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))

        # ── Per-class OvR curves ──────────────────────────────────────────
        fpr_all, tpr_all, auc_all = {}, {}, {}
        for i in range(N_CLASSES):
            if i not in classes:        # class absent from test fold
                continue
            fpr_all[i], tpr_all[i], _ = roc_curve(Y_bin[:, i], y_proba[:, i])
            auc_all[i] = roc_auc_score(Y_bin[:, i], y_proba[:, i])
            ax.plot(fpr_all[i], tpr_all[i],
                    color=CLASS_CLRS[i], lw=2.0,
                    label=f"{CLASS_LBLS[i]}  AUC={auc_all[i]:.3f}")

        # ── Macro-average curve (mean of interpolated per-class curves) ───
        all_fpr   = np.unique(np.concatenate(
            [fpr_all[i] for i in fpr_all]))
        mean_tpr  = np.zeros_like(all_fpr)
        for i in fpr_all:
            mean_tpr += np.interp(all_fpr, fpr_all[i], tpr_all[i])
        mean_tpr /= len(fpr_all)
        macro_auc = np.mean(list(auc_all.values()))
        ax.plot(all_fpr, mean_tpr, color="black", lw=2.2, ls="--",
                label=f"Macro avg  AUC={macro_auc:.3f}")

        # ── Micro-average curve (flatten all classes) ─────────────────────
        fpr_micro, tpr_micro, _ = roc_curve(
            Y_bin[:, list(fpr_all.keys())].ravel(),
            y_proba[:, list(fpr_all.keys())].ravel())
        micro_auc = roc_auc_score(
            Y_bin[:, list(fpr_all.keys())],
            y_proba[:, list(fpr_all.keys())],
            average="micro")
        ax.plot(fpr_micro, tpr_micro, color="black", lw=2.2, ls=":",
                label=f"Micro avg  AUC={micro_auc:.3f}")

        ax.plot([0,1],[0,1],"silver", lw=1, ls="--")
        ax.set_title(f"{fs}\nBest model: {best_m}", fontweight="bold",
                     fontsize=10)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.25)

    axes_a[0].set_ylabel("True Positive Rate", fontsize=10)
    fig_a.suptitle(
        "Multiclass ROC Curves (One-vs-Rest) — Best Model per Feature Set\n"
        "Micro & Macro averages shown as dashed/dotted black lines",
        fontweight="bold", fontsize=11)
    plt.tight_layout()
    fig_a.savefig(out/"fig_roc_multiclass_A.png", dpi=150,
                  bbox_inches="tight")
    plt.close(fig_a)
    print("    ✓  fig_roc_multiclass_A.png")

    # ── FIGURE B: FS3 Combined — per-class panel, all models ─────────────
    fs3      = "FS3 Combined"
    n_panels = N_CLASSES
    fig_b, axes_b = plt.subplots(1, n_panels,
                                 figsize=(4.2 * n_panels, 5.2), sharey=True)

    for ax, cls_idx in zip(axes_b, range(N_CLASSES)):
        cls_name = CLASS_LBLS[cls_idx]
        ax.plot([0,1],[0,1],"silver", lw=1, ls="--")

        for mname in MODELS_ORDER:
            r      = test_res[fs3][mname]
            y_true = r["y_true"]
            y_proba= r["y_proba"]

            if cls_idx not in np.unique(y_true):
                continue

            Y_bin_col = (y_true == cls_idx).astype(int)
            fpr, tpr, _ = roc_curve(Y_bin_col, y_proba[:, cls_idx])
            auc = roc_auc_score(Y_bin_col, y_proba[:, cls_idx])
            ax.plot(fpr, tpr, lw=2.0, color=MODEL_COLOURS[mname],
                    label=f"{mname}  {auc:.3f}")

        # Prevalence baseline
        r_ref  = test_res[fs3][MODELS_ORDER[0]]
        prev   = (r_ref["y_true"] == cls_idx).mean()
        ax.axhline(prev, color="dimgray", lw=0.8, ls=":",
                   label=f"Prevalence={prev:.3f}")

        ax.set_title(f"{cls_name}", fontweight="bold",
                     color=CLASS_CLRS[cls_idx], fontsize=11)
        ax.set_xlabel("False Positive Rate", fontsize=9)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.25)

    axes_b[0].set_ylabel("True Positive Rate", fontsize=10)
    fig_b.suptitle(
        "Multiclass ROC Curves (One-vs-Rest) — FS3 Combined | One Panel per Severity Class\n"
        "All 6 models compared; dotted line = class prevalence baseline",
        fontweight="bold", fontsize=11)
    plt.tight_layout()
    fig_b.savefig(out/"fig_roc_multiclass_B.png", dpi=150,
                  bbox_inches="tight")
    plt.close(fig_b)
    print("    ✓  fig_roc_multiclass_B.png")


def fig_cm(test_res, task, classes, out):
    """2×3 confusion matrices — best model per feature set."""
    lmap = ({0:"Non-Serious",1:"Serious"} if task=="binary" else SCORE_LBL)
    cls_lbls = [lmap.get(c,str(c)) for c in sorted(classes)]
    n = len(cls_lbls); fw = max(15, n*3.2)

    fig, axes = plt.subplots(1, 3, figsize=(fw, max(5, n*1.4)))
    for ax, fs in zip(axes, FS_ORDER):
        best = max(MODELS_ORDER, key=lambda m: test_res[fs][m]["auroc"])
        r    = test_res[fs][best]
        yt   = [lmap.get(v,v) for v in r["y_true"]]
        yp   = [lmap.get(v,v) for v in r["y_pred"]]
        cm   = confusion_matrix(yt, yp, labels=cls_lbls)
        cmn  = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(1)
        sns.heatmap(cmn, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=cls_lbls, yticklabels=cls_lbls, ax=ax,
                    cbar_kws={"label":"Proportion"}, vmin=0, vmax=1)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{fs}\nBest: {best} (AUROC={r['auroc']:.3f})",
                     fontweight="bold", fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

    tlbl = "Binary" if task=="binary" else "Multiclass"
    fig.suptitle(f"Normalised Confusion Matrices — {tlbl} Severity\n"
                 "Best model per Feature Set | Test Set",
                 fontweight="bold", fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(out/f"fig_cm_{task}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓  fig_cm_{task}.png")


def fig_feature_importance(df, feature_sets, Xs, y_bin, out, rs=42):
    """
    RF feature importance for FS3 Combined — binary task.
    Bars coloured by feature set membership (FS1 vs FS2).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    X = Xs["FS3 Combined"]
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                n_jobs=-1, random_state=rs)
    rf.fit(StandardScaler().fit_transform(X), y_bin)
    imp   = rf.feature_importances_
    names = feature_sets["FS3 Combined"]
    fs1_set = set(feature_sets["FS1 Base"])

    order = np.argsort(imp)
    fig, ax = plt.subplots(figsize=(10, 7))
    clrs = [FS_EDGE["FS1 Base"] if names[i] in fs1_set else FS_EDGE["FS2 Signal"]
            for i in order]
    bars = ax.barh(range(len(order)), imp[order], color=clrs,
                   edgecolor="white", height=0.7, alpha=0.88)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order], fontsize=9)
    ax.set_xlabel("RF Gini Importance")
    ax.set_title("Feature Importance — FS3 Combined (RF, Binary Severity)\n"
                 "Blue = Base features  |  Orange = Signal features (C1 contribution)",
                 fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=FS_EDGE["FS1 Base"], label="FS1 Base features"),
                       Patch(color=FS_EDGE["FS2 Signal"], label="FS2 Signal features (novel)")],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out/"fig_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    ✓  fig_feature_importance.png")


def fig_cv_boxplots(cv_results, task, out):
    """AUROC boxplots: rows = feature sets, cols = models."""
    tlbl = "Binary" if task=="binary" else "Multiclass"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, fs in zip(axes, FS_ORDER):
        data  = [cv_results[fs][m]["auroc"]["folds"] for m in MODELS_ORDER]
        clrs  = [MODEL_COLOURS[m] for m in MODELS_ORDER]
        bp    = ax.boxplot(data, patch_artist=True, widths=0.5,
                           medianprops={"color":"black","linewidth":2})
        for patch, c in zip(bp["boxes"], clrs):
            patch.set_facecolor(c); patch.set_alpha(0.75)
        for i, d in enumerate(data):
            ax.scatter([i+1]*len(d), d, color=clrs[i], s=18, alpha=0.5, zorder=3)
        ax.set_xticklabels(MODELS_ORDER, fontsize=9)
        ax.set_title(fs, fontweight="bold")
        ax.set_xlabel("Model"); ax.set_ylabel("AUROC")
        lo = max(0, min(min(d) for d in data)-0.03)
        ax.set_ylim(lo, 1.02)
    fig.suptitle(f"CV AUROC Distribution — {tlbl} Severity | One panel per Feature Set",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(out/f"fig_cv_boxplots_{task}.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    ✓  fig_cv_boxplots_{task}.png")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run(csv_path, sig_path, sample=None, n_folds=5,
        out_dir="severity_v3", rs=42):

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("  ADR SEVERITY — SIGNAL-ENRICHED FEATURE ABLATION (C1/C2/C3)")
    print(f"{'='*70}")

    # Load & merge
    df = load(csv_path, sig_path, sample, rs)

    # Feature engineering
    df, feature_sets = engineer(df)

    # Signal-severity relationship figure (C3)
    print("[3/6] Signal-severity characterisation figure ...")
    fig_signal_severity(df, out)

    # Train/test split
    y_bin   = df["serious_binary"].values
    y_multi = df["severity_score"].values
    Xs_all  = {fs: df[cols].values.astype(float) for fs, cols in feature_sets.items()}

    splits = train_test_split(*([Xs_all[fs] for fs in FS_ORDER] + [y_bin, y_multi]),
                               test_size=0.2, stratify=y_bin, random_state=rs)
    # Unpack: 3 feature sets × 2 (train/test) + 2 targets × 2 = 8 arrays
    n = len(FS_ORDER)
    Xs_tr = {fs: splits[2*i]   for i, fs in enumerate(FS_ORDER)}
    Xs_te = {fs: splits[2*i+1] for i, fs in enumerate(FS_ORDER)}
    yb_tr, yb_te   = splits[2*n],     splits[2*n+1]
    ym_tr, ym_te   = splits[2*n+2],   splits[2*n+3]

    print(f"\n      Train: {len(yb_tr):,}  |  Test: {len(yb_te):,}")

    # Feature importance (C1 illustration)
    print("[4/6] Feature importance ...")
    fig_feature_importance(df, feature_sets, Xs_tr, yb_tr, out, rs)

    # Cross-validation
    print(f"\n[5/6] Cross-validation ({n_folds} folds) ...")
    all_cv   = {}
    all_test = {}

    for task, ytr, yte in [("binary", yb_tr, yb_te), ("multiclass", ym_tr, ym_te)]:
        tlbl = "Binary" if task=="binary" else "Multiclass"
        print(f"\n  ── Task: {tlbl} ({len(np.unique(ytr))} classes) ──")
        cv   = run_cv(Xs_tr, ytr, feature_sets, task, n_folds, rs)
        test = run_test(Xs_tr, Xs_te, ytr, yte, task, rs)
        all_cv[task]   = cv
        all_test[task] = test

        for fs in FS_ORDER:
            print_table(cv, task, fs, n_folds)
        save_csv(cv, out, f"cv_results_{task}.csv")

    # Save test results
    rows = []
    for task, tres in all_test.items():
        for fs in FS_ORDER:
            for mname in MODELS_ORDER:
                r = tres[fs][mname]
                rows.append({"task":task,"feature_set":fs,"model":mname,
                             "auroc":round(r["auroc"],4),"auprc":round(r["auprc"],4),
                             "f1":round(r["f1"],4),"precision":round(r["precision"],4),
                             "recall":round(r["recall"],4),"accuracy":round(r["accuracy"],4)})
    pd.DataFrame(rows).to_csv(out/"test_results.csv", index=False)
    print("      → test_results.csv")

    # Figures
    print("\n[6/6] Generating figures ...")
    for task, cv, test in [("binary",  all_cv["binary"],  all_test["binary"]),
                            ("multiclass", all_cv["multiclass"], all_test["multiclass"])]:
        for m in ["auroc","f1","recall"]:
            fig_ablation_heatmap(cv, task, m, out)
            fig_grouped_bar(cv, task, m, out)
        fig_cv_boxplots(cv, task, out)
        cls = np.unique(yb_te if task=="binary" else ym_te)
        fig_cm(test, task, cls, out)

    fig_roc(all_test["binary"], "binary", out)
    fig_roc_multiclass(all_test["multiclass"], out)

    print(f"\n  All outputs → {out}")
    print("  Pipeline complete.\n")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ADR Severity — Signal-Enriched Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv",    default="Neonatal_ADR_Expanded_Dataset.csv")
    parser.add_argument("--sig",    default="signals_all.csv")
    parser.add_argument("--sample", type=int, default=50000)
    parser.add_argument("--folds",  type=int, default=5)
    parser.add_argument("--out",    default="severity_v3")
    a = parser.parse_args()
    run(a.csv, a.sig, a.sample, a.folds, a.out)
