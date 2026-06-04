"""
count_missing_outcomes_v2.py
============================
Fixed for datasets where FAERS outcome flags are stored as individual
binary columns rather than a single concatenated string column.

Detected column schema:
  outcome_died                 → DE (Fatal)
  outcome_life_threatening     → LT (Critical)
  outcome_hospitalized         → HO (Serious)
  outcome_disabled             → DS (Serious)
  outcome_congenital_anomaly   → CA (Serious)
  outcome_required_intervention→ RI (Moderate)
  outcome_other                → OT (Non-Serious)
  outcome_non_serious          → additional Non-Serious flag

Usage
-----
  python count_missing_outcomes_v2.py \
      --csv Neonatal_ADR_Expanded_Dataset.csv \
      --sig signals_all.csv
"""

import argparse, warnings
import pandas as pd
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

# Map binary outcome columns → severity tier
FLAG_COLS = {
    "outcome_died":                  "DE",
    "outcome_life_threatening":      "LT",
    "outcome_hospitalized":          "HO",
    "outcome_disabled":              "DS",
    "outcome_congenital_anomaly":    "CA",
    "outcome_required_intervention": "RI",
    "outcome_other":                 "OT",
    "outcome_non_serious":           "NS",
}

# Columns that represent a genuine (non-missing) outcome
SERIOUS_COLS = [
    "outcome_died", "outcome_life_threatening", "outcome_hospitalized",
    "outcome_disabled", "outcome_congenital_anomaly", "outcome_required_intervention",
]
OT_COLS = ["outcome_other", "outcome_non_serious"]
ALL_OUTCOME_COLS = SERIOUS_COLS + OT_COLS


def count_missing(csv_path: str, sig_path: str):
    print(f"\n{'='*70}")
    print("  MISSING-OUTCOME COUNT  —  MANUSCRIPT SECTION 3.2")
    print(f"{'='*70}\n")

    # ── Load ─────────────────────────────────────────────────────────────────
    print(f"[1/3] Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    N = len(df)
    print(f"      Total expanded rows (full neonatal extract): {N:,}")

    # Confirm which outcome columns are present
    present = [c for c in ALL_OUTCOME_COLS if c in df.columns]
    missing_cols = [c for c in ALL_OUTCOME_COLS if c not in df.columns]
    print(f"      Outcome columns found : {present}")
    if missing_cols:
        print(f"      Outcome columns absent: {missing_cols}")

    # ── Counting ─────────────────────────────────────────────────────────────
    print("\n[2/3] Counting missing-outcome records ...")

    # Treat a column value as "flagged" if it is 1, True, "Yes", "Y", "DE", etc.
    def is_flagged(series: pd.Series) -> pd.Series:
        s = series.fillna(0)
        if s.dtype == bool or set(s.unique()).issubset({0, 1, True, False}):
            return s.astype(bool)
        return s.astype(str).str.strip().str.upper().isin(
            {"1", "TRUE", "YES", "Y", "DE", "LT", "HO", "DS", "CA", "RI", "OT"}
        )

    # (A) ALL outcome columns are zero/null/false → truly missing
    if present:
        all_zero = pd.concat(
            [~is_flagged(df[c]) for c in present], axis=1
        ).all(axis=1)
    else:
        all_zero = pd.Series(True, index=df.index)

    N_missing = int(all_zero.sum())
    pct_missing = 100 * N_missing / N

    # (B) At least one serious flag is set
    if SERIOUS_COLS:
        serious_present = [c for c in SERIOUS_COLS if c in df.columns]
        has_serious = pd.concat(
            [is_flagged(df[c]) for c in serious_present], axis=1
        ).any(axis=1) if serious_present else pd.Series(False, index=df.index)
    else:
        has_serious = pd.Series(False, index=df.index)

    N_has_serious = int(has_serious.sum())

    # (C) OT-only: at least one OT flag but NO serious flag
    ot_present_cols = [c for c in OT_COLS if c in df.columns]
    if ot_present_cols:
        has_ot = pd.concat(
            [is_flagged(df[c]) for c in ot_present_cols], axis=1
        ).any(axis=1)
    else:
        has_ot = pd.Series(False, index=df.index)

    mask_ot_only = has_ot & ~has_serious & ~all_zero
    N_ot_only = int(mask_ot_only.sum())
    pct_ot_only = 100 * N_ot_only / N

    # (D) Excluded from severity-labelled subset = null + OT-only
    N_excluded = N_missing + N_ot_only
    pct_excluded = 100 * N_excluded / N

    # ── Print table ───────────────────────────────────────────────────────────
    W = 58
    print(f"\n  {'Category':<{W}} {'N':>10}   {'%':>7}")
    print(f"  {'-'*78}")
    print(f"  {'Total expanded rows (full neonatal extract)':<{W}} {N:>10,}")
    print(f"  {'(A) All outcome flags null/zero → truly missing':<{W}} {N_missing:>10,}   {pct_missing:>6.2f}%")
    print(f"  {'(B) Has at least one serious flag (DE/LT/HO/DS/CA/RI)':<{W}} {N_has_serious:>10,}   {100*N_has_serious/N:>6.2f}%")
    print(f"  {'(C) OT-only (no serious flag, not null)':<{W}} {N_ot_only:>10,}   {pct_ot_only:>6.2f}%")
    print(f"  {'(D)=(A)+(C) excluded from severity-labelled subset':<{W}} {N_excluded:>10,}   {pct_excluded:>6.2f}%")

    # ── Per-flag breakdown ────────────────────────────────────────────────────
    print(f"\n  Per-flag counts:")
    print(f"  {'Column':<40} {'Code':<6} {'N flagged':>10}   {'%':>7}")
    print(f"  {'-'*68}")
    for col, code in FLAG_COLS.items():
        if col in df.columns:
            n_flag = int(is_flagged(df[col]).sum())
            print(f"  {col:<40} {code:<6} {n_flag:>10,}   {100*n_flag/N:>6.2f}%")

    # ── Severity label distribution (cross-check) ─────────────────────────────
    if "severity_label" in df.columns:
        print(f"\n  Severity label distribution (cross-check):")
        for lbl, cnt in df["severity_label"].value_counts().items():
            print(f"    {lbl:<15} {cnt:>10,}  ({100*cnt/N:.1f}%)")

    # ── Signal-join miss ──────────────────────────────────────────────────────
    print(f"\n[3/3] Signal-join miss rate ...")
    sig = pd.read_csv(sig_path)
    keep = [c for c in ["drug","adr","PRR","ROR","IC","EBGM","n_methods","sig_all4"]
            if c in sig.columns]
    df2 = df.merge(sig[keep], on=["drug","adr"], how="left")
    N_hit  = int(df2["PRR"].notna().sum())
    N_miss = N - N_hit
    print(f"  Signal-matched pairs  : {N_hit:>10,}  ({100*N_hit/N:.1f}%)")
    print(f"  Signal-unmatched (FS2 imputed as no-signal): {N_miss:>10,}  ({100*N_miss/N:.1f}%)")

    # ── Manuscript fill-in ────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  MANUSCRIPT FILL-IN  (copy into Section 3.2)")
    print(f"{'='*70}")
    print(f"\n  Strict definition (null/blank outcome only):")
    print(f"    N = {N_missing:,}; {pct_missing:.1f}% of the neonatal extract")
    print(f"\n  Broader definition (null + OT-only, no serious qualifier):")
    print(f"    N = {N_excluded:,}; {pct_excluded:.1f}% of the neonatal extract")
    print(f"\n  FS2 signal sparsity (for methods / feature engineering section):")
    print(f"    {N_miss:,} unmatched pairs ({100*N_miss/N:.1f}%) imputed as no-signal")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Count missing-outcome records — manuscript Section 3.2",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--csv", required=True, help="Neonatal_ADR_Expanded_Dataset.csv")
    ap.add_argument("--sig", required=True, help="signals_all.csv")
    a = ap.parse_args()
    for p in (a.csv, a.sig):
        if not Path(p).exists():
            raise FileNotFoundError(f"File not found: {p}")
    count_missing(a.csv, a.sig)