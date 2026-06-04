"""
confirm_table4.py
=================
Confirms the N values for Table 4 from signals_all.csv and prints
the revised Table 4 ready for manuscript insertion.

Handles the actual column schema of signals_all.csv:
  n_methods_flagged  →  number of methods that flagged the pair
  signal_ALL_4       →  1 if flagged by all 4 methods
  signal_ANY_3       →  1 if flagged by any 3+ methods
  signal_PRR / signal_ROR / signal_IC / signal_EBGM  →  individual flags

Usage:
  python confirm_table4.py --sig signals_all.csv --csv Neonatal_ADR_Expanded_Dataset.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def detect_columns(sig: pd.DataFrame) -> dict:
    """Auto-detect the actual column names for key fields."""
    print("\n  Columns in signals_all.csv:")
    for c in sig.columns:
        print(f"    {c}")

    col_lower = {c.lower(): c for c in sig.columns}

    def find(candidates):
        for c in candidates:
            if c in sig.columns:
                return c
            if c.lower() in col_lower:
                return col_lower[c.lower()]
        return None

    return {
        "n_methods": find(["n_methods", "n_methods_flagged", "num_methods",
                           "nmethods", "methods_count", "consensus"]),
        "sig_all4":  find(["sig_all4", "signal_ALL_4", "signal_all_4",
                           "all4", "sig_all_4", "signal_all4"]),
        "sig_any3":  find(["signal_ANY_3", "signal_any_3", "sig_any3",
                           "any3", "sig_any_3"]),
        "sig_prr":   find(["signal_PRR", "signal_prr", "sig_prr", "PRR_sig"]),
        "sig_ror":   find(["signal_ROR", "signal_ror", "sig_ror", "ROR_sig"]),
        "sig_ic":    find(["signal_IC",  "signal_ic",  "sig_ic",  "IC_sig"]),
        "sig_ebgm":  find(["signal_EBGM","signal_ebgm","sig_ebgm","EBGM_sig"]),
    }


def derive_n_methods(sig: pd.DataFrame, cols: dict) -> pd.Series:
    """Derive n_methods from individual flag columns if not directly available."""
    if cols["n_methods"] and cols["n_methods"] in sig.columns:
        return sig[cols["n_methods"]].fillna(0).astype(int)

    # Fallback: sum individual signal flags
    flag_cols = [cols[k] for k in ["sig_prr","sig_ror","sig_ic","sig_ebgm"]
                 if cols[k] and cols[k] in sig.columns]
    if flag_cols:
        print(f"  Deriving n_methods from: {flag_cols}")
        return sig[flag_cols].fillna(0).astype(int).sum(axis=1)

    # Last resort: threshold rules
    print("  Deriving n_methods from threshold rules (PRR≥2, ROR≥1, IC>0, EBGM≥2)")
    n = pd.Series(0, index=sig.index)
    if "PRR"  in sig.columns: n += (sig["PRR"]  >= 2).astype(int)
    if "ROR"  in sig.columns: n += (sig["ROR"]  >= 1).astype(int)
    if "IC"   in sig.columns: n += (sig["IC"]   >  0).astype(int)
    if "EBGM" in sig.columns: n += (sig["EBGM"] >= 2).astype(int)
    return n


def confirm_table4(sig_path: str, csv_path: str = None):
    print(f"\n{'='*65}")
    print("  TABLE 4 CONFIRMATION")
    print(f"{'='*65}")

    # ── Load signals ──────────────────────────────────────────────────────────
    sig = pd.read_csv(sig_path)
    print(f"\n  Loaded signals_all.csv: {len(sig):,} rows")

    cols = detect_columns(sig)
    print(f"\n  Detected column mapping:")
    for k, v in cols.items():
        print(f"    {k:<12} → {v if v else '⚠ NOT FOUND'}")

    # Derive n_methods
    sig["_n_methods"] = derive_n_methods(sig, cols)

    # Derive sig_all4
    if cols["sig_all4"] and cols["sig_all4"] in sig.columns:
        sig["_sig_all4"] = sig[cols["sig_all4"]].fillna(0).astype(int)
    else:
        sig["_sig_all4"] = (sig["_n_methods"] >= 4).astype(int)
        print("  Derived sig_all4 from n_methods >= 4")

    # ── Candidate pool ────────────────────────────────────────────────────────
    # signals_all.csv contains one row per unique drug-ADR pair evaluated
    N_candidate = len(sig)

    # If expanded CSV provided, also count unique pairs from there
    N_expanded_unique = None
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path, usecols=["drug", "adr"])
        N_expanded_unique = df.drop_duplicates().shape[0]
        N_total_rows      = len(df)
        print(f"\n  From expanded dataset:")
        print(f"    Total expanded rows    : {N_total_rows:,}")
        print(f"    Unique drug-ADR pairs  : {N_expanded_unique:,}")

    # ── Consensus counts ──────────────────────────────────────────────────────
    flagged_ge1 = int((sig["_n_methods"] >= 1).sum())
    flagged_ge2 = int((sig["_n_methods"] >= 2).sum())
    flagged_ge3 = int((sig["_n_methods"] >= 3).sum())
    flagged_all4= int((sig["_sig_all4"]  == 1).sum())
    not_flagged = N_candidate - flagged_ge1

    # ── Print confirmed N values ──────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  CONFIRMED N VALUES")
    print(f"{'─'*65}")
    print(f"  Total candidate pairs (signals_all.csv rows) : {N_candidate:,}")
    if N_expanded_unique:
        print(f"  Unique pairs in expanded dataset             : {N_expanded_unique:,}")
    print(f"  Flagged by ≥ 1 method                        : {flagged_ge1:,}")
    print(f"  Flagged by ≥ 2 methods                       : {flagged_ge2:,}")
    print(f"  Flagged by ≥ 3 methods                       : {flagged_ge3:,}")
    print(f"  Flagged by all 4 methods                     : {flagged_all4:,}")
    print(f"  Not flagged by any method                    : {not_flagged:,}")

    # ── Table 4 ───────────────────────────────────────────────────────────────
    rows = [
        ("Flagged by ≥ 1 method (any signal)",  flagged_ge1),
        ("Flagged by ≥ 2 methods",               flagged_ge2),
        ("Flagged by ≥ 3 methods",               flagged_ge3),
        ("Flagged by all 4 methods (consensus)", flagged_all4),
        ("Not flagged by any method",            not_flagged),
    ]

    print(f"\n{'─'*65}")
    print("  REVISED TABLE 4 (manuscript-ready)")
    print(f"{'─'*65}")
    print(f"\n  {'Consensus Level':<42} {'N':>7}  "
          f"{'% Flagged':>11}  {'% All Pairs':>11}")
    print(f"  {'(full candidate pool N = '+str(N_candidate)+')':>42}")
    print(f"  {'-'*70}")

    for label, n in rows:
        pct_flagged = (100 * n / flagged_ge1) if label != "Not flagged by any method" else None
        pct_all     = 100 * n / N_candidate
        pf_str = f"{pct_flagged:>10.1f}%" if pct_flagged is not None else f"{'—':>11}"
        print(f"  {label:<42} {n:>7,}  {pf_str}  {pct_all:>10.1f}%")

    # ── Comparison with manuscript's original figures ─────────────────────────
    print(f"\n{'─'*65}")
    print("  COMPARISON WITH ORIGINAL MANUSCRIPT FIGURES")
    print(f"{'─'*65}")
    orig = {"candidate": 124841, "flagged_ge1": 10111, "all4": 6251}
    print(f"  {'Metric':<35} {'Original':>10}  {'Confirmed':>10}  {'Match':>8}")
    print(f"  {'-'*65}")
    checks = [
        ("Total candidate pairs",    orig["candidate"],   N_candidate),
        ("Flagged by ≥ 1 method",    orig["flagged_ge1"], flagged_ge1),
        ("Flagged by all 4 methods", orig["all4"],        flagged_all4),
    ]
    all_match = True
    for label, orig_n, conf_n in checks:
        match = "✓ YES" if orig_n == conf_n else "✗ NO"
        if orig_n != conf_n:
            all_match = False
        print(f"  {label:<35} {orig_n:>10,}  {conf_n:>10,}  {match:>8}")

    print()
    if all_match:
        print("  ✓ All figures match. Original Table 4 N values are confirmed.")
        print("    Update Table 4 by adding the '% All Pairs' column only.")
    else:
        print("  ✗ Figures differ. Use the CONFIRMED values above for Table 4.")
        print("    The original manuscript used a different dataset version.")
        print(f"\n  ACTION REQUIRED: Replace in manuscript:")
        if orig["candidate"] != N_candidate:
            print(f"    '124,841 unique pairs' → '{N_candidate:,} unique pairs'")
        if orig["flagged_ge1"] != flagged_ge1:
            print(f"    '10,111 flagged pairs' → '{flagged_ge1:,} flagged pairs'")
        if orig["all4"] != flagged_all4:
            print(f"    '6,251 all-four-method' → '{flagged_all4:,} all-four-method'")
        # Recompute rates with confirmed figures
        pct_within = 100 * flagged_all4 / flagged_ge1 if flagged_ge1 > 0 else 0
        pct_all    = 100 * flagged_all4 / N_candidate
        print(f"\n  REVISED KEY SENTENCE FOR MANUSCRIPT:")
        print(f"    \"Of {N_candidate:,} unique drug-ADR candidate pairs, "
              f"{flagged_ge1:,} ({100*flagged_ge1/N_candidate:.1f}%) were flagged")
        print(f"    by at least one method; of those flagged pairs, "
              f"{flagged_all4:,} ({pct_within:.1f}% of flagged pairs;")
        print(f"    {pct_all:.1f}% of all candidate pairs) achieved "
              f"all-four-method consensus.\"")

    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Confirm Table 4 N values from signals_all.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--sig", required=True, help="signals_all.csv")
    ap.add_argument("--csv", default=None,
                    help="Neonatal_ADR_Expanded_Dataset.csv (optional, "
                         "for unique-pair cross-check)")
    a = ap.parse_args()
    if not Path(a.sig).exists():
        raise FileNotFoundError(f"Not found: {a.sig}")
    confirm_table4(a.sig, a.csv)