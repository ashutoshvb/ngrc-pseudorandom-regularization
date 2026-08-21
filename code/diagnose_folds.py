"""
Reproduces the mechanistic diagnosis behind Sec. 3.3 (Table 3), and the two
interventions behind Sec. 3.3/Fig. 2 (Table 4: forcing alpha at fixed
M; Table 5: varying M at fixed alpha) -- from source, rather than relying on
the static results CSVs alone.

The three "catastrophic" folds and three matched "stable" folds used in
these tables were originally identified by an exhaustive sweep over every
fold at every training length in the original overlapping-window design
(the 125-fold sweep referenced in Sec. 3.3: "individual-fold NRMSE reached
182.7, 71.2, 53.9, 45.9, and 38.0"). That sweep is reproduced below (see
`_find_diagnostic_folds`) and its result matches the (train_len, fold)
pairs already hard-coded in this script -- rerun the sweep yourself if you
want to confirm this rather than take it on faith.

Run order: `repro_folds.py` first, then this script. Both resolve their
input/output paths relative to their own file location (matching every
other script in code/), so they can be run from anywhere:

    python code/repro_folds.py
    python code/diagnose_folds.py

Writes (to the sibling results/ directory): diagnostic_condition_table.csv,
intervention_force_alpha.csv, intervention_vary_M.csv -- matching the
files of the same name already shipped in results/, so you can diff them
against this script's output.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import fit_tuned_prngrc, PseudorandomNGRC, nrmse

# Fixed, known repo layout: reads repro_folds.py's output from, and writes
# its own output to, the sibling results/ directory, resolved relative to
# this script's own location (matching repro_folds.py/rerun_widegrid.py/
# verify.py). Run repro_folds.py first.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

HORIZON = 20
RNG_SEED = 0
PRNGRC_M = 100

# (label, train_len, fold_index) for the three catastrophic + three matched
# stable folds examined in Sec. 3.3. Identified by exhaustive sweep -- see
# `_find_diagnostic_folds` below.
DIAGNOSTIC_FOLDS = [
    ("blowup", 1000, 3),
    ("blowup", 250, 10),
    ("blowup", 60, 23),
    ("stable", 1000, 0),
    ("stable", 250, 0),
    ("stable", 60, 2),
]


def _get_fold(realized_vol, all_folds, tl, fold_idx):
    start = all_folds[tl][fold_idx]
    train = realized_vol[start:start + tl]
    test = realized_vol[start + tl:start + tl + HORIZON]
    return train, test


def _find_diagnostic_folds(realized_vol, all_folds):
    """Exhaustively sweeps every fold at every training length with the
    validation-tuned pseudorandom-projection NGRC and reports, per training
    length, the worst 5 and best 5 folds by NRMSE. This is what originally
    surfaced the five catastrophic values quoted in Sec. 3.3 and the three
    matched-stable folds used in Table 3; it is not needed to regenerate
    the tables below (DIAGNOSTIC_FOLDS already has the answer) but is
    included so the fold selection itself is reproducible and auditable,
    not just asserted."""
    print("Sweeping all folds (this reproduces the fold selection, not just the tables)...")
    for tl in [60, 120, 250, 500, 1000]:
        row = []
        for fold in range(len(all_folds[tl])):
            train, test = _get_fold(realized_vol, all_folds, tl, fold)
            mu, sd = train.mean(), train.std()
            sd = sd if sd > 1e-12 else 1.0
            train_n = ((train - mu) / sd).reshape(-1, 1)
            try:
                m = fit_tuned_prngrc(train_n, seed=RNG_SEED + fold, M=PRNGRC_M)
                pred = m.predict(HORIZON)[:, 0] * sd + mu
                score = nrmse(test, pred)
            except Exception:
                score = np.nan
            row.append((fold, score))
        row.sort(key=lambda x: x[1] if np.isfinite(x[1]) else -1)
        print(f"  TL={tl:4d}  worst 5: {row[-5:]}")
        print(f"  TL={tl:4d}  best  5: {row[:5]}")


def build_diagnostic_table(realized_vol, all_folds):
    rows = []
    for status, tl, fold in DIAGNOSTIC_FOLDS:
        train, test = _get_fold(realized_vol, all_folds, tl, fold)
        mu, sd = train.mean(), train.std()
        sd = sd if sd > 1e-12 else 1.0
        train_n = ((train - mu) / sd).reshape(-1, 1)

        m = fit_tuned_prngrc(train_n, seed=RNG_SEED + fold, M=PRNGRC_M)
        pred = m.predict(HORIZON)[:, 0] * sd + mu
        score = nrmse(test, pred)

        # Recompute the projected-feature Gram matrix exactly as
        # PseudorandomNGRC.fit does, to report its condition number.
        lin = m._linear_features(train_n[:-1])
        lin_norm = m._normalize(lin)
        feats = m._project(lin_norm)
        G = feats.T @ feats
        eigs = np.linalg.eigvalsh(G)
        lam_max, lam_min = eigs[-1], eigs[0]
        # A numerically negative smallest eigenvalue (floating-point-level
        # rank deficiency) makes the true condition number undefined; floor
        # lam_min at 1e-300 so this shows up as an unambiguous >1e300 flag
        # rather than a spurious finite-but-huge or negative number.
        cond_number = lam_max / max(lam_min, 1e-300)

        rows.append({
            "status": status,
            "train_len": tl,
            "fold": fold,
            "max_to_median": train.max() / np.median(train),
            "skew": skew(train),
            "kurtosis": kurtosis(train),
            "chosen_alpha": m.tuned_ridge_alpha,
            "cond_number": cond_number,
            "W_out_norm": np.linalg.norm(m.W_out),
            "nrmse": score,
        })
    return pd.DataFrame(rows)


def build_intervention_tables(realized_vol, all_folds):
    catastrophic = [(tl, fold) for status, tl, fold in DIAGNOSTIC_FOLDS if status == "blowup"]

    alpha_grid = [1e-6, 1e-4, 1e-2, 1e-1, 1, 10]
    alpha_rows = []
    for tl, fold in catastrophic:
        train, test = _get_fold(realized_vol, all_folds, tl, fold)
        mu, sd = train.mean(), train.std()
        train_n = ((train - mu) / sd).reshape(-1, 1)
        for a in alpha_grid:
            m = PseudorandomNGRC(k=3, skip=1, M=100, ridge_alpha=a, seed=RNG_SEED + fold)
            m.fit(train_n)
            pred = m.predict(HORIZON)[:, 0] * sd + mu
            alpha_rows.append({"train_len": tl, "fold": fold, "alpha": a,
                                "nrmse": nrmse(test, pred)})
    force_alpha_df = pd.DataFrame(alpha_rows)

    M_grid = [5, 10, 20, 50, 100, 200]
    M_rows = []
    for tl, fold in catastrophic:
        train, test = _get_fold(realized_vol, all_folds, tl, fold)
        mu, sd = train.mean(), train.std()
        train_n = ((train - mu) / sd).reshape(-1, 1)
        for Mval in M_grid:
            m = PseudorandomNGRC(k=3, skip=1, M=Mval, ridge_alpha=0.01, seed=RNG_SEED + fold)
            m.fit(train_n)
            pred = m.predict(HORIZON)[:, 0] * sd + mu
            M_rows.append({"train_len": tl, "fold": fold, "M": Mval,
                            "nrmse": nrmse(test, pred)})
    vary_M_df = pd.DataFrame(M_rows)

    return force_alpha_df, vary_M_df


if __name__ == "__main__":
    import pickle

    realized_vol = np.load(RESULTS_DIR / "realized_vol.npy")
    with open(RESULTS_DIR / "folds.pkl", "rb") as f:
        all_folds = pickle.load(f)

    if "--sweep" in sys.argv:
        _find_diagnostic_folds(realized_vol, all_folds)

    diag_df = build_diagnostic_table(realized_vol, all_folds)
    diag_df.to_csv(RESULTS_DIR / "diagnostic_condition_table.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'diagnostic_condition_table.csv'}")
    print(diag_df.to_string(index=False))

    force_alpha_df, vary_M_df = build_intervention_tables(realized_vol, all_folds)
    force_alpha_df.to_csv(RESULTS_DIR / "intervention_force_alpha.csv", index=False)
    vary_M_df.to_csv(RESULTS_DIR / "intervention_vary_M.csv", index=False)
    print(f"wrote {RESULTS_DIR / 'intervention_force_alpha.csv'}, "
          f"{RESULTS_DIR / 'intervention_vary_M.csv'}")
