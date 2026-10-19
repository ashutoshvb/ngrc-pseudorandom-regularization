"""
Ablation A: is the benefit specific to spectral (lambda_max) scaling, or does any
sufficiently strong regularization do the same?  All 162 non-overlapping folds,
both markets, five training lengths, M=100, same seeds as the main backtest.

Rules compared (per fold):
  rel_lmax_c1e-5         alpha = 1e-5 * lambda_max(F)            (the paper's rule)
  abs_<a>                a fixed absolute alpha for every fold
  rel_trace_c<c>         alpha = c * trace(F)/M                  (mean-eigenvalue scale)
  rel_n_a<a>             alpha = a * n_train                     (scales with sample count only)
  tuned_orig             validation-tuned, grid 1e-6..1e-1       (the paper's "tuned")
  tuned_wide             validation-tuned, grid 1e-6..1e+2       (just widen the grid)
  oracle                 best alpha per fold on the TEST window  (unattainable upper bound)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_common import *  # noqa

OUT = REPO / "results"
OUT.mkdir(exist_ok=True)

ABS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
TRACE_C = [1e-4, 1e-3, 1e-2]          # alpha = c * tr/M ; tr/M ~ lambda_max/M-scale
N_A = [1e-4, 1e-3, 1e-2, 1e-1]        # alpha = a * n
GRID_ORIG = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
GRID_WIDE = GRID_ORIG + [1.0, 10.0, 100.0]
ORACLE_GRID = [10.0 ** e for e in range(-8, 4)]

rows = []
for ds in DATASETS:
    for tl in TRAIN_LENGTHS:
        for fold, train, test in folds(ds, tl):
            mu, sd = train.mean(), train.std()
            sd = sd if sd > 1e-12 else 1.0
            tn = ((train - mu) / sd).reshape(-1, 1)
            seed = fold  # RNG_SEED(0) + fold, as in run_full_pipeline.py
            P = Prepared(tn, seed)
            n = len(tn)

            def score(alpha):
                try:
                    return nrmse(test, P.forecast(alpha) * sd + mu)
                except Exception:
                    return np.nan

            r = dict(dataset=ds, train_len=tl, fold=fold, lmax=P.lmax, lmin=P.lmin, tr=P.tr, n=n)
            r["rel_lmax_c1e-5"] = score(1e-5 * P.lmax)
            for c in [1e-7, 1e-6, 1e-4, 1e-3, 1e-2]:
                r[f"rel_lmax_c{c:g}"] = score(c * P.lmax)
            for a in ABS:
                r[f"abs_{a:g}"] = score(a)
            for c in TRACE_C:
                r[f"rel_trace_c{c:g}"] = score(c * P.tr / M)
            for a in N_A:
                r[f"rel_n_a{a:g}"] = score(a * n)
            r["tuned_orig"] = score(tuned_alpha(tn, seed, GRID_ORIG))
            r["tuned_wide"] = score(tuned_alpha(tn, seed, GRID_WIDE))
            r["oracle"] = np.nanmin([score(a) for a in ORACLE_GRID])
            rows.append(r)
    print(ds, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "ablation_rules_folds.csv", index=False)
print("wrote", OUT / "ablation_rules_folds.csv", len(df), "folds")
