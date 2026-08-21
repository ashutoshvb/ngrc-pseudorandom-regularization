"""
Sensitivity of the conditioning-relative ridge rule (alpha = c * lambda_max)
to the choice of c, evaluated on the SAME non-overlapping-fold backtest used
for the main results (both markets, all five training lengths), not just the
six diagnostic folds originally used to pick c=1e-5. This is a broader,
out-of-sample check of whether c=1e-5 is a fragile choice.

Usage: python sensitivity_c.py
Reads data/{sp500,nasdaq}_realized_vol.npy
Writes results/sensitivity_c.csv
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import nrmse
from relative_ridge import PseudorandomNGRC_RelRidge

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RNG_SEED = 0
HORIZON = 20
PRNGRC_M = 100
C_GRID = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
TRAIN_LENGTHS = [60, 120, 250, 500, 1000]
DATASETS = ["sp500", "nasdaq"]
CATASTROPHIC_THRESHOLD = 5.0  # NRMSE above this = catastrophic failure, matches paper's usage

rows = []
t0 = time.time()
for dataset in DATASETS:
    realized_vol = np.load(DATA_DIR / f"{dataset}_realized_vol.npy")
    for tl in TRAIN_LENGTHS:
        block = tl + HORIZON
        max_folds = len(realized_vol) // block
        n_folds = min(25, max_folds)
        starts = [i * block for i in range(n_folds)]
        for fold, start in enumerate(starts):
            train = realized_vol[start:start + tl]
            test = realized_vol[start + tl:start + tl + HORIZON]
            mu, sd = train.mean(), train.std()
            sd = sd if sd > 1e-12 else 1.0
            train_n = ((train - mu) / sd).reshape(-1, 1)
            for c in C_GRID:
                try:
                    m = PseudorandomNGRC_RelRidge(k=3, skip=1, M=PRNGRC_M, c=c, seed=RNG_SEED + fold)
                    m.fit(train_n)
                    pred = m.predict(HORIZON)[:, 0] * sd + mu
                    score = nrmse(test, pred)
                except Exception:
                    score = np.nan
                rows.append(dict(dataset=dataset, train_len=tl, fold=fold, c=c, nrmse=score))
    print(f"{dataset} done, elapsed={time.time()-t0:.1f}s")

df = pd.DataFrame(rows)
out_path = RESULTS_DIR / "sensitivity_c.csv"
df.to_csv(out_path, index=False)

summary = df.groupby("c")["nrmse"].agg(
    mean_nrmse="mean",
    median_nrmse="median",
    max_nrmse="max",
    n_folds="count",
)
summary["n_catastrophic"] = df.groupby("c")["nrmse"].apply(lambda s: (s > CATASTROPHIC_THRESHOLD).sum())
summary["frac_catastrophic"] = summary["n_catastrophic"] / summary["n_folds"]
print(summary.to_string())
summary.to_csv(RESULTS_DIR / "sensitivity_c_summary.csv")
print(f"\nWrote {out_path} and {RESULTS_DIR / 'sensitivity_c_summary.csv'}")
print(f"Total elapsed: {time.time()-t0:.1f}s")
