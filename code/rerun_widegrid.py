from pathlib import Path
import numpy as np, pandas as pd, pickle, sys, time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import fit_tuned_prngrc, nrmse

# Fixed, known repo layout: reads repro_folds.py's output and writes its own
# output to the sibling results/ directory, resolved relative to this
# script's own location (run repro_folds.py first).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

realized_vol = np.load(RESULTS_DIR / "realized_vol.npy")
with open(RESULTS_DIR / "folds.pkl", "rb") as f:
    all_folds = pickle.load(f)

HORIZON = 20
RNG_SEED = 0
PRNGRC_M = 100
WIDE_GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]

TRAIN_LENS = [60, 120, 250, 500, 1000]

rows = []
t0 = time.time()
for tl in TRAIN_LENS:
    starts = all_folds[tl]
    for fold, start in enumerate(starts):
        train = realized_vol[start:start+tl]
        test = realized_vol[start+tl:start+tl+HORIZON]
        mu, sd = train.mean(), train.std()
        sd = sd if sd > 1e-12 else 1.0
        train_n = ((train - mu) / sd).reshape(-1, 1)
        try:
            m = fit_tuned_prngrc(train_n, seed=RNG_SEED+fold, M=PRNGRC_M, ridge_grid=WIDE_GRID)
            pred = m.predict(HORIZON)[:,0]*sd+mu
            val = nrmse(test, pred)
            alpha = m.tuned_ridge_alpha
        except Exception as e:
            val, alpha = np.nan, np.nan
        rows.append(dict(train_len=tl, fold=fold, nrmse_NGRC_pseudorandom_wide=val, chosen_alpha=alpha))
    print(f"tl={tl} done  elapsed={time.time()-t0:.1f}s", flush=True)

df_new = pd.DataFrame(rows)
out_path = RESULTS_DIR / "sp500_ngrc_pseudorandom_widegrid.csv"
df_new.to_csv(out_path, index=False)
print(f"saved to {out_path}. total elapsed", time.time()-t0)
