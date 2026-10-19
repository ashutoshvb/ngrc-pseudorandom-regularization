"""
Ablation B: direct test of "noise-induced stabilization" on the financial series.

Gaussian noise of standard deviation sigma (in units of the training series' own
standard deviation, i.e. the z-scored scale the model sees) is added to the
TRAINING window only; the 20-day test window is the clean realized volatility.
The autonomous rollout starts from the (noisy) last observed window, since that is
all a practitioner would have.  3 independent noise realizations per fold.

Models: fixed absolute alpha=0.01 (the setting that produced the catastrophic
folds), fixed absolute alpha=0.1, validation-tuned (grid 1e-6..1e-1),
and the conditioning-relative rule alpha=1e-5*lambda_max.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ablation_common import *  # noqa

OUT = REPO / "results"
SIGMAS = [0.0, 0.01, 0.03, 0.1, 0.3]
REPS = 3
GRID_ORIG = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

rows = []
for di, ds in enumerate(DATASETS):
    for tl in TRAIN_LENGTHS:
        for fold, train, test in folds(ds, tl):
            mu, sd = train.mean(), train.std()
            sd = sd if sd > 1e-12 else 1.0
            clean = ((train - mu) / sd).reshape(-1, 1)
            for s in SIGMAS:
                for r in range(REPS if s > 0 else 1):
                    rng = np.random.default_rng([di, tl, fold, r, int(round(s * 1000))])
                    tn = clean + s * rng.standard_normal(clean.shape)
                    P = Prepared(tn, fold)

                    def score(alpha):
                        try:
                            return nrmse(test, P.forecast(alpha) * sd + mu)
                        except Exception:
                            return np.nan

                    rows.append(dict(
                        dataset=ds, train_len=tl, fold=fold, sigma=s, rep=r,
                        abs_0_01=score(0.01), abs_0_1=score(0.1),
                        tuned=score(tuned_alpha(tn, fold, GRID_ORIG)),
                        rel_1e5=score(1e-5 * P.lmax)))
    print(ds, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "ablation_noise_runs.csv", index=False)
print("wrote", len(df), "rows")
