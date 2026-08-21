from pathlib import Path
import numpy as np, pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import fit_tuned_prngrc, nrmse

# Fixed, known repo layout: reads repro_folds.py's output from the sibling
# results/ directory (run repro_folds.py first).
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

realized_vol = np.load(RESULTS_DIR / "realized_vol.npy")
with open(RESULTS_DIR / "folds.pkl", "rb") as f:
    all_folds = pickle.load(f)

HORIZON = 20
RNG_SEED = 0
PRNGRC_M = 100

def get_fold(tl, fold):
    start = all_folds[tl][fold]
    train = realized_vol[start:start+tl]
    test = realized_vol[start+tl:start+tl+HORIZON]
    return train, test

for tl, fold, expected in [(1000, 3, 182.68999509598515), (250, 10, 71.24596490717119)]:
    train, test = get_fold(tl, fold)
    mu, sd = train.mean(), train.std()
    sd = sd if sd > 1e-12 else 1.0
    train_n = ((train - mu) / sd).reshape(-1, 1)
    m = fit_tuned_prngrc(train_n, seed=RNG_SEED + fold, M=PRNGRC_M)
    pred = m.predict(HORIZON)[:, 0] * sd + mu
    got = nrmse(test, pred)
    print(f"tl={tl} fold={fold}  expected={expected:.4f}  got={got:.4f}  alpha={m.tuned_ridge_alpha}")
