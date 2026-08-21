from pathlib import Path
import numpy as np
import pandas as pd
from arch.data import sp500

# Fixed, known repo layout: outputs go to the sibling results/ directory,
# resolved relative to this script's own location.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RNG_SEED = 0
np.random.seed(RNG_SEED)

df = sp500.load()
price = df["Adj Close"].astype(float).values
logret = np.diff(np.log(price))
WIN = 20
realized_vol = pd.Series(np.abs(logret)).rolling(WIN).mean().dropna().values

HORIZON = 20
N_FOLDS = 25
TRAIN_LENS = [60, 120, 250, 500, 1000]

rng = np.random.default_rng(RNG_SEED)
all_folds = {}
for tl in TRAIN_LENS:
    max_start = len(realized_vol) - tl - HORIZON
    starts = rng.choice(max_start, size=min(N_FOLDS, max_start), replace=False)
    all_folds[tl] = starts

print("train_len=1000 starts:", all_folds[1000][:6])
print("train_len=250 starts:", all_folds[250][:12])
np.save(RESULTS_DIR / "realized_vol.npy", realized_vol)
import pickle
with open(RESULTS_DIR / "folds.pkl", "wb") as f:
    pickle.dump(all_folds, f)
print(f"saved to {RESULTS_DIR}")
