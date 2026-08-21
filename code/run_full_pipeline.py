import warnings, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import NGRC, nrmse, tuned_esn_predict, fit_tuned_ngrc, fit_tuned_prngrc
from relative_ridge import PseudorandomNGRC_RelRidge
from arch import arch_model
warnings.filterwarnings("ignore")

# Fixed, known repo layout: this script lives in code/, inputs live in the
# sibling data/ directory, outputs go to the sibling results/ directory.
# Resolved relative to the script's own location so it works regardless of
# the caller's current working directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RNG_SEED = 0
HORIZON = 20
PRNGRC_M = 100
REL_RIDGE_C = 1e-5
WIN = 20

dataset = sys.argv[1]        # 'sp500' or 'nasdaq'
tl = int(sys.argv[2])

realized_vol = np.load(DATA_DIR / f"{dataset}_realized_vol.npy")
logret = np.load(DATA_DIR / f"{dataset}_logret.npy")


def ar1_forecast(train, n_steps):
    y, x = train[1:], train[:-1]
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, b = coef
    out = np.zeros(n_steps)
    last = train[-1]
    for i in range(n_steps):
        last = a + b * last
        out[i] = last
    return out


def persistence_forecast(train, n_steps):
    return np.full(n_steps, train[-1])


def har_forecast(train, n_steps, d=1, w=5, m=22):
    n = len(train)
    X, y = [], []
    for t in range(m, n):
        y.append(train[t])
        rv_d = train[t - 1]
        rv_w = train[t - w:t].mean()
        rv_m = train[t - m:t].mean()
        X.append([1.0, rv_d, rv_w, rv_m])
    X = np.array(X)
    y = np.array(y)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    c, bd, bw, bm = coef
    hist = list(train)
    out = np.zeros(n_steps)
    for i in range(n_steps):
        rv_d = hist[-1]
        rv_w = np.mean(hist[-w:])
        rv_m = np.mean(hist[-m:])
        pred = c + bd * rv_d + bw * rv_w + bm * rv_m
        out[i] = pred
        hist.append(pred)
    return out


def garch_forecast(train_logret_full, n_steps=HORIZON, win=WIN):
    am = arch_model(train_logret_full * 100, mean="Zero", vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(disp="off")
    fc = res.forecast(horizon=n_steps, reindex=False)
    sigma = np.sqrt(fc.variance.values[-1]) / 100.0
    expected_abs_r = sigma * np.sqrt(2.0 / np.pi)
    trailing_actual = np.abs(train_logret_full[-(win - 1):])
    combined = np.concatenate([trailing_actual, expected_abs_r])
    return np.array([combined[i:i + win].mean() for i in range(n_steps)])


block = tl + HORIZON
max_folds = len(realized_vol) // block
n_folds = min(25, max_folds)
starts = [i * block for i in range(n_folds)]

results = []
t0 = time.time()
for fold, start in enumerate(starts):
    train = realized_vol[start:start + tl]
    test = realized_vol[start + tl:start + tl + HORIZON]
    mu, sd = train.mean(), train.std()
    sd = sd if sd > 1e-12 else 1.0
    train_n = ((train - mu) / sd).reshape(-1, 1)

    preds = {
        "persistence": persistence_forecast(train, HORIZON),
        "AR1": ar1_forecast(train, HORIZON),
    }
    try:
        m = NGRC(k=3, skip=1, poly_degree=2, ridge_alpha=1e-2, predict_derivative=False)
        m.fit(train_n)
        preds["NGRC_fixed"] = m.predict(HORIZON)[:, 0] * sd + mu
    except Exception:
        preds["NGRC_fixed"] = np.full(HORIZON, np.nan)
    try:
        m = fit_tuned_ngrc(train_n, seed=RNG_SEED + fold, predict_derivative=False)
        preds["NGRC_tuned"] = m.predict(HORIZON)[:, 0] * sd + mu
    except Exception:
        preds["NGRC_tuned"] = np.full(HORIZON, np.nan)
    try:
        m = fit_tuned_prngrc(train_n, seed=RNG_SEED + fold, M=PRNGRC_M)
        preds["NGRC_pseudorandom"] = m.predict(HORIZON)[:, 0] * sd + mu
    except Exception:
        preds["NGRC_pseudorandom"] = np.full(HORIZON, np.nan)
    try:
        m = PseudorandomNGRC_RelRidge(k=3, skip=1, M=PRNGRC_M, c=REL_RIDGE_C, seed=RNG_SEED + fold)
        m.fit(train_n)
        preds["NGRC_pseudorandom_relridge"] = m.predict(HORIZON)[:, 0] * sd + mu
    except Exception:
        preds["NGRC_pseudorandom_relridge"] = np.full(HORIZON, np.nan)
    try:
        preds["ESN"] = tuned_esn_predict(train_n, HORIZON, seed=RNG_SEED + fold)[:, 0] * sd + mu
    except Exception:
        preds["ESN"] = np.full(HORIZON, np.nan)
    try:
        train_logret_full = logret[start: start + tl + WIN - 1]
        preds["GARCH"] = garch_forecast(train_logret_full)
    except Exception:
        preds["GARCH"] = np.full(HORIZON, np.nan)
    try:
        preds["HAR"] = har_forecast(train, HORIZON)
    except Exception:
        preds["HAR"] = np.full(HORIZON, np.nan)

    row = dict(dataset=dataset, train_len=tl, fold=fold)
    for name, p in preds.items():
        row[f"nrmse_{name}"] = nrmse(test, p)
    results.append(row)

df = pd.DataFrame(results)
out_path = RESULTS_DIR / f"full_{dataset}_tl{tl}.csv"
df.to_csv(out_path, index=False)
print(f"{dataset} tl={tl} n_folds={n_folds} elapsed={time.time()-t0:.1f}s -> wrote {out_path}")
