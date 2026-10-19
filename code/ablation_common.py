"""Shared helpers for the regularization-rule and noise-injection ablations."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent  # place these scripts in <repo>/code/
sys.path.insert(0, str(REPO / "code"))
from ngrc_lib import PseudorandomNGRC, nrmse  # noqa: E402

HORIZON = 20
M = 100
TRAIN_LENGTHS = [60, 120, 250, 500, 1000]
DATASETS = ["sp500", "nasdaq"]
DATA_DIR = REPO / "data"


def folds(dataset, tl):
    rv = np.load(DATA_DIR / f"{dataset}_realized_vol.npy")
    block = tl + HORIZON
    n_folds = min(25, len(rv) // block)
    for fold in range(n_folds):
        s = fold * block
        yield fold, rv[s:s + tl], rv[s + tl:s + tl + HORIZON]


class Prepared:
    """Features / Gram matrix for one fold, so many alpha rules reuse one projection."""

    def __init__(self, train_n, seed):
        X = np.asarray(train_n, float)
        m = PseudorandomNGRC(k=3, skip=1, M=M, seed=seed)
        span = (m.k - 1) * m.skip
        lin = m._linear_features(X[:-1])
        m._lo, m._hi = lin.min(), lin.max()
        m._pairs = m._build_pairs(lin.shape[1])
        F = m._project(m._normalize(lin))
        self.m, self.F, self.X, self.span = m, F, X, span
        self.T = X[span + 1:]
        self.G = F.T @ F
        self.FT = F.T @ self.T
        ev = np.linalg.eigvalsh(self.G)
        self.lmax, self.lmin, self.tr = ev[-1], ev[0], np.trace(self.G)
        self.d = X.shape[1]
        self.m.d = self.d

    def forecast(self, alpha, steps=HORIZON):
        self.m.W_out = np.linalg.solve(self.G + alpha * np.eye(M), self.FT)
        self.m._last_window = self.X[-self.span - 1:]
        return self.m.predict(steps)[:, 0]


def tuned_alpha(train_n, seed, grid):
    """Validation-window selection exactly as fit_tuned_prngrc, over an arbitrary grid."""
    n = len(train_n)
    val_len = max(5, min(HORIZON, n // 5))
    best, best_score = None, np.inf
    if n - val_len >= 20:
        sub, val = train_n[:n - val_len], train_n[n - val_len:]
        P = Prepared(sub, seed)
        for a in grid:
            try:
                p = P.forecast(a, val_len).reshape(-1, 1)
                sc = nrmse(val, p)
                if np.isfinite(sc) and sc < best_score:
                    best_score, best = sc, a
            except Exception:
                continue
    return best if best is not None else 1e-2
