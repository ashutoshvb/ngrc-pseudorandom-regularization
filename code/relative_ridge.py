"""
Conditioning-aware (eigenvalue-scaled) ridge for PseudorandomNGRC.
alpha = c * lambda_max(feats^T feats), so regularization strength scales
automatically with each fold's own feature-space conditioning instead of
being an absolute constant picked from a fixed grid.
"""
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ngrc_lib import PseudorandomNGRC, nrmse


class PseudorandomNGRC_RelRidge(PseudorandomNGRC):
    def __init__(self, k=3, skip=1, M=200, c=1e-4, seed=0, eps=1e-6):
        super().__init__(k=k, skip=skip, M=M, ridge_alpha=None, seed=seed, eps=eps)
        self.c = c

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.d = X.shape[1]
        span = (self.k - 1) * self.skip
        lin = self._linear_features(X[:-1])
        m0 = lin.shape[1]
        self._lo, self._hi = lin.min(), lin.max()
        lin_norm = self._normalize(lin)
        self._pairs = self._build_pairs(m0)
        feats = self._project(lin_norm)
        G = feats.T @ feats
        lam_max = np.linalg.eigvalsh(G)[-1]
        alpha = self.c * max(lam_max, 1e-300)
        self.ridge_alpha = alpha
        targets = X[span + 1:]
        reg = alpha * np.eye(feats.shape[1])
        self.W_out = np.linalg.solve(G + reg, feats.T @ targets)
        self._last_window = X[-span - 1:]
        return self


def fit_tuned_prngrc_relridge(train_n, seed=0, k=3, skip=1, M=100, c_grid=None, horizon_for_val=20):
    if c_grid is None:
        c_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    n = len(train_n)
    val_len = max(5, min(horizon_for_val, n // 5))
    best, best_score = None, np.inf
    if n - val_len >= 20:
        sub_train, val = train_n[:n - val_len], train_n[n - val_len:]
        for c in c_grid:
            try:
                m = PseudorandomNGRC_RelRidge(k=k, skip=skip, M=M, c=c, seed=seed)
                m.fit(sub_train)
                p = m.predict(val_len)
                score = nrmse(val, p)
                if np.isfinite(score) and score < best_score:
                    best_score, best = score, c
            except Exception:
                continue
    c = best if best is not None else 1e-4
    m = PseudorandomNGRC_RelRidge(k=k, skip=skip, M=M, c=c, seed=seed)
    m.fit(train_n)
    m.tuned_c = c
    return m
