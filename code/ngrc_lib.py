"""
Core library: Next-Generation Reservoir Computing (NG-RC) and a standard
Echo State Network (ESN) baseline, plus shared utilities.

NG-RC follows Gauthier, Bollt, Griffith & Barbosa, "Next generation
reservoir computing", Nat. Commun. 12, 5564 (2021): features are built from
k time-delayed observations (linear part) plus their unique degree-2
polynomial combinations (nonlinear part), and a linear (ridge-regularized)
readout is trained to predict either the next state or the state
increment. No random reservoir, no recurrent dynamics to tune -- the only
hyperparameters are the number of delays k, the polynomial degree, and the
ridge coefficient.

ESN follows the standard formulation (Jaeger 2001 / Lukosevicius & Jaeger
2009 practical guide): a large sparse random recurrent reservoir driven by
the input, a linear ridge-regularized readout trained on reservoir states
after a washout period.
"""

from __future__ import annotations
import numpy as np
from itertools import combinations_with_replacement


# --------------------------------------------------------------------------
# Next-Generation Reservoir Computing
# --------------------------------------------------------------------------
class NGRC:
    def __init__(self, k=4, skip=1, poly_degree=2, ridge_alpha=1e-4,
                 predict_derivative=True, bias=True):
        """
        k:            number of delay taps (including current step)
        skip:         spacing between taps (1 = consecutive samples)
        poly_degree:  highest polynomial degree of nonlinear features (2 -> quadratic)
        ridge_alpha:  L2 regularization strength for the readout
        predict_derivative: if True, train the readout to predict the
                      increment x[t+1]-x[t] (works well for smooth flows
                      like Lorenz); if False, predict x[t+1] directly
                      (better for series without a natural "velocity",
                      e.g. financial returns).
        bias:         include a constant feature
        """
        self.k = k
        self.skip = skip
        self.poly_degree = poly_degree
        self.ridge_alpha = ridge_alpha
        self.predict_derivative = predict_derivative
        self.bias = bias
        self.W_out = None
        self.d = None
        self._idx_cache = {}  # m -> (idx_pairs, idx_triples), avoids rebuilding
                               # itertools.combinations_with_replacement on
                               # every single-step call during a long
                               # autonomous rollout (was the dominant cost:
                               # ~7x slower than the ESN per rollout step
                               # before this cache was added)

    def _linear_features(self, X):
        """X: (T, d) -> linear feature matrix using k delayed taps.
        Returns features aligned to the *last* time index of each window,
        i.e. output row t corresponds to input rows
        [t - (k-1)*skip, ..., t - skip, t]."""
        T, d = X.shape
        span = (self.k - 1) * self.skip
        n_out = T - span
        feats = np.zeros((n_out, self.k * d))
        for j in range(self.k):
            offset = span - j * self.skip
            feats[:, j * d:(j + 1) * d] = X[span - offset: span - offset + n_out]
        return feats  # (n_out, k*d)

    def _nonlinear_features(self, lin_feats):
        """Unique polynomial combinations up to poly_degree of the linear
        feature vector (captures cross-terms, matching the NG-RC papers)."""
        n, m = lin_feats.shape
        if m not in self._idx_cache:
            idx_pairs = (np.array(list(combinations_with_replacement(range(m), 2)))
                         if self.poly_degree >= 2 else None)
            idx_triples = (np.array(list(combinations_with_replacement(range(m), 3)))
                           if self.poly_degree >= 3 else None)
            self._idx_cache[m] = (idx_pairs, idx_triples)
        idx_pairs, idx_triples = self._idx_cache[m]

        cols = [lin_feats] if self.poly_degree >= 1 else []
        if idx_pairs is not None:
            quad = lin_feats[:, idx_pairs[:, 0]] * lin_feats[:, idx_pairs[:, 1]]
            cols.append(quad)
        if idx_triples is not None:
            cube = (lin_feats[:, idx_triples[:, 0]] * lin_feats[:, idx_triples[:, 1]]
                    * lin_feats[:, idx_triples[:, 2]])
            cols.append(cube)
        feats = np.concatenate(cols, axis=1)
        if self.bias:
            feats = np.concatenate([np.ones((n, 1)), feats], axis=1)
        return feats

    def _featurize(self, X):
        lin = self._linear_features(X)
        return self._nonlinear_features(lin)

    def fit(self, X):
        """X: (T, d) training trajectory. Trains readout to map the feature
        vector at time t to x[t+1] (or the increment)."""
        X = np.asarray(X, dtype=float)
        self.d = X.shape[1]
        span = (self.k - 1) * self.skip
        feats = self._featurize(X[:-1])          # rows correspond to t = span .. T-2
        targets_x = X[span + 1:]                  # x[t+1] for those same t
        if self.predict_derivative:
            targets = targets_x - X[span:-1]
        else:
            targets = targets_x
        F = feats
        reg = self.ridge_alpha * np.eye(F.shape[1])
        self.W_out = np.linalg.solve(F.T @ F + reg, F.T @ targets)
        self._last_window = X[-span - 1:]  # store final k-window for forecasting
        return self

    def _step(self, window):
        """window: (k, d) most recent states (oldest first). Returns next state."""
        feats = self._nonlinear_features(window.flatten()[None, :])
        pred = feats @ self.W_out
        x_next = (window[-1] + pred[0]) if self.predict_derivative else pred[0]
        return x_next

    def predict(self, n_steps, init_window=None, clip=50.0):
        """Autonomous multi-step forecast starting from the end of the
        training data (or a supplied init_window of shape (k, d)).

        `clip`: safety bound (in units of the training-data scale, which is
        O(1) if the caller normalized the input as recommended). NG-RC's
        quadratic feedback loop can occasionally run away on short/noisy
        training data; once a prediction exceeds this bound the forecast is
        held at the last stable value rather than propagating inf/NaN. This
        is a standard practical safeguard, not part of the "real" method --
        runs that trigger it indicate the fit was unstable for that
        training window."""
        window = self._last_window.copy() if init_window is None else init_window.copy()
        d = window.shape[1]
        out = np.zeros((n_steps, d))
        unstable = False
        last_stable = window[-1].copy()
        for t in range(n_steps):
            if not unstable:
                x_next = self._step(window)
                if not np.all(np.isfinite(x_next)) or np.any(np.abs(x_next) > clip):
                    unstable = True
                else:
                    last_stable = x_next
            if unstable:
                x_next = last_stable
            out[t] = x_next
            window = np.vstack([window[1:], x_next])
        return out


# --------------------------------------------------------------------------
# Pseudorandom-projection NGRC (Cestnik & Martens, Chaos 36, 013115, 2026)
# --------------------------------------------------------------------------
class PseudorandomNGRC:
    """NGRC variant using a pseudorandom nonlinear projection of the
    time-delay embedded input instead of explicit polynomial features
    (Cestnik & Martens, "Next-generation reservoir computing for
    dynamical inference", Chaos 36, 013115, 2026). Built specifically to
    test whether that paper's noise-tolerance claims -- and the
    stability/accuracy distinction found in this project's
    run_lorenz_escape.py -- hold for their projection method too, or are
    specific to the polynomial NGRC used elsewhere in this project.

    Method (their Sec. III.B): the delay-embedded input is normalized to
    the open interval (eps, 1-eps), then M new features are built one at
    a time by pseudo-randomly selecting two existing components p_i, p_j
    from a growing pool (starting with the N*k normalized inputs
    themselves) and combining them as p_m = (1 - p_i) ** p_j. Because
    p_i, p_j in (0,1), this combination is guaranteed to stay in (0,1)
    too, which is the point: it's a projection designed to never leave a
    bounded, numerically well-behaved range, unlike an unbounded
    polynomial expansion. The projected vector [p_1, ..., p_M] is the
    feature vector; a linear ridge-regularized readout maps it to the
    next raw state, exactly as in NGRC.

    The pairing sequence (i, j) is generated once per model (seeded, for
    reproducibility) and reused for every time step and for every
    autonomous-rollout step, matching how NGRC's polynomial index pairs
    are fixed at fit time.
    """

    def __init__(self, k=3, skip=1, M=200, ridge_alpha=1e-2, seed=0, eps=1e-6):
        self.k = k
        self.skip = skip
        self.M = M
        self.ridge_alpha = ridge_alpha
        self.seed = seed
        self.eps = eps
        self.W_out = None
        self.d = None
        self._pairs = None   # cached (i, j) pseudorandom index sequence
        self._lo = None
        self._hi = None

    def _linear_features(self, X):
        T, d = X.shape
        span = (self.k - 1) * self.skip
        n_out = T - span
        feats = np.zeros((n_out, self.k * d))
        for j in range(self.k):
            offset = span - j * self.skip
            feats[:, j * d:(j + 1) * d] = X[span - offset: span - offset + n_out]
        return feats

    def _build_pairs(self, m0):
        """Deterministically generate M (i, j) index pairs into a pool
        that starts at size m0 and grows by one each step, per Cestnik &
        Martens Sec. III.B. Duplicate (i, j) pairs are skipped."""
        rng = np.random.default_rng(self.seed)
        pairs = []
        seen = set()
        pool_size = m0
        while len(pairs) < self.M:
            i = rng.integers(0, pool_size)
            j = rng.integers(0, pool_size)
            if (i, j) in seen:
                continue
            seen.add((i, j))
            pairs.append((i, j))
            pool_size += 1
        return pairs

    def _normalize(self, lin_feats):
        rng_span = self._hi - self._lo
        rng_span = rng_span if rng_span > 1e-12 else 1.0
        z = (lin_feats - self._lo) / rng_span
        z = self.eps + (1 - 2 * self.eps) * z
        return np.clip(z, self.eps, 1 - self.eps)

    def _project(self, lin_feats_norm):
        n, m0 = lin_feats_norm.shape
        pool = np.concatenate([lin_feats_norm,
                                np.zeros((n, self.M))], axis=1)
        for m, (i, j) in enumerate(self._pairs):
            pool[:, m0 + m] = (1.0 - pool[:, i]) ** pool[:, j]
        return pool[:, m0:]

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

        targets = X[span + 1:]  # direct next-state mapping, matching Eq. 3 of the paper
        reg = self.ridge_alpha * np.eye(feats.shape[1])
        self.W_out = np.linalg.solve(feats.T @ feats + reg, feats.T @ targets)
        self._last_window = X[-span - 1:]
        return self

    def _step(self, window):
        lin = window.flatten()[None, :]
        lin_norm = self._normalize(lin)
        feats = self._project(lin_norm)
        pred = feats @ self.W_out
        return pred[0]

    def predict(self, n_steps, init_window=None, clip=50.0):
        """Same bounded-rollout safeguard as NGRC.predict/ESN.predict, for
        a like-for-like comparison (this projection is itself designed
        to be more numerically stable, so triggering the safeguard here
        at all is informative)."""
        window = self._last_window.copy() if init_window is None else init_window.copy()
        d = window.shape[1]
        out = np.zeros((n_steps, d))
        unstable = False
        last_stable = window[-1].copy()
        for t in range(n_steps):
            if not unstable:
                x_next = self._step(window)
                if not np.all(np.isfinite(x_next)) or np.any(np.abs(x_next) > clip):
                    unstable = True
                else:
                    last_stable = x_next
            if unstable:
                x_next = last_stable
            out[t] = x_next
            window = np.vstack([window[1:], x_next])
        return out


def fit_tuned_prngrc(train_n, seed=0, k=3, skip=1, M=200, ridge_grid=None, horizon_for_val=20):
    """Small validation-set ridge grid search, mirroring fit_tuned_ngrc,
    so the pseudorandom-projection variant is compared on equal footing."""
    if ridge_grid is None:
        ridge_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    n = len(train_n)
    val_len = max(5, min(horizon_for_val, n // 5))
    best, best_score = None, np.inf
    if n - val_len >= 20:
        sub_train, val = train_n[:n - val_len], train_n[n - val_len:]
        for alpha in ridge_grid:
            try:
                m = PseudorandomNGRC(k=k, skip=skip, M=M, ridge_alpha=alpha, seed=seed)
                m.fit(sub_train)
                p = m.predict(val_len)
                score = nrmse(val, p)
                if np.isfinite(score) and score < best_score:
                    best_score, best = score, alpha
            except Exception:
                continue
    alpha = best if best is not None else 1e-2
    m = PseudorandomNGRC(k=k, skip=skip, M=M, ridge_alpha=alpha, seed=seed)
    m.fit(train_n)
    m.tuned_ridge_alpha = alpha
    return m


# --------------------------------------------------------------------------
# Standard Echo State Network baseline
# --------------------------------------------------------------------------
class ESN:
    def __init__(self, n_reservoir=300, spectral_radius=0.9, sparsity=0.9,
                 input_scale=1.0, leak_rate=0.3, ridge_alpha=1e-4,
                 washout=50, seed=0):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.input_scale = input_scale
        self.leak_rate = leak_rate
        self.ridge_alpha = ridge_alpha
        self.washout = washout
        self.rng = np.random.default_rng(seed)
        self.W_out = None
        self.d = None

    def _init_weights(self, d):
        n = self.n_reservoir
        W = self.rng.normal(0, 1, (n, n))
        mask = self.rng.random((n, n)) < self.sparsity
        W[mask] = 0.0
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        if radius > 0:
            W *= self.spectral_radius / radius
        W_in = self.rng.uniform(-self.input_scale, self.input_scale, (n, d))
        self.W, self.W_in = W, W_in

    def _run_reservoir(self, X, state0=None):
        T, d = X.shape
        n = self.n_reservoir
        states = np.zeros((T, n))
        s = np.zeros(n) if state0 is None else state0.copy()
        for t in range(T):
            pre = self.W @ s + self.W_in @ X[t]
            s = (1 - self.leak_rate) * s + self.leak_rate * np.tanh(pre)
            states[t] = s
        return states, s

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.d = X.shape[1]
        self._init_weights(self.d)
        states, s_final = self._run_reservoir(X[:-1])
        targets = X[1:]
        w = self.washout if self.washout < len(states) - 1 else 0
        S = states[w:]
        Y = targets[w:]
        ext = np.concatenate([np.ones((S.shape[0], 1)), S, X[w:-1]], axis=1)
        reg = self.ridge_alpha * np.eye(ext.shape[1])
        self.W_out = np.linalg.solve(ext.T @ ext + reg, ext.T @ Y)
        self._last_state = s_final
        self._last_x = X[-1]
        return self

    def predict(self, n_steps, state0=None, x0=None, clip=50.0):
        """Autonomous multi-step forecast. `clip`: same safety bound as
        NGRC.predict (in units of the normalized training scale) -- a
        spectral radius >1, which our grid search does try, has no
        stability guarantee for the *autonomous* (closed-loop) regime, and
        can blow up numerically over long rollouts. Once triggered, the
        forecast holds at the last stable value rather than propagating
        inf/NaN, applied symmetrically with NGRC's safeguard so neither
        model is penalized or protected differently in the comparison."""
        s = self._last_state.copy() if state0 is None else state0.copy()
        x = self._last_x.copy() if x0 is None else x0.copy()
        out = np.zeros((n_steps, self.d))
        unstable = False
        last_stable = x.copy()
        for t in range(n_steps):
            if not unstable:
                pre = self.W @ s + self.W_in @ x
                s_new = (1 - self.leak_rate) * s + self.leak_rate * np.tanh(pre)
                ext = np.concatenate([[1.0], s_new, x])
                x_new = ext @ self.W_out
                if not np.all(np.isfinite(x_new)) or np.any(np.abs(x_new) > clip):
                    unstable = True
                else:
                    s, x, last_stable = s_new, x_new, x_new.copy()
            if unstable:
                x = last_stable
            out[t] = x
        return out


def fit_tuned_esn(train_n, seed, n_reservoir=100, combos=None, horizon_for_val=20):
    """Small validation-set grid search over (spectral_radius, leak_rate)
    instead of one fixed setting, so ESN comparisons aren't handicapped by
    an arbitrarily under-tuned baseline. Works for any state dimension d
    (train_n: (T, d) normalized array). Returns the *fitted model* (not a
    forecast), so callers can generate multiple rollouts (e.g. from several
    perturbed initial conditions) without retraining -- needed for the
    escape-fraction stability protocol.

    This is the single tuned-ESN implementation shared by every benchmark
    in this project (Lorenz and S&P 500) so that "NG-RC beats ESN" claims
    are made against the same opponent everywhere, not a weaker one in
    some experiments and a stronger one in others."""
    if combos is None:
        combos = [(0.8, 0.3), (0.8, 0.6), (1.1, 0.3), (1.1, 0.6)]
    n = len(train_n)
    val_len = max(5, min(horizon_for_val, n // 5))
    best, best_score = None, np.inf
    if n - val_len >= 20:
        sub_train, val = train_n[:n - val_len], train_n[n - val_len:]
        for sr, lr in combos:
            try:
                e = ESN(n_reservoir=n_reservoir, spectral_radius=sr, sparsity=0.9,
                        input_scale=0.5, leak_rate=lr, ridge_alpha=1e-2,
                        washout=min(20, len(sub_train) // 4), seed=seed)
                e.fit(sub_train)
                p = e.predict(val_len)
                score = nrmse(val, p)
                if np.isfinite(score) and score < best_score:
                    best_score, best = score, (sr, lr)
            except Exception:
                continue
    sr, lr = best if best is not None else (0.9, 0.3)
    e = ESN(n_reservoir=n_reservoir, spectral_radius=sr, sparsity=0.9, input_scale=0.5,
            leak_rate=lr, ridge_alpha=1e-2, washout=min(20, n // 4), seed=seed)
    e.fit(train_n)
    return e


def tuned_esn_predict(train_n, horizon, seed, n_reservoir=100, combos=None):
    """Convenience wrapper: fit_tuned_esn then predict in one call (the
    common case when only a single forecast is needed)."""
    e = fit_tuned_esn(train_n, seed, n_reservoir=n_reservoir, combos=combos,
                       horizon_for_val=horizon)
    return e.predict(horizon)


def fit_tuned_ngrc(train_n, seed=0, k=3, skip=1, poly_degree=2,
                    predict_derivative=True, ridge_grid=None, horizon_for_val=20):
    """Small validation-set grid search over ridge_alpha, mirroring
    fit_tuned_esn -- so NG-RC isn't compared against a tuned ESN while
    using one arbitrarily fixed ridge value itself. Both papers this
    project builds on note that the ridge strength needed for a clean
    signal is not the ridge strength needed for a noisy one (Gauthier et
    al. 2021, Supplementary Note 5; Cestnik & Martens 2026, Fig. 4-5), so
    fixing it across noise levels is exactly the kind of gap that would
    make a noise-robustness comparison misleading. `seed` is accepted for
    interface symmetry with fit_tuned_esn (NGRC's fit is deterministic
    given data, but call sites iterate over "independently trained
    models" for both methods uniformly)."""
    if ridge_grid is None:
        ridge_grid = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    n = len(train_n)
    val_len = max(5, min(horizon_for_val, n // 5))
    best, best_score = None, np.inf
    if n - val_len >= 20:
        sub_train, val = train_n[:n - val_len], train_n[n - val_len:]
        for alpha in ridge_grid:
            try:
                m = NGRC(k=k, skip=skip, poly_degree=poly_degree, ridge_alpha=alpha,
                         predict_derivative=predict_derivative)
                m.fit(sub_train)
                p = m.predict(val_len)
                score = nrmse(val, p)
                if np.isfinite(score) and score < best_score:
                    best_score, best = score, alpha
            except Exception:
                continue
    alpha = best if best is not None else 1e-4
    m = NGRC(k=k, skip=skip, poly_degree=poly_degree, ridge_alpha=alpha,
             predict_derivative=predict_derivative)
    m.fit(train_n)
    m.tuned_ridge_alpha = alpha  # stashed for inspection/logging
    return m


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def nrmse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.sqrt(np.mean((y_true - y_pred) ** 2))
    den = np.std(y_true)
    return num / den if den > 0 else np.nan


def valid_prediction_time(y_true, y_pred, dt, lyapunov_time=None, threshold=0.4):
    """Number of steps (converted to time, and optionally Lyapunov times)
    until the normalized error exceeds `threshold`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.std(y_true, axis=0)
    err = np.sqrt(np.mean(((y_true - y_pred) / sigma) ** 2, axis=1))
    over = np.where(err > threshold)[0]
    steps = over[0] if len(over) else len(err)
    t = steps * dt
    if lyapunov_time:
        return t / lyapunov_time
    return t
