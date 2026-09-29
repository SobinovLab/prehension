#!python3
# -*- coding: utf-8 -*-
"""
Time-resolved cross-validated decoding helpers.

Two families, both generic and data-agnostic (no prehension model):

  * classification: linear-discriminant decoding over an ``(n_samples, n_features, n_time)``
    activity tensor -- a stratified k-fold cross-validated accuracy at every time bin (real
    labels plus one label-shuffle for a chance estimate), farmed out to a multiprocessing.Pool,
    and a percentile-based chance level pooled across time;
  * regression: Kalman (KalmanFilterDecoder, Wu et al. 2006) and Wiener / linear-FIR
    (WienerFilterDecoder) decoders of a continuous behavioural state from neural firing rates,
    with trial-wise cross-validation -- pooled over held-out trials (kalman_decode_cv) or
    keeping each held-out trial separate (decode_cv_per_trial, with an optional per-fold PCA
    reduction of the neural observations).  A time-alignment shuffle (shuffle_trial_alignment)
    provides a per-trial chance null; coefficient_of_determination / rmse / pearson_r score a
    single trial.

scikit-learn is imported lazily (inside the workers / functions) so importing this module
stays cheap.

Copyright (C) 2026 Anton Sobinov
https://github.com/SobinovLab/prehension

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import multiprocessing

import numpy as np


def cv_lda_accuracy(features, labels, n_folds, seed):
    """Mean k-fold cross-validated LDA accuracy for one (samples, features) set.

    Folds are stratified so every class appears in each split; the fold count is capped
    at the smallest class size.  Returns np.nan when there are fewer than 2 classes or
    fewer than 2 samples in the smallest class (classification undefined).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Integer-encode the labels: float class values (e.g. target forces like 0.5,
    # 1.4, ...) are otherwise read as a 'continuous' target and rejected by
    # StratifiedKFold/LDA, even though they are discrete classes.
    classes, encoded, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return np.nan
    folds = int(min(n_folds, counts.min()))
    if folds < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearDiscriminantAnalysis(), features, encoded, cv=skf)
    return float(np.mean(scores))


def _classify_timepoint(task):
    """Worker: real and single-shuffle cross-validated accuracy for one time bin.

    task = (features (n_samples, n_features), labels, n_folds, seed).  The label vector
    is permuted once (a single shuffle) for the chance estimate.  Returns
    (accuracy, shuffle_accuracy).  Top-level (picklable) so it works with
    multiprocessing under spawn (Windows).
    """
    features, labels, n_folds, seed = task
    accuracy = cv_lda_accuracy(features, labels, n_folds, seed)
    rng = np.random.RandomState(seed + 1)
    shuffle = cv_lda_accuracy(
        features, labels[rng.permutation(len(labels))], n_folds, seed + 2)
    return accuracy, shuffle


def classify_through_time(X, labels, n_folds=5, processes=1, seed=None):
    """Classify the labels at every time bin, optionally over a process pool.

    X is (n_samples, n_features, n_time); each time bin is an independent classification
    (features = the columns at that bin), evaluated on the real labels and once on a
    shuffled copy.  The per-bin work is dispatched to a multiprocessing.Pool of
    `processes` workers (serial when processes <= 1).  Each bin gets a distinct seed drawn
    from a RandomState(`seed`) so the folds/shuffle differ across bins; seed=None (the
    default) draws them randomly (a fresh run each time), an int makes the run reproducible.

    Returns (accuracy (n_time,), shuffle (n_time,)); empty feature sets give all-nan
    results.  The single-shuffle accuracies are meant to be pooled across time and
    reduced to a chance level by the caller (see chance_level).
    """
    n_time = X.shape[2]
    if X.shape[1] == 0:
        return np.full(n_time, np.nan), np.full(n_time, np.nan)

    # distinct per-bin seeds (random when seed is None); the workers do seed arithmetic, so
    # they always receive an int here rather than None.
    bin_seeds = np.random.RandomState(seed).randint(0, 2 ** 31 - 1, size=n_time)
    tasks = [(np.ascontiguousarray(X[:, :, t]), labels, n_folds, int(bin_seeds[t]))
             for t in range(n_time)]
    if processes and processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(_classify_timepoint, tasks)
    else:
        results = [_classify_timepoint(task) for task in tasks]

    accuracy = np.array([r[0] for r in results])
    shuffle = np.array([r[1] for r in results])
    return accuracy, shuffle


def chance_level(shuffle, percentile=90.0):
    """Pool the per-time-bin shuffled accuracies and return their `percentile` (chance).

    A single shuffle is drawn per time bin (see classify_through_time); pooling those
    across all bins and taking the given percentile yields one scalar chance level for
    the dataset.  Returns np.nan when no finite shuffle accuracy is available.
    """
    shuffle = np.asarray(shuffle, dtype=float)
    if not np.any(np.isfinite(shuffle)):
        return np.nan
    return float(np.nanpercentile(shuffle, percentile))


class KalmanFilterDecoder:
    """Wu et al. (2006) Kalman filter neural decoder: latent state = behaviour, obs = rates.

    The latent state x_t is the behavioural variable to decode (e.g. joint angles); the
    observation z_t is the neural population's binned firing rate at the same time bin.  A
    linear-Gaussian state-space model is fit in closed form from paired training sequences and
    the state is then estimated from neural observations with the forward Kalman filter:

        x_t = A x_{t-1} + w,   w ~ N(0, W)          (state / dynamics model)
        z_t = H x_t     + q,   q ~ N(0, Q)          (observation / tuning model)

    fit() estimates A, W (from within-trial consecutive state pairs) and H, Q (from all aligned
    state/observation samples) by least squares; predict() runs the filter on one trial's
    observations.  Pure numpy (linear algebra only), so it stays in tools.
    """

    def __init__(self):
        self.A = self.W = self.H = self.Q = None
        self.x0 = self.P0 = None

    def fit(self, states, observations):
        """Fit the model from per-trial sequences.

        `states` / `observations` are lists of per-trial arrays, (T_i, n_state) and
        (T_i, n_obs), aligned in time within each trial.  A / W come from consecutive state
        pairs within a trial (so trial boundaries are not paired); H / Q from every aligned
        sample.  The initial state / covariance are the training states' mean and covariance.
        """
        X = np.vstack([np.asarray(s, dtype=float) for s in states])          # (N, d)
        Z = np.vstack([np.asarray(z, dtype=float) for z in observations])    # (N, n)

        prev = np.vstack([np.asarray(s, dtype=float)[:-1] for s in states
                          if np.asarray(s).shape[0] >= 2])                   # x_{t-1}
        curr = np.vstack([np.asarray(s, dtype=float)[1:] for s in states
                          if np.asarray(s).shape[0] >= 2])                   # x_t
        # x_t = A x_{t-1}: A = (x_t x_{t-1}^T)(x_{t-1} x_{t-1}^T)^{-1}
        self.A = curr.T @ prev @ np.linalg.pinv(prev.T @ prev)
        state_resid = curr.T - self.A @ prev.T
        self.W = state_resid @ state_resid.T / prev.shape[0]
        # z_t = H x_t: H = (z x^T)(x x^T)^{-1}
        self.H = Z.T @ X @ np.linalg.pinv(X.T @ X)
        obs_resid = Z.T - self.H @ X.T
        self.Q = obs_resid @ obs_resid.T / X.shape[0]

        self.x0 = X.mean(axis=0)
        self.P0 = np.atleast_2d(np.cov(X.T))
        return self

    def predict(self, observations):
        """Decode one trial: (T, n_obs) observations -> (T, n_state) filtered state estimate."""
        Z = np.asarray(observations, dtype=float)
        d = self.A.shape[0]
        x, P = self.x0.copy(), self.P0.copy()
        eye = np.eye(d)
        out = np.zeros((Z.shape[0], d))
        for t in range(Z.shape[0]):
            x_pred = self.A @ x
            P_pred = self.A @ P @ self.A.T + self.W
            innovation_cov = self.H @ P_pred @ self.H.T + self.Q          # (n, n)
            cross = P_pred @ self.H.T                                     # (d, n)
            # Kalman gain K = cross @ innovation_cov^{-1}, via a solve (pinv fallback)
            try:
                gain = np.linalg.solve(innovation_cov, cross.T).T
            except np.linalg.LinAlgError:
                gain = cross @ np.linalg.pinv(innovation_cov)
            x = x_pred + gain @ (Z[t] - self.H @ x_pred)
            P = (eye - gain @ self.H) @ P_pred
            out[t] = x
        return out


def kalman_decode_cv(states, observations, n_folds=5, seed=None):
    """Trial-wise k-fold cross-validated Kalman decoding; returns pooled held-out (true, pred).

    `states` / `observations` are per-trial arrays (T_i, n_state) and (T_i, n_obs).  Whole trials
    are held out (shuffled k-fold, capped at the trial count): the decoder is fit on the training
    trials and run on each held-out trial, and the held-out true and decoded states are
    concatenated over all folds.  Returns (true (N, n_state), pred (N, n_state)); empty arrays
    (shaped (0, n_state)) when there are fewer than two trials.
    """
    from sklearn.model_selection import KFold

    n_trials = len(states)
    n_state = np.asarray(states[0]).shape[1] if n_trials else 0
    if n_trials < 2:
        return np.empty((0, n_state)), np.empty((0, n_state))

    trues, preds = [], []
    folds = int(min(n_folds, n_trials))
    for train_idx, test_idx in KFold(n_splits=folds, shuffle=True, random_state=seed).split(
            range(n_trials)):
        decoder = KalmanFilterDecoder().fit(
            [states[i] for i in train_idx], [observations[i] for i in train_idx])
        for i in test_idx:
            trues.append(np.asarray(states[i], dtype=float))
            preds.append(decoder.predict(observations[i]))
    return np.vstack(trues), np.vstack(preds)


class WienerFilterDecoder:
    """Wiener (linear FIR) filter decoder: continuous state = behaviour, obs = firing rates.

    A regularized least-squares (ridge) map from the neural observation z_t to the continuous
    state x_t.  The observation can be augmented with `n_history_taps` preceding bins so the map
    is a finite-impulse-response filter over recent neural history (the classic Wiener filter for
    neural decoding, e.g. Carmena et al. 2003); n_history_taps=0 makes it an instantaneous linear
    regression.  The design columns are standardized (mean / std learned on the training samples)
    so the ridge penalty `alpha` is scale-invariant across units / principal components, and a
    bias term is always included and left unpenalized.  fit() solves the normal equations over all
    aligned training samples; predict() applies the filter to one trial.  Pure numpy (linear
    algebra only), so it stays in tools alongside KalmanFilterDecoder and shares its fit(states,
    observations) / predict(observations) interface (usable through make_decoder /
    decode_cv_per_trial).
    """

    def __init__(self, n_history_taps=0, alpha=1.0):
        self.n_history_taps = int(n_history_taps)
        self.alpha = float(alpha)
        self.W = None            # (n_features + 1, n_state), last row the bias
        self._mean = self._std = None

    def _lagged(self, observations):
        """One trial's (T, n_obs) observations -> (T, n_obs * (n_history_taps + 1)) design.

        Column block k (k = 0..n_history_taps) is the observation delayed by k bins, the leading
        k rows edge-padded with the first sample so every row is defined.
        """
        Z = np.asarray(observations, dtype=float)
        if Z.ndim == 1:
            Z = Z[:, np.newaxis]
        blocks = [Z]
        for k in range(1, self.n_history_taps + 1):
            shifted = np.empty_like(Z)
            shifted[k:] = Z[:-k]
            shifted[:k] = Z[0]   # edge-pad the leading bins with the first sample
            blocks.append(shifted)
        return np.hstack(blocks) if len(blocks) > 1 else Z

    def _design(self, observations):
        """Standardized lagged design with a trailing bias column (predict path)."""
        feats = (self._lagged(observations) - self._mean) / self._std
        return np.hstack([feats, np.ones((feats.shape[0], 1))])

    def fit(self, states, observations):
        """Fit from per-trial sequences: `states` (T_i, n_state), `observations` (T_i, n_obs).

        Standardizes the (lagged) design by the training mean / std, then solves the ridge normal
        equations (X^T X + alpha I) W = X^T Y with no penalty on the bias.  Trial boundaries need
        no special handling -- every aligned sample contributes one row.
        """
        feats = np.vstack([self._lagged(z) for z in observations])
        self._mean = feats.mean(axis=0)
        self._std = feats.std(axis=0)
        self._std[self._std == 0] = 1.0
        X = np.hstack([(feats - self._mean) / self._std, np.ones((feats.shape[0], 1))])
        Y = np.vstack([np.asarray(s, dtype=float) for s in states])
        reg = self.alpha * np.eye(X.shape[1])
        reg[-1, -1] = 0.0        # do not penalize the bias term
        try:
            self.W = np.linalg.solve(X.T @ X + reg, X.T @ Y)
        except np.linalg.LinAlgError:
            self.W = np.linalg.pinv(X.T @ X + reg) @ (X.T @ Y)
        return self

    def predict(self, observations):
        """Decode one trial: (T, n_obs) observations -> (T, n_state) state estimate."""
        return self._design(observations) @ self.W


DECODE_METHODS = ('wiener', 'kalman')


def make_decoder(method='wiener', wiener_taps=0, wiener_alpha=1.0):
    """Zero-argument factory returning a fresh decoder of the requested method.

    'wiener' -> WienerFilterDecoder(n_history_taps=wiener_taps, alpha=wiener_alpha);
    'kalman' -> KalmanFilterDecoder().  Returned as a callable so each cross-validation fold gets
    an independent decoder (decode_cv_per_trial calls it once per fold).
    """
    method = str(method).lower()
    if method == 'wiener':
        return lambda: WienerFilterDecoder(n_history_taps=wiener_taps, alpha=wiener_alpha)
    if method == 'kalman':
        return lambda: KalmanFilterDecoder()
    raise ValueError('Unknown decode method {!r}; expected one of {}.'.format(
        method, DECODE_METHODS))


def _identity_transform(observations):
    """No-op neural-observation transform (used when no PCA reduction is requested)."""
    return np.asarray(observations, dtype=float)


def _fit_neural_reducer(train_observations, n_pcs, seed=None):
    """Standardize-then-PCA reducer fit on the training observations only (leak-free per fold).

    Fits a StandardScaler and PCA(n_components=min(n_pcs, n_features, n_samples)) on the stacked
    training observations and returns a transform(obs) -> (T, k) callable; when `n_pcs` is falsy
    returns the identity transform (raw rates).  scikit-learn is imported lazily.
    """
    if not n_pcs:
        return _identity_transform

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = np.vstack([np.asarray(z, dtype=float) for z in train_observations])
    scaler = StandardScaler().fit(X)
    k = int(min(n_pcs, X.shape[1], X.shape[0]))
    pca = PCA(n_components=k, random_state=seed).fit(scaler.transform(X))

    def transform(observations):
        return pca.transform(scaler.transform(np.asarray(observations, dtype=float)))
    return transform


def decode_cv_per_trial(states, observations, decoder_factory, n_folds=5, n_pcs=None, seed=None):
    """Trial-wise k-fold cross-validated decoding that keeps each held-out trial's (true, pred).

    `states` / `observations` are per-trial arrays (T_i, n_state) and (T_i, n_obs).  Whole trials
    are held out (shuffled k-fold, capped at the trial count): on each fold a fresh decoder from
    the `decoder_factory()` callable (zero-arg, returning a fit()/predict() object; see
    make_decoder) is fit on the training trials and run on each held-out trial.  When `n_pcs` is
    set the neural observations are reduced to their top n_pcs principal components by a
    scaler + PCA fit on that fold's training trials only (_fit_neural_reducer; no leakage).
    Unlike kalman_decode_cv, the held-out results are not concatenated: returns
    (trues, preds, order) where trues[k] / preds[k] are one held-out trial's (T_k, n_state) true
    and decoded state and order[k] is that trial's index into the inputs (fold order).  Empty
    lists when there are fewer than two trials.
    """
    from sklearn.model_selection import KFold

    n_trials = len(states)
    if n_trials < 2:
        return [], [], []

    folds = int(min(n_folds, n_trials))
    trues, preds, order = [], [], []
    for train_idx, test_idx in KFold(n_splits=folds, shuffle=True, random_state=seed).split(
            range(n_trials)):
        transform = _fit_neural_reducer([observations[i] for i in train_idx], n_pcs, seed)
        decoder = decoder_factory().fit(
            [states[i] for i in train_idx],
            [transform(observations[i]) for i in train_idx])
        for i in test_idx:
            trues.append(np.asarray(states[i], dtype=float))
            preds.append(decoder.predict(transform(observations[i])))
            order.append(int(i))
    return trues, preds, order


def shuffle_trial_alignment(states, seed=None, min_shift_fraction=0.1):
    """Break each trial's within-trial temporal alignment (a time-alignment chance null).

    Circularly rolls every trial's state by an independent random offset, so the neural
    observation no longer lines up with the behaviour in time while each trial's marginal
    distribution and autocorrelation -- hence its per-trial coefficient-of-determination baseline
    -- are preserved.  Offsets are at least `min_shift_fraction` of the trial length away from 0
    (and from a full wrap) so the alignment is genuinely broken.  Returns a new list; the inputs
    are not modified.  Feeding the result through the same decode_cv_per_trial as the real states
    yields the shuffled (chance) distribution of per-trial scores.
    """
    rng = np.random.RandomState(seed)
    shuffled = []
    for s in states:
        s = np.asarray(s, dtype=float)
        n = s.shape[0]
        if n < 3:
            shuffled.append(s.copy())
            continue
        lo = max(1, int(round(min_shift_fraction * n)))
        hi = n - lo
        shift = int(rng.randint(lo, hi + 1)) if hi >= lo else int(rng.randint(1, n))
        shuffled.append(np.roll(s, shift, axis=0))
    return shuffled


def coefficient_of_determination(true, pred):
    """Coefficient of determination R2 of a decoded vs true trace, pooled over its dimensions.

    R2 = 1 - SS_res / SS_tot, with SS_tot taken about the true mean per dimension and both sums
    pooled over dimensions -- the fraction of the true variance the prediction captures.  Ranges
    from negative (worse than predicting the mean) through 0 to 1 (perfect).  For continuous
    decoding this is the per-trial 'pseudo-R2' the force-decoding figures report.  Returns NaN
    when the true trace is constant (SS_tot == 0).
    """
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if true.ndim == 1:
        true = true[:, np.newaxis]
    pred = pred.reshape(true.shape)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - true.mean(axis=0)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def rmse(true, pred):
    """Root mean squared error between a decoded and true trace (pooled over all samples/dims).

    Divide by a scale (e.g. the signal's range) for a normalized RMSE (NRMSE).
    """
    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((true - pred) ** 2)))


def pearson_r(true, pred):
    """Pearson correlation between a decoded and true trace (both raveled).

    Returns NaN when either is constant (correlation undefined).  Raveling pools all dimensions,
    so this is the natural per-trial score for a scalar target such as the summed grasp force.
    """
    true = np.asarray(true, dtype=float).ravel()
    pred = np.asarray(pred, dtype=float).ravel()
    if true.size < 2 or true.std() == 0 or pred.std() == 0:
        return np.nan
    return float(np.corrcoef(true, pred)[0, 1])
