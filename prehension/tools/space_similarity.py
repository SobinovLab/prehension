#!python3
# -*- coding: utf-8 -*-
"""
Generic subspace / dynamics similarity kernels shared by the alignment analysis.

Library-agnostic array transforms (numpy, with scikit-learn imported lazily for its
cross-validation splitter) that quantify how much structure two multivariate time series
X (T, N) and Y (T, M) share, plus the surrogate generators that build a null distribution for
any of them.  Three families:

  * reduced-rank regression (rrr_cv / reduced_rank_regression): cross-validated R2 as a
    function of rank and the B = A @ C factorization, run in either direction;
  * canonical correlation analysis (cca_fit / cca_cv): (ridge-regularized) canonical
    correlations, weights, loadings and a held-out cross-validated version;
  * dynamical similarity analysis (dsa_distance and its pieces delay_embed / dmd_operator /
    procrustes_vector_field_distance): a delay-embedded linear (HAVOK/DMD) operator per
    system compared up to an orthogonal change of basis, wrapping the ``DSA`` package when it
    is importable and falling back to an explicit implementation otherwise.

The surrogate generators (time_shuffle, circular_shift, phase_randomize, trial_shuffle) and
effect_size turn any scalar similarity into an effect size against a null.  These operate
only on arrays (no prehension model / file IO), mirroring tools.encoding and tools.decoding,
so the prehension-specific data assembly, dict schema and file output live in
analysis.neural_kinematic_alignment.

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
import warnings

import numpy as np


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------
def validate_matrix(X, name='X', min_samples=3):
    """Return X as a finite float (n_samples, n_features) 2-D array or raise informative errors.

    Catches the shape / NaN / degenerate-input failures early (before a linear-algebra
    routine turns them into an opaque LinAlgError): a non-2-D array, fewer than
    `min_samples` rows, zero columns, or any non-finite entry all raise a ValueError that
    names `name` and the offending property.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError('{} must be 2-D (n_samples, n_features); got shape {}.'.format(
            name, X.shape))
    if X.shape[0] < min_samples:
        raise ValueError('{} has too few samples ({} < {}) to fit / cross-validate.'.format(
            name, X.shape[0], min_samples))
    if X.shape[1] < 1:
        raise ValueError('{} has no columns (n_features=0).'.format(name))
    if not np.all(np.isfinite(X)):
        n_bad = int(np.sum(~np.isfinite(X)))
        raise ValueError('{} contains {} non-finite value(s) (NaN/Inf); clean or drop them '
                         'before the comparison.'.format(name, n_bad))
    return X


def validate_pair(X, Y, min_samples=3):
    """Validate X and Y and check they share the sample axis; returns (X, Y) as float arrays.

    Both are run through validate_matrix (2-D, enough samples, all finite -- hard errors),
    then their row counts must match (the two spaces are compared sample-for-sample, so a T
    mismatch is a hard error rather than a silent truncation).  Constant (zero-variance)
    columns -- a dead unit, an unmoved DOF -- make a covariance rank-deficient, so they are
    reported with a warning (the regularized defaults tolerate them; unregularized fits will
    not), naming the offending columns rather than failing silently or aborting the run.
    """
    X = validate_matrix(X, 'X', min_samples)
    Y = validate_matrix(Y, 'Y', min_samples)
    if X.shape[0] != Y.shape[0]:
        raise ValueError('X and Y must have the same number of samples (rows); got {} vs {}. '
                         'They are compared timepoint-for-timepoint.'.format(
                             X.shape[0], Y.shape[0]))
    for name, A in (('X', X), ('Y', Y)):
        const = np.flatnonzero(np.std(A, axis=0) == 0)
        if const.size:
            warnings.warn(
                '{} has {} constant (zero-variance) column(s) at indices {} -- rank deficient; '
                'the regularized (ridge) fits tolerate them, but drop them for an unregularized '
                'run.'.format(name, const.size, const.tolist()[:10]), stacklevel=2)
    return X, Y


# --------------------------------------------------------------------------------------
# Block standardization / whitening / weighting (for concatenated multi-unit Y)
# --------------------------------------------------------------------------------------
def blocks_from_sizes(sizes):
    """Contiguous (start, stop) column ranges for concatenated blocks of the given sizes."""
    edges = np.concatenate([[0], np.cumsum(sizes)])
    return [(int(edges[i]), int(edges[i + 1])) for i in range(len(sizes))]


def standardize_blocks(Y, blocks):
    """Z-score each block of columns of Y independently (mean 0, unit variance per channel).

    `blocks` is a list of (start, stop) column ranges (blocks_from_sizes).  Standardizing
    per block, rather than globally, is what lets blocks with very different units and
    variances (e.g. joint angles in degrees vs forces in newtons) enter a shared analysis
    on comparable footing.  Zero-variance columns are left unscaled.  Returns a new array.
    """
    Y = np.asarray(Y, dtype=float).copy()
    for a, b in blocks:
        seg = Y[:, a:b]
        mean = seg.mean(axis=0)
        std = seg.std(axis=0)
        std[std == 0] = 1.0
        Y[:, a:b] = (seg - mean) / std
    return Y


def whiten_block(Y, blocks, reg=1e-6):
    """ZCA-whiten each block so its channels are decorrelated with unit variance.

    After whitening, a block contributes an isotropic, unit-variance subspace: a block's
    influence then scales with its *number of channels*, not its raw variance or internal
    correlations.  `reg` is added to the eigenvalues for numerical stability.  Use with
    block_weights=1/sqrt(n_channels) (apply_block_weights) when you additionally want to
    stop a wide block from swamping a narrow one by sheer channel count.  Returns a new array.
    """
    Y = np.asarray(Y, dtype=float).copy()
    for a, b in blocks:
        seg = Y[:, a:b]
        seg = seg - seg.mean(axis=0)
        cov = seg.T @ seg / max(seg.shape[0] - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, reg, None)
        whitening = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
        Y[:, a:b] = seg @ whitening
    return Y


def apply_block_weights(Y, blocks, weights):
    """Scale each block of columns by a scalar weight (per-block).

    `weights` is one scalar per block (or the string 'equal' for 1/sqrt(n_channels), which
    equalizes each block's total variance contribution regardless of how many channels it
    has -- e.g. so 30 position channels do not swamp 3 force channels).  Returns a new array.
    """
    Y = np.asarray(Y, dtype=float).copy()
    if isinstance(weights, str):
        if weights != 'equal':
            raise ValueError("block weights string must be 'equal'; got {!r}.".format(weights))
        weights = [1.0 / np.sqrt(b - a) for a, b in blocks]
    if len(weights) != len(blocks):
        raise ValueError('Expected one weight per block ({}); got {}.'.format(
            len(blocks), len(weights)))
    for (a, b), w in zip(blocks, weights):
        Y[:, a:b] *= float(w)
    return Y


# --------------------------------------------------------------------------------------
# Reduced-rank regression
# --------------------------------------------------------------------------------------
def _ridge_coef(X, Y, alpha):
    """Ridge OLS coefficient B = (X'X + alpha I)^-1 X'Y for centred X, Y (n_features, n_targets)."""
    p = X.shape[1]
    gram = X.T @ X + alpha * np.eye(p)
    return np.linalg.solve(gram, X.T @ Y)


def reduced_rank_regression(X, Y, rank, alpha=1.0):
    """Reduced-rank ridge regression of Y on X at a fixed rank; returns the B = A @ C factors.

    Fits the full ridge coefficient B_full = (X'X + alpha I)^-1 X'Y, then constrains it to
    `rank` by projecting the fitted response onto its leading `rank` singular directions
    (Reinsel & Velu): B = B_full V_r V_r', factorized as A = B_full V_r (n_features, rank)
    and C = V_r' (rank, n_targets).  X and Y are centred internally.  Returns a dict with
    'A', 'C', 'B', the intercept, and the singular values of the fitted response (so the
    rank spectrum can be inspected).  Rank is clamped to [1, min(n_features, n_targets)].
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
    Xc, Yc = X - x_mean, Y - y_mean

    B_full = _ridge_coef(Xc, Yc, alpha)          # (p, q)
    Yhat = Xc @ B_full                           # (T, q)
    _, svals, Vt = np.linalg.svd(Yhat, full_matrices=False)
    r = int(np.clip(rank, 1, min(X.shape[1], Y.shape[1], Vt.shape[0])))
    Vr = Vt[:r].T                                # (q, r)
    A = B_full @ Vr                              # (p, r)
    C = Vr.T                                     # (r, q)
    intercept = y_mean - x_mean @ (A @ C)
    return {'A': A, 'C': C, 'B': A @ C, 'intercept': intercept,
            'singular_values': svals, 'rank': r}


def rrr_cv(X, Y, ranks=None, alpha=1.0, n_folds=5, seed=None, return_fit=True):
    """Cross-validated reduced-rank regression of Y on X: R2 as a function of rank.

    K-fold (shuffled) cross-validation: on each fold the RRR is fit on the training rows and
    scored on the held-out rows for every rank in `ranks` (default 1..min(p, q)).  The
    held-out squared errors are accumulated across folds so each rank gets one honest R2
    (residual sums pooled over all out-of-fold predictions -- more stable than averaging
    noisy per-fold R2s, and without materializing a prediction matrix per rank).  Held-out
    R2 is scored against the *training* mean (the baseline a predictor that never saw the
    test fold actually beats).  A final RRR is then fit on all the data at the selected
    (R2-maximizing) rank to return the B = A @ C factorization.  Returns a dict:
      'ranks', 'cv_r2' (per rank), 'selected_rank', 'cv_r2_selected',
      'per_target_r2' (held-out, at the selected rank), 'A', 'C', 'B', 'intercept',
      'singular_values', 'n_samples', 'n_features', 'n_targets', 'alpha'.
    R2 is the fraction of Y's variance the rank-constrained neural->behaviour (or reverse)
    map predicts out of sample; it is the reduced-rank analogue of the encoding pseudo-R2.
    """
    from sklearn.model_selection import KFold

    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    T, p = X.shape
    q = Y.shape[1]
    max_rank = min(p, q)
    if ranks is None:
        ranks = list(range(1, max_rank + 1))
    ranks = [int(r) for r in ranks if 1 <= int(r) <= max_rank]
    if not ranks:
        raise ValueError('No valid ranks in [1, {}] for X ({}, {}) -> Y (*, {}).'.format(
            max_rank, T, p, q))
    folds = int(min(n_folds, T))
    if folds < 2:
        raise ValueError('Need at least 2 samples per fold; got T={} for {} folds.'.format(
            T, n_folds))

    # accumulate held-out squared errors per rank (and the reference total) across folds,
    # so R2 is pooled over all out-of-fold predictions without storing a (T, q) matrix per rank
    n_ranks = len(ranks)
    ss_res_total = np.zeros(n_ranks)
    ss_res_target = np.zeros((n_ranks, q))
    ss_tot_total = 0.0
    ss_tot_target = np.zeros(q)
    for train, test in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        x_mean, y_mean = X[train].mean(axis=0), Y[train].mean(axis=0)
        B_full = _ridge_coef(X[train] - x_mean, Y[train] - y_mean, alpha)
        _, _, Vt = np.linalg.svd((X[train] - x_mean) @ B_full, full_matrices=False)
        Xte, Yte = X[test] - x_mean, Y[test]
        ref = Yte - y_mean
        ss_tot_total += float(np.sum(ref ** 2))
        ss_tot_target += np.sum(ref ** 2, axis=0)
        for ri, r in enumerate(ranks):
            Vr = Vt[:min(r, Vt.shape[0])].T
            diff = Yte - (Xte @ (B_full @ Vr @ Vr.T) + y_mean)
            ss_res_total[ri] += float(np.sum(diff ** 2))
            ss_res_target[ri] += np.sum(diff ** 2, axis=0)

    cv_r2 = (1.0 - ss_res_total / ss_tot_total) if ss_tot_total > 0 else np.full(n_ranks, np.nan)
    best = int(np.nanargmax(cv_r2))
    selected_rank = ranks[best]
    with np.errstate(invalid='ignore', divide='ignore'):
        per_target = np.where(ss_tot_target > 0, 1.0 - ss_res_target[best] / ss_tot_target, np.nan)

    # the final full-data factorization is skipped in the null loop (return_fit=False), where
    # only cv_r2_selected is read -- it would otherwise be refit once per surrogate draw
    fit = (reduced_rank_regression(X, Y, selected_rank, alpha=alpha) if return_fit else
           {'A': None, 'C': None, 'B': None, 'intercept': None, 'singular_values': None})
    return {'ranks': ranks, 'cv_r2': np.asarray(cv_r2), 'selected_rank': selected_rank,
            'cv_r2_selected': float(cv_r2[best]), 'per_target_r2': per_target,
            'A': fit['A'], 'C': fit['C'], 'B': fit['B'], 'intercept': fit['intercept'],
            'singular_values': fit['singular_values'],
            'n_samples': int(T), 'n_features': int(p), 'n_targets': int(q), 'alpha': float(alpha)}


# --------------------------------------------------------------------------------------
# Canonical correlation analysis
# --------------------------------------------------------------------------------------
def _inv_sqrt_cov(A, alpha):
    """Regularized inverse square root of a covariance: (A'A/(n-1) + alpha I)^{-1/2}.

    Symmetric eigendecomposition with the ridge `alpha` added to the eigenvalues, so it stays
    well conditioned when the feature count approaches the sample count (the classic CCA
    over-fit regime).  A is a centred (n_samples, n_features) matrix.
    """
    n = A.shape[0]
    cov = A.T @ A / max(n - 1, 1) + alpha * np.eye(A.shape[1])
    evals, evecs = np.linalg.eigh(cov)
    evals = np.clip(evals, 1e-12, None)
    return evecs @ np.diag(evals ** -0.5) @ evecs.T


def cca_fit(X, Y, alpha_x=0.0, alpha_y=0.0, n_components=None):
    """(Ridge-regularized) canonical correlation analysis of X and Y.

    Whitens each view with a regularized inverse-sqrt covariance (alpha_x / alpha_y; a
    positive ridge is essential when n_features approaches n_samples, e.g. many neural
    channels), then takes the SVD of the whitened cross-covariance.  Returns a dict:
      'correlations' (in-sample canonical correlations, descending),
      'x_weights' / 'y_weights' (columns map the views to canonical variates),
      'x_loadings' / 'y_loadings' (structure correlations: corr of each original channel
        with each canonical variate -- what the components actually represent),
      'x_scores' / 'y_scores' (the canonical variates), 'n_components'.
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    Xc, Yc = X - X.mean(axis=0), Y - Y.mean(axis=0)
    k = min(X.shape[1], Y.shape[1])
    if n_components:
        k = min(k, int(n_components))

    Kx = _inv_sqrt_cov(Xc, alpha_x)
    Ky = _inv_sqrt_cov(Yc, alpha_y)
    n = Xc.shape[0]
    cross = Kx @ (Xc.T @ Yc / max(n - 1, 1)) @ Ky
    U, svals, Vt = np.linalg.svd(cross, full_matrices=False)

    x_weights = Kx @ U[:, :k]
    y_weights = Ky @ Vt[:k].T
    x_scores = Xc @ x_weights
    y_scores = Yc @ y_weights
    correlations = np.clip(svals[:k], 0.0, 1.0)

    return {'correlations': correlations, 'x_weights': x_weights, 'y_weights': y_weights,
            'x_loadings': _loadings(Xc, x_scores), 'y_loadings': _loadings(Yc, y_scores),
            'x_scores': x_scores, 'y_scores': y_scores, 'n_components': int(k)}


def _loadings(data_c, scores):
    """Structure correlations: Pearson corr of every centred channel with every variate.

    Returns (n_channels, n_components).  A canonical variate's loadings say which original
    channels it is built from -- e.g. whether a shared component lives in the position block
    or the force block.
    """
    n_ch = data_c.shape[1]
    n_comp = scores.shape[1]
    out = np.full((n_ch, n_comp), np.nan)
    s_std = scores.std(axis=0)
    d_std = data_c.std(axis=0)
    for j in range(n_comp):
        if s_std[j] == 0:
            continue
        for i in range(n_ch):
            if d_std[i] == 0:
                continue
            out[i, j] = np.corrcoef(data_c[:, i], scores[:, j])[0, 1]
    return out


def cca_cv(X, Y, alpha_x=0.0, alpha_y=0.0, n_components=None, n_folds=5, seed=None,
           n_perm=100, sig_percentile=95.0, return_fit=True):
    """Cross-validated CCA: canonical correlations measured on held-out folds, not in sample.

    In-sample canonical correlations are upward biased (badly so when n_features approaches
    n_samples): the weights are fit to maximize them.  Here the weights are fit on the
    training fold and the correlation of the projected canonical variates is measured on the
    held-out fold, pooled across folds -- the honest number.  A component is counted
    significant if its held-out correlation exceeds the `sig_percentile` of a row-permutation
    null (pairing destroyed) built from the same held-out projections.  Returns a dict:
      'cv_correlations' (held-out, per component), 'n_significant', 'sig_threshold',
      plus the full-data 'fit' (cca_fit) for the weights / loadings.
    """
    from sklearn.model_selection import KFold

    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    T = X.shape[0]
    k = min(X.shape[1], Y.shape[1])
    if n_components:
        k = min(k, int(n_components))
    folds = int(min(n_folds, T))
    if folds < 2:
        raise ValueError('Need at least 2 folds for cross-validated CCA; T={}.'.format(T))

    # held-out canonical variates per fold; the correlation is measured PER FOLD and averaged,
    # which is invariant to the SVD sign ambiguity of the per-fold canonical weights (pooling
    # the projections across folds and correlating once is not -- the arbitrary per-fold sign
    # would partially cancel).
    xs_folds, ys_folds = [], []
    for train, test in KFold(n_splits=folds, shuffle=True, random_state=seed).split(X):
        fit = cca_fit(X[train], Y[train], alpha_x, alpha_y, n_components=k)
        xs_folds.append((X[test] - X[train].mean(axis=0)) @ fit['x_weights'])
        ys_folds.append((Y[test] - Y[train].mean(axis=0)) @ fit['y_weights'])

    def _fold_mean_corr(perm_rng=None):
        vals = np.full((len(xs_folds), k), np.nan)
        for f, (xt, yt) in enumerate(zip(xs_folds, ys_folds)):
            yy = yt[perm_rng.permutation(yt.shape[0])] if perm_rng is not None else yt
            for j in range(k):
                vals[f, j] = _safe_corr(xt[:, j], yy[:, j])
        return np.nanmean(vals, axis=0)

    cv_corr = _fold_mean_corr()

    # significance: break the neural<->behaviour pairing within each fold and take the
    # sig_percentile of the resulting (chance) |correlation| over permutations and components
    rng = np.random.RandomState(seed)
    perm = []
    for _ in range(int(n_perm)):
        perm.extend(np.abs(_fold_mean_corr(perm_rng=rng)).tolist())
    perm = [v for v in perm if np.isfinite(v)]
    sig_threshold = float(np.nanpercentile(perm, sig_percentile)) if perm else np.nan
    n_significant = int(np.sum(np.abs(cv_corr) > sig_threshold)) if np.isfinite(sig_threshold) \
        else int(np.sum(cv_corr > 0))

    # the full-data fit (weights / loadings) is skipped in the null loop (return_fit=False),
    # where only the leading cross-validated correlation is read
    return {'cv_correlations': cv_corr, 'n_significant': n_significant,
            'sig_threshold': sig_threshold, 'n_components': int(k),
            'fit': cca_fit(X, Y, alpha_x, alpha_y, n_components=k) if return_fit else None}


def _safe_corr(a, b):
    """Pearson correlation of two vectors, or NaN when either is constant."""
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------------------
# Dynamical Similarity Analysis (DSA)
# --------------------------------------------------------------------------------------
def delay_embed(X, n_delays, delay=1):
    """Hankel delay embedding: stack `n_delays` lagged copies of X (spacing `delay`).

    X is (T, d); returns (T - (n_delays - 1) * delay, d * n_delays) where row t is
    [x_t, x_{t-delay}, ..., x_{t-(n_delays-1) delay}].  Delay embedding lifts the observed
    signal into a space where its (possibly nonlinear) dynamics are approximately linear --
    the state on which DSA fits a linear operator (Takens / HAVOK).
    """
    X = np.asarray(X, dtype=float)
    span = (n_delays - 1) * delay
    if span >= X.shape[0]:
        raise ValueError('Delay embedding span {} >= T {}; reduce n_delays ({}) or delay '
                         '({}).'.format(span, X.shape[0], n_delays, delay))
    cols = [X[span - i * delay: X.shape[0] - i * delay] for i in range(n_delays)]
    return np.concatenate(cols, axis=1)


def dmd_operator(Z, rank=None, use_pydmd='auto'):
    """Least-squares linear operator A with Z_{t+1} ~ A Z_t, optionally SVD-reduced to `rank`.

    Fits the DMD / HAVOK linear map on the (delay-embedded) state Z (T, k): when `rank` is
    given the state is first projected onto its top-`rank` singular directions (the HAVOK
    reduction), and A is the reduced-space operator.  Returns (A, basis) where A is (r, r)
    and basis is (k, r) mapping the full state -> reduced coordinates (identity when rank is
    None / >= k).

    `use_pydmd`: 'auto' (default) uses the reduced operator (``dmd.operator.as_numpy_array``)
    from the ``PyDMD`` package when it is importable and a rank is given, and falls back to
    the numpy least-squares fit otherwise; True requires PyDMD (raises an ImportError naming
    the pip install if missing); False forces the numpy fit.  The numpy fit is the exact
    reduced-space least-squares operator, so the fallback is not an approximation -- PyDMD is
    only an alternative/validated backend.
    """
    Z = np.asarray(Z, dtype=float)
    Z1, Z2 = Z[:-1], Z[1:]
    if rank is not None and rank < Z.shape[1]:
        # project onto the leading singular directions of the state (the HAVOK spatial modes)
        _, _, Vt = np.linalg.svd(Z1, full_matrices=False)
        basis = Vt[:rank].T                       # (k, r)
        A = _pydmd_reduced_operator(Z, rank, use_pydmd)
        if A is None:
            Z1r, Z2r = Z1 @ basis, Z2 @ basis
            A = Z2r.T @ Z1r @ np.linalg.pinv(Z1r.T @ Z1r)
        return A, basis
    if use_pydmd is True:
        raise ImportError(
            "use_pydmd=True requires a rank (PyDMD fits a rank-reduced operator); pass a rank "
            "or use_pydmd=False.")
    A = Z2.T @ Z1 @ np.linalg.pinv(Z1.T @ Z1)
    return A, np.eye(Z.shape[1])


def _pydmd_reduced_operator(Z, rank, use_pydmd):
    """Reduced DMD operator (r, r) via PyDMD, or None to signal the numpy fallback.

    Honours `use_pydmd` ('auto'/True/False): returns None (fall back) when PyDMD is absent
    and use_pydmd != True, and raises a clear ImportError naming the pip install when it is
    required (True) but missing.
    """
    if use_pydmd is False:
        return None
    try:
        from pydmd import DMD
    except Exception:  # noqa: BLE001 - not installed / import error
        if use_pydmd is True:
            raise ImportError(
                "The 'PyDMD' package is required for use_pydmd=True but is not importable. "
                "Install it with `pip install pydmd`, or pass use_pydmd=False (or 'auto') to "
                "use the built-in numpy least-squares DMD fit.")
        return None
    dmd = DMD(svd_rank=int(rank))
    dmd.fit(Z.T)                                   # PyDMD expects (space, time)
    # the reduced DMD operator can be complex (conjugate eigenpairs); the Procrustes
    # comparison is over real operators, so take the real part (imaginary parts cancel for
    # real data and are tiny otherwise)
    return np.real(np.asarray(dmd.operator.as_numpy_array))


def procrustes_vector_field_distance(A1, A2, n_restarts=5, n_iter=200, lr=0.05, seed=None):
    """Distance between two linear vector fields up to an orthogonal change of basis.

    Minimizes f(C) = ||A1 C - C A2||_F over the orthogonal group O(k) by projected gradient
    descent (retracting to the nearest orthogonal matrix each step) from several random
    starts, then normalizes by the operators' scale.  This is the Procrustes-over-vector-
    fields comparison at the heart of DSA (Ostrow et al. 2023): two systems are similar when
    one operator becomes the other under a rotation of the state space.  A1 and A2 must be
    square and the same size (the caller truncates to a common rank).  Returns
    (distance, C_best) with distance in [0, ~2]; 0 means the fields coincide under C_best.
    """
    A1, A2 = np.asarray(A1, dtype=float), np.asarray(A2, dtype=float)
    if A1.shape != A2.shape or A1.shape[0] != A1.shape[1]:
        raise ValueError('procrustes_vector_field_distance needs two square operators of the '
                         'same size; got {} and {}.'.format(A1.shape, A2.shape))
    k = A1.shape[0]
    rng = np.random.RandomState(seed)
    scale = 0.5 * (np.linalg.norm(A1) + np.linalg.norm(A2)) or 1.0

    best_f, best_C = np.inf, np.eye(k)
    for r in range(max(1, n_restarts)):
        C = np.eye(k) if r == 0 else _nearest_orthogonal(rng.standard_normal((k, k)))
        for _ in range(n_iter):
            R = A1 @ C - C @ A2
            grad = A1.T @ R - R @ A2.T          # d/dC ||A1 C - C A2||_F^2
            C = _nearest_orthogonal(C - lr * grad)
        f = np.linalg.norm(A1 @ C - C @ A2)
        if f < best_f:
            best_f, best_C = f, C
    return float(best_f / scale), best_C


def _nearest_orthogonal(M):
    """Nearest orthogonal matrix to M (polar factor via SVD): U V' from M = U S V'."""
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    return U @ Vt


def dsa_distance(X, Y, n_delays=10, delay=1, rank=10, use_package='auto',
                 use_pydmd='auto', n_restarts=5, seed=None):
    """Dynamical Similarity Analysis distance between two systems' delay-embedded dynamics.

    Pipeline (explicit fallback): delay-embed X and Y (delay_embed), fit a reduced linear
    HAVOK/DMD operator to each (dmd_operator at `rank`), and compare the two operators up to
    an orthogonal change of basis (procrustes_vector_field_distance).  Lower distance = more
    similar dynamics; unlike RRR / CCA it does not require X and Y to be sampled on the same
    clock, only to have comparable dynamics.

    `use_package`: 'auto' (default) uses the ``DSA`` package when importable and falls back
    to the explicit pipeline otherwise; True requires ``DSA`` (raises a clear ImportError
    naming the pip install if missing); False forces the explicit pipeline.  Returns a dict:
      'distance', 'A_x', 'A_y' (the fitted operators, for inspection), 'rank_used',
      'n_delays', 'delay', 'backend' ('dsa_package' or 'explicit').
    """
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)

    if use_package in ('auto', True):
        dsa_mod = _import_dsa()
        if dsa_mod is None and use_package is True:
            raise ImportError(
                "The 'DSA' package is required for use_package=True but is not importable. "
                "Install it with `pip install DSA` (https://github.com/mitchellostrow/DSA), "
                "or pass use_package=False to use the explicit fallback implementation.")
        if dsa_mod is not None:
            try:
                return _dsa_via_package(dsa_mod, X, Y, n_delays, delay, rank)
            except Exception as e:  # noqa: BLE001
                if use_package is True:
                    raise
                warnings.warn('DSA package call failed ({}); using the explicit fallback. Pass '
                              'use_package=False to silence this.'.format(e), stacklevel=2)

    Ex, Ey = delay_embed(X, n_delays, delay), delay_embed(Y, n_delays, delay)
    eff_rank = int(min(rank, Ex.shape[1], Ey.shape[1], Ex.shape[0] - 1, Ey.shape[0] - 1))
    if eff_rank < 1:
        raise ValueError('DSA rank collapses to {} (n_delays={}, delay={}, T too small); '
                         'reduce n_delays / delay or provide more data.'.format(
                             eff_rank, n_delays, delay))
    A_x, _ = dmd_operator(Ex, rank=eff_rank, use_pydmd=use_pydmd)
    A_y, _ = dmd_operator(Ey, rank=eff_rank, use_pydmd=use_pydmd)
    dist, _ = procrustes_vector_field_distance(A_x, A_y, n_restarts=n_restarts, seed=seed)
    return {'distance': float(dist), 'A_x': A_x, 'A_y': A_y, 'rank_used': eff_rank,
            'n_delays': int(n_delays), 'delay': int(delay), 'backend': 'explicit'}


def _import_dsa():
    """Return the imported ``DSA`` module, or None when it is not installed."""
    try:
        import DSA  # noqa: F401
        return DSA
    except Exception:  # noqa: BLE001 - any import failure -> use the fallback
        return None


def _dsa_via_package(dsa_mod, X, Y, n_delays, delay, rank):
    """Compute the DSA distance through the installed ``DSA`` package (Ostrow et al.).

    fit_score() returns either a scalar distance or a pairwise distance matrix; for the
    2-system case the cross-distance is the off-diagonal entry (the diagonal is 0), so a
    2-D result is read at [0, -1] rather than [0, 0].
    """
    dsa = dsa_mod.DSA(X, Y, n_delays=n_delays, delay=delay, rank=rank)
    score = np.asarray(dsa.fit_score(), dtype=float)
    dist = float(score[0, -1]) if score.ndim >= 2 else float(score.ravel()[0])
    A_x = np.asarray(getattr(dsa, 'dmd1', np.nan))
    A_y = np.asarray(getattr(dsa, 'dmd2', np.nan))
    return {'distance': dist, 'A_x': A_x, 'A_y': A_y, 'rank_used': int(rank),
            'n_delays': int(n_delays), 'delay': int(delay), 'backend': 'dsa_package'}


# --------------------------------------------------------------------------------------
# Surrogates / null distributions
# --------------------------------------------------------------------------------------
def _trial_slices(n_rows, trials):
    """(start, stop) row ranges for concatenated trials of the given lengths; validates sum."""
    if sum(trials) != n_rows:
        raise ValueError('Trial lengths sum to {} but the data has {} rows.'.format(
            sum(trials), n_rows))
    return blocks_from_sizes(trials)


def time_shuffle(Y, rng, trials=None):
    """Randomly permute the time (row) order of Y -- destroys all temporal structure.

    The most aggressive null: it breaks both Y's autocorrelation and its alignment to X, so
    any similarity metric collapses to its floor.  Useful as a sanity check that the metric
    reads ~0 on unrelated data; the temporally-structured nulls (circular_shift,
    phase_randomize) are the more honest comparison.  `trials` is ignored (kept for a uniform
    surrogate signature).
    """
    Y = np.asarray(Y, dtype=float)
    return Y[rng.permutation(Y.shape[0])]


def circular_shift(Y, rng, trials=None, min_shift=1):
    """Circularly shift Y in time by a random offset -- preserves each channel's autocorrelation.

    Rolling Y relative to X keeps Y's own temporal structure (autocorrelation, spectra,
    within-Y cross-correlations) intact while destroying its timepoint alignment to X, so the
    null captures "two signals with this much temporal smoothness, but unrelated".  This is
    the recommended default null for continuous neural/behaviour traces.  With `trials`
    (per-trial row counts) each trial is shifted independently, so the shift never mixes
    across trial boundaries.
    """
    Y = np.asarray(Y, dtype=float)
    if trials is None:
        s = rng.randint(min_shift, max(min_shift + 1, Y.shape[0]))
        return np.roll(Y, s, axis=0)
    out = np.empty_like(Y)
    for a, b in _trial_slices(Y.shape[0], trials):
        n = b - a
        s = rng.randint(min_shift, max(min_shift + 1, n)) if n > 1 else 0
        out[a:b] = np.roll(Y[a:b], s, axis=0)
    return out


def phase_randomize(Y, rng, trials=None, independent=True):
    """Phase-randomized surrogate: keep each channel's power spectrum, randomize its phases.

    FFT each channel and rotate its phases by random amounts (with DC / Nyquist kept real so
    the inverse transform is real), then invert.  By construction the surrogate has the same
    power spectrum -- hence the same autocorrelation -- as the original channel, but no phase
    relationship to X.  With independent=True (default) each channel is given independent
    random phases (the classic univariate spectrum-preserving surrogate -- scrambles both the
    coupling to X and Y's internal cross-structure); independent=False rotates every channel
    by the *same* random phase (Prichard & Theiler 1994), which additionally preserves Y's
    cross-spectrum / inter-channel covariance and only destroys the coupling to X.  For DSA
    prefer the independent form: the multivariate one preserves Y's dynamics almost entirely
    (the operator is set by the cross-spectrum), making it a weak null for a dynamics metric.
    `trials` is ignored (the spectrum is defined over the whole record).
    """
    Y = np.asarray(Y, dtype=float)
    T = Y.shape[0]
    fft = np.fft.rfft(Y, axis=0)
    n_freq = fft.shape[0]

    def _phases():
        ph = rng.uniform(0, 2 * np.pi, size=n_freq)
        ph[0] = 0.0                                   # DC stays real
        if T % 2 == 0:
            ph[-1] = 0.0                              # Nyquist stays real
        return ph

    if independent:
        # rotate each channel's original spectrum by its own random phase (keeps |FFT| per
        # channel, i.e. the per-channel power spectrum; destroys the cross-structure)
        rot = np.stack([np.exp(1j * _phases()) for _ in range(Y.shape[1])], axis=1)
        new = fft * rot
    else:
        # one shared rotation applied to the original spectra -> preserves the cross-spectrum
        new = fft * np.exp(1j * _phases())[:, None]
    return np.fft.irfft(new, n=T, axis=0)


def trial_shuffle(Y, rng, trials=None):
    """Permute whole trials of Y -- breaks the trial-to-trial correspondence with X.

    Requires `trials` (per-trial row counts).  Reorders Y's trial blocks so trial i of Y is
    paired with a different trial of X, preserving each trial's internal structure while
    destroying the across-trial pairing.  The natural null for a trialized design: it holds
    the within-trial dynamics fixed and asks whether the *matching* of neural and behavioural
    trials carries the similarity.  Falls back to a whole-record time_shuffle-free identity
    when there is only one trial (nothing to permute) -- the caller should prefer another
    surrogate then.
    """
    Y = np.asarray(Y, dtype=float)
    if not trials:
        raise ValueError("trial_shuffle needs per-trial row counts ('trials'); none given.")
    slices = _trial_slices(Y.shape[0], trials)
    if len(slices) < 2:
        return Y.copy()
    order = rng.permutation(len(slices))
    return np.vstack([Y[slices[i][0]:slices[i][1]] for i in order])


SURROGATES = {
    'time_shuffle': time_shuffle,
    'circular_shift': circular_shift,
    'phase_randomize': phase_randomize,
    'trial_shuffle': trial_shuffle,
}


def effect_size(observed, null_values, higher_is_more_similar=True):
    """Summarize an observed similarity against its null distribution as an effect size.

    Reports the standardized distance from the null (the headline effect size, in null SDs),
    a robust variant (median / MAD), the empirical percentile of the observed value within
    the null, and a permutation p-value -- all oriented by `higher_is_more_similar` (True for
    R2 / correlation, False for the DSA distance).  The p-value uses the (n_null + 1)
    correction so it can never be exactly 0.  Returns a dict; effect sizes are NaN when the
    null has zero spread.
    """
    null = np.asarray(null_values, dtype=float)
    null = null[np.isfinite(null)]
    n = null.size
    if n == 0:
        return {'z': np.nan, 'robust_z': np.nan, 'percentile': np.nan, 'p_value': np.nan,
                'null_mean': np.nan, 'null_std': np.nan, 'null_median': np.nan, 'n_null': 0,
                'higher_is_more_similar': bool(higher_is_more_similar)}
    mean, std = float(np.mean(null)), float(np.std(null))
    median = float(np.median(null))
    mad = float(np.median(np.abs(null - median))) * 1.4826
    z = (observed - mean) / std if std > 0 else np.nan
    rz = (observed - median) / mad if mad > 0 else np.nan
    percentile = float(np.mean(null <= observed) * 100.0)
    if higher_is_more_similar:
        p = (1 + int(np.sum(null >= observed))) / (n + 1)
    else:
        z = -z if np.isfinite(z) else z         # orient so positive z = more similar
        rz = -rz if np.isfinite(rz) else rz
        p = (1 + int(np.sum(null <= observed))) / (n + 1)
    return {'z': float(z) if np.isfinite(z) else np.nan,
            'robust_z': float(rz) if np.isfinite(rz) else np.nan,
            'percentile': percentile, 'p_value': float(p),
            'null_mean': mean, 'null_std': std, 'null_median': median, 'n_null': int(n),
            'higher_is_more_similar': bool(higher_is_more_similar)}
