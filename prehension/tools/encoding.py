#!python3
# -*- coding: utf-8 -*-
"""
Generic Poisson GLM encoding: cross-validated fit and pseudo-R2 goodness of fit.

Fits a Poisson generalized linear model f ~ Poisson(exp(X.beta)) of a single response
(a neuron's / channel's binned, smoothed firing rate) on a design matrix of behavioural
covariates, and scores it with the deviance-based pseudo-R2 (Cameron & Windmeijer 1996)
against a mean-only null model, cross-validated.  An optional predictor-count constraint
selects a subset of predictors with a lasso (L1) selector before the Poisson fit.

These are library-agnostic array transforms (scikit-learn only, imported lazily) so they
live in tools; the prehension-specific data assembly and file output live in
analysis.encoding.

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


def poisson_pr2(y_true, y_pred):
    """Deviance-based Poisson pseudo-R2 of predictions vs a mean-only null model.

    pR2 = 1 - deviance(y, y_pred) / deviance(y, mean(y)) (scikit-learn d2_tweedie_score
    with power=1).  Ranges from negative (worse than the null) through 0 (no better than
    the mean) to 1 (perfect).  Returns NaN when y is constant (null deviance is 0).
    """
    from sklearn.metrics import d2_tweedie_score

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 1e-9, None)
    if np.allclose(y_true, y_true.mean()):
        return np.nan
    return float(d2_tweedie_score(y_true, y_pred, power=1))


def adjusted_pr2(pr2, n_samples, n_predictors):
    """Adjusted pseudo-R2: pR2 penalized for the number of predictors.

    1 - (1 - pR2) * (n - 1) / (n - p - 1), the standard adjusted-R2 correction applied to
    the pseudo-R2 so models with different input dimensionality compare fairly.  Returns
    NaN when there are too few samples (n <= p + 1).
    """
    denom = n_samples - n_predictors - 1
    if denom <= 0:
        return np.nan
    return 1.0 - (1.0 - pr2) * (n_samples - 1) / denom


def select_predictors(X, y, max_predictors, seed=None):
    """Indices of up to `max_predictors` columns of X, chosen by a lasso (L1) selector.

    Standardizes X, fits a Lasso to the variance-stabilized response sqrt(y) (a Gaussian
    surrogate that keeps this scikit-learn only), and keeps the columns with the largest
    absolute coefficients.  Returns all columns when max_predictors is falsy or >= the
    number of columns.  This is the "lasso or otherwise constrain the number of
    predictors" option; the returned indices are sorted.
    """
    p = X.shape[1]
    if not max_predictors or max_predictors >= p:
        return np.arange(p)

    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler

    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    target = np.sqrt(np.clip(np.asarray(y, dtype=float), 0.0, None))
    coef = Lasso(alpha=0.01, max_iter=5000, random_state=seed).fit(Xs, target).coef_
    order = np.argsort(np.abs(coef))[::-1]
    return np.sort(order[:max_predictors])


def fit_poisson_glm_cv(X, y, n_folds=5, alpha=1e-4, max_predictors=None, seed=None):
    """Cross-validated Poisson GLM fit of y on X, scored by pseudo-R2.

    K-fold (shuffled) cross-validation: on each fold the predictors are optionally
    reduced to `max_predictors` by a lasso selector (fit on the training fold), a Poisson
    GLM (scikit-learn PoissonRegressor, L2 penalty `alpha`; a small alpha approximates the
    maximum-likelihood fit) is fit on the training fold, and the held-out pseudo-R2 is
    computed.  Returns (pr2, adj_pr2, pr2_per_fold, fit_info): pr2 the mean held-out
    pseudo-R2 over folds, adj_pr2 that adjusted for the number of predictors used, and
    fit_info the fit specification actually used (for provenance in the saved output).
    """
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    from sklearn.model_selection import KFold

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n_samples, n_features = X.shape

    pr2_per_fold = []
    n_used = n_features
    # Some units' lbfgs / coordinate-descent fits do not fully converge within max_iter; that
    # ConvergenceWarning is expected across many per-unit fits and is suppressed so it does not
    # flood the output (the fitted model is still used for the pseudo-R2).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        for train, test in KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X):
            cols = select_predictors(X[train], y[train], max_predictors, seed)
            n_used = len(cols)
            model = PoissonRegressor(alpha=alpha, max_iter=300)
            model.fit(X[train][:, cols], y[train])
            pr2_per_fold.append(poisson_pr2(y[test], model.predict(X[test][:, cols])))

    pr2 = float(np.nanmean(pr2_per_fold)) if pr2_per_fold else np.nan
    adj = adjusted_pr2(pr2, n_samples, n_used)
    fit_info = {'model': 'poisson_glm', 'regularization': 'l2', 'alpha': alpha,
                'max_predictors': max_predictors, 'n_predictors': int(n_used),
                'n_folds': n_folds, 'n_samples': int(n_samples)}
    return pr2, adj, pr2_per_fold, fit_info


# Per-unit fits share one design matrix X, so it is set once per worker through a pool
# initializer (not pickled per unit), mirroring tools.decoding.  The worker is top-level
# and picklable so it runs under spawn (Windows).
_GLM_CTX = None


def _glm_pool_init(ctx):
    """multiprocessing.Pool initializer: stash the shared (X, fit kwargs) context."""
    global _GLM_CTX
    _GLM_CTX = ctx


def _glm_unit_worker(y):
    """Fit one response column against the shared design matrix (see fit_poisson_glm_cv)."""
    X, n_folds, alpha, max_predictors, seed = _GLM_CTX
    return fit_poisson_glm_cv(X, y, n_folds=n_folds, alpha=alpha,
                              max_predictors=max_predictors, seed=seed)


def fit_glms_over_units(X, Y, n_folds=5, alpha=1e-4, max_predictors=None, processes=1, seed=None,
                        progress=True, desc='GLM units'):
    """Fit a Poisson GLM of every column of Y on X, optionally over a process pool.

    X is (n_samples, n_features); Y is (n_samples, n_units).  Returns a list of
    (pr2, adj_pr2, pr2_per_fold, fit_info) per unit (column of Y), in column order.  The
    units are independent, so they are farmed out to `processes` workers (serial when
    <= 1); the shared design matrix is passed once through the pool initializer.  When
    `progress` (default True), a tqdm bar labelled `desc` reports units as they finish.
    """
    import tqdm

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    tasks = [Y[:, i] for i in range(Y.shape[1])]
    ctx = (X, n_folds, alpha, max_predictors, seed)
    if processes and processes > 1:
        with multiprocessing.Pool(processes=processes, initializer=_glm_pool_init,
                                  initargs=(ctx,)) as pool:
            results = pool.imap(_glm_unit_worker, tasks)   # yields in task order
            if progress:
                results = tqdm.tqdm(results, total=len(tasks), desc=desc, ncols=100)
            return list(results)
    _glm_pool_init(ctx)
    iterator = tqdm.tqdm(tasks, desc=desc, ncols=100) if progress else tasks
    return [_glm_unit_worker(t) for t in iterator]
