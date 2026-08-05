#!python3
# -*- coding: utf-8 -*-
"""
Time-resolved cross-validated decoding helpers.

Generic, data-agnostic linear-discriminant decoding over an
``(n_samples, n_features, n_time)`` activity tensor: a stratified k-fold
cross-validated accuracy at every time bin (real labels plus one label-shuffle for
a chance estimate), farmed out to a multiprocessing.Pool, and a percentile-based
chance level pooled across time.  scikit-learn is imported lazily inside the worker
so importing this module stays cheap.

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

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return np.nan
    folds = int(min(n_folds, counts.min()))
    if folds < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearDiscriminantAnalysis(), features, labels, cv=skf)
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


def classify_through_time(X, labels, n_folds=5, processes=1, seed=0):
    """Classify the labels at every time bin, optionally over a process pool.

    X is (n_samples, n_features, n_time); each time bin is an independent classification
    (features = the columns at that bin), evaluated on the real labels and once on a
    shuffled copy.  The per-bin work is dispatched to a multiprocessing.Pool of
    `processes` workers (serial when processes <= 1).  Each bin gets a distinct seed
    (seed + bin index) so folds/shuffle are reproducible and differ across bins.

    Returns (accuracy (n_time,), shuffle (n_time,)); empty feature sets give all-nan
    results.  The single-shuffle accuracies are meant to be pooled across time and
    reduced to a chance level by the caller (see chance_level).
    """
    n_time = X.shape[2]
    if X.shape[1] == 0:
        return np.full(n_time, np.nan), np.full(n_time, np.nan)

    tasks = [(np.ascontiguousarray(X[:, :, t]), labels, n_folds, seed + t)
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
