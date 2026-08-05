#!python3
# -*- coding: utf-8 -*-
"""
Population aggregation of pooled per-neuron activity (reusable across neural work).

Pure transforms over the pooled ``entries`` / ``sessions_data`` structures produced
by the plotting poolers: a light activity filter, the (condition x time, neuron)
matrix used for PCA, the trial-averaged tensor used for demixed PCA (plus a thin
matlab_dpca wrapper), and a condition-matched pseudo-population for decoding.  None
of these need a NeuralConfig or behavioural meta, so they are shared by both neural
processing and plotting.

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
import numpy as np

from ...tools.logs import rs, ws

MIN_RATE_HZ = 1.0   # light activity selection: keep neurons with mean rate above this

# One condition parameter (label 1) + time (label 2). Group the condition main
# effect with the condition/time interaction ("Object"), and keep the time main
# effect on its own ("Condition-independent").  See figure_spaces_dpca_pooled.
DPCA_COMBINED_PARAMS = [[[1], [1, 2]], [[2]]]


def select_active_neurons(entries, min_rate=MIN_RATE_HZ):
    """Keep neurons whose mean rate (over conditions and time) exceeds min_rate Hz."""
    kept = [e for e in entries if float(np.mean(e['frs_avg'])) > min_rate]
    rs('Activity filter: kept {} / {} neurons with mean rate > {} Hz.'.format(
        len(kept), len(entries), min_rate))
    return kept


def build_condition_matrix(entries):
    """Build the (conditions*time, neurons) matrix of per-condition average activity.

    Uses the conditions (force groups) common to every neuron so the columns align.
    Returns (X, conditions, numbins, labels): X has one column per neuron and rows
    ordered (condition-major) as (condition, timebin).
    """
    if not entries:
        raise ValueError('No neurons to build a matrix from.')

    common = set(entries[0]['group_ids'])
    for e in entries[1:]:
        common &= set(e['group_ids'])
    conditions = sorted(common)
    if not conditions:
        raise ValueError('No condition (force group) is common to all pooled neurons.')
    all_conditions = set().union(*[set(e['group_ids']) for e in entries])
    dropped = sorted(all_conditions - set(conditions))
    if dropped:
        ws('Dropping conditions not present for every neuron: {}.'.format(dropped))

    numbins = entries[0]['frs_avg'].shape[1]
    cols, labels = [], []
    for e in entries:
        gid_to_row = {g: i for i, g in enumerate(e['group_ids'])}
        mat = np.vstack([e['frs_avg'][gid_to_row[c], :] for c in conditions])  # (n_cond, numbins)
        cols.append(mat.reshape(-1))  # condition-major flatten -> (n_cond*numbins,)
        labels.append(e['label'])
    X = np.column_stack(cols)  # (n_cond*numbins, n_neurons)

    # # square root the activity
    # X = np.sqrt(X)
    return X, conditions, numbins, labels


def build_condition_tensors(entries):
    """Build the trial-averaged tensor R (and its s.e.m.) and the shared conditions.

    Returns (R, R_sem, conditions, numbins) with R and R_sem of shape
    (n_neurons, n_conditions, n_time); axis 0 is neurons, axis 1 the condition
    parameter (matlab_dpca label 1) and axis 2 time (label 2).  The conditions are
    those (force groups) common to every neuron, so R columns align.  "Conditions
    specified by averaging" == each neuron's per-condition mean activity (frs_avg),
    with frs_sem carried alongside for the projected-trace error bands.

    Future multiple conditions (parameters of object_id at once): make R
    (n_neurons, n_cond1, n_cond2, ..., n_time); everything downstream keys off the
    matlab_dpca marginalization indices, so it generalises without change.
    """
    if not entries:
        raise ValueError('No neurons to build a tensor from.')

    common = set(entries[0]['group_ids'])
    for e in entries[1:]:
        common &= set(e['group_ids'])
    conditions = sorted(common)
    if not conditions:
        raise ValueError('No condition (force group) is common to all pooled neurons.')
    dropped = sorted(set().union(*[set(e['group_ids']) for e in entries]) - set(conditions))
    if dropped:
        ws('Dropping conditions not present for every neuron: {}.'.format(dropped))

    numbins = entries[0]['frs_avg'].shape[1]
    n_neurons, n_cond = len(entries), len(conditions)
    R = np.zeros((n_neurons, n_cond, numbins))
    R_sem = np.zeros((n_neurons, n_cond, numbins))
    for i_n, e in enumerate(entries):
        gid_to_row = {g: i for i, g in enumerate(e['group_ids'])}
        for i_c, c in enumerate(conditions):
            R[i_n, i_c, :] = e['frs_avg'][gid_to_row[c], :]
            R_sem[i_n, i_c, :] = e['frs_sem'][gid_to_row[c], :]
    return R, R_sem, conditions, numbins


def run_dpca(R, n_components=10, combined_params=DPCA_COMBINED_PARAMS):
    """Run demixed PCA on the trial-averaged tensor R (local 'matlab_dpca' package).

    R: (n_neurons, <condition axes...>, n_time).  Returns (W, V, which_marg,
    expl_var): W is the decoder, V the encoder (columns ordered by explained
    variance), which_marg the 0-based marginalization index of each component, and
    expl_var the matlab_dpca ExplainedVariance for R/W/V.  With a scalar
    n_components, matlab_dpca keeps the top n_components components overall.

    Requires the local 'matlab_dpca' package (pip install -e . from its source
    tree).  Imported lazily; not installed or run here.  No regularization is used
    because only the trial-averaged tensor is available (regularization/noise
    covariance need single trials).
    """
    try:
        from matlab_dpca import dpca, explained_variance
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "demixed PCA requires the local 'matlab_dpca' package: install it with "
            "`pip install -e .` from its source tree.") from e

    W, V, which_marg = dpca(R, n_components, combined_params=combined_params)
    expl_var = explained_variance(R, W, V, combined_params=combined_params)
    return W, V, which_marg, expl_var


def build_pooled_pseudopopulation(sessions_data, seed=0):
    """Concatenate neurons across sessions into one condition-matched pseudo-population.

    Because sessions do not share trials, a pseudo-population is built per condition:
    for every condition common to all sessions, the smallest per-session trial count
    n_c is used, n_c trials are drawn (without replacement, order shuffled independently
    per session so noise correlations are not fabricated) from each session, and the
    per-session neuron blocks are concatenated along the neuron axis.  This yields
    sum_c n_c pseudo-trials, each carrying every session's neurons.

    Returns (X_pooled (n_pseudo_trials, n_pooled_neurons, n_time), labels_pooled,
    conditions).  Raises ValueError if no condition is common to all sessions or if
    the common conditions have too few trials to classify.
    """
    if not sessions_data:
        raise ValueError('No sessions to pool.')

    common = set(np.unique(sessions_data[0]['labels']).tolist())
    for sd in sessions_data[1:]:
        common &= set(np.unique(sd['labels']).tolist())
    conditions = sorted(common)
    if len(conditions) < 2:
        raise ValueError('Fewer than 2 conditions common to all sessions; nothing to classify.')
    dropped = sorted(set().union(
        *[set(sd['labels'].tolist()) for sd in sessions_data]) - set(conditions))
    if dropped:
        ws('Dropping conditions not present in every session for pooling: {}.'.format(dropped))

    rng = np.random.RandomState(seed)
    blocks, labels_pooled = [], []
    for c in conditions:
        n_c = min(int(np.sum(sd['labels'] == c)) for sd in sessions_data)
        if n_c == 0:
            continue
        per_session = []
        for sd in sessions_data:
            idx = np.nonzero(sd['labels'] == c)[0]
            rng.shuffle(idx)
            per_session.append(sd['X'][idx[:n_c]])          # (n_c, n_neurons_s, n_time)
        blocks.append(np.concatenate(per_session, axis=1))   # (n_c, n_pooled_neurons, n_time)
        labels_pooled.extend([c] * n_c)

    if not blocks:
        raise ValueError('No condition has trials in every session; cannot pool.')
    return np.concatenate(blocks, axis=0), np.array(labels_pooled), conditions
