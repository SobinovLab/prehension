#!python3
# -*- coding: utf-8 -*-
"""
Cross-validated classification of an object condition through time, pooled across sessions.

Pools per-trial, causally-smoothed, square-rooted activity across sessions
(common.pooling.pool_trials), then, at every time bin, trains a cross-validated LDA
to predict the trial's condition (tools.decoding).  The classification is run per
session (thin lines) and on a condition-matched pseudo-population pooled across
sessions (common.population.build_pooled_pseudopopulation; thick line).  A second
session set can be classified separately and overlaid.  Chance is estimated by
label-shuffling (see tools.decoding.chance_level); the theoretical chance level is
1 / n_conditions.  Plotting lives in neural_plotting.common.traces.

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

from ..tools.logs import rs, ws
from ..tools.cmd_args import sessions_name_stub
from ..tools.decoding import classify_through_time, chance_level
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH)
from ..neural_processing.common.population import MIN_RATE_HZ, build_pooled_pseudopopulation
from .common.pooling import pool_trials, CAUSAL_SIGMA
from .common.traces import plot_classification_time, resolve_pooled_save_dir

# defaults; overridable through the calling function / script
N_FOLDS = 5                # cross-validation folds (trials held out)
SHUFFLE_PERCENTILE = 90.0  # percentile of the pooled shuffled accuracies -> chance level

# style per plotted session set; extended if more sets are ever added
GROUP_STYLES = [
    {'label': 'set 1', 'pooled_color': 'k', 'session_color': '0.6', 'chance_color': 'g'},
    {'label': 'set 2', 'pooled_color': 'tab:red', 'session_color': 'lightcoral',
     'chance_color': 'tab:red'},
]


def _pool_and_classify(server, processed_server, sessions, align_timepoint, group_column,
                       before, after, bin_width, causal_sigma, avg_window, n_folds,
                       shuffle_percentile, only_good, min_rate, processes, seed):
    """Pool one set of sessions and classify it per session and as a pooled pseudo-population.

    Runs the full pipeline for a single session set: pool_trials -> per-session k-fold
    cross-validated LDA through time (each with its own shuffle-based chance level) ->
    condition-matched pooled pseudo-population classification.  Returns a group dict
    {'per_session_results', 'pooled_result', 'bin_centers', 'all_conditions'} (the last
    being the set of conditions seen, for the theoretical chance level), or None when the
    set has no usable neural data.
    """
    sessions_data, bin_centers = pool_trials(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        causal_sigma=causal_sigma, avg_window=avg_window, only_good=only_good,
        min_rate=min_rate)
    if not sessions_data:
        return None

    # per-session classification, each with its own shuffle-based chance level
    per_session_results = []
    for sd in sessions_data:
        accuracy, shuffle = classify_through_time(
            sd['X'], sd['labels'], n_folds=n_folds, processes=processes, seed=seed)
        chance = chance_level(shuffle, shuffle_percentile)
        per_session_results.append({
            'session': sd['session'], 'accuracy': accuracy, 'chance': chance,
            'n_neurons': sd['X'].shape[1]})
        rs('Session {}: classified {} time bins (peak accuracy {:.2f}, chance {:.2f}).'.format(
            sd['session'], len(accuracy),
            np.nanmax(accuracy) if accuracy.size else np.nan, chance))

    all_conditions = set().union(*[set(sd['labels'].tolist()) for sd in sessions_data])

    # pooled pseudo-population classification + its shuffle-based chance level
    pooled_result = None
    try:
        X_pooled, labels_pooled, conditions = build_pooled_pseudopopulation(
            sessions_data, seed=seed)
        accuracy, shuffle = classify_through_time(
            X_pooled, labels_pooled, n_folds=n_folds, processes=processes, seed=seed)
        chance = chance_level(shuffle, shuffle_percentile)
        pooled_result = {'accuracy': accuracy, 'chance': chance,
                         'conditions': conditions, 'n_neurons': X_pooled.shape[1]}
        rs('Pooled {} neurons across {} sessions, {} conditions, {} pseudo-trials '
           '(peak accuracy {:.2f}, chance {:.2f}).'.format(
               X_pooled.shape[1], len(sessions_data), len(conditions), len(labels_pooled),
               np.nanmax(accuracy) if accuracy.size else np.nan, chance))
    except ValueError as e:
        ws('No pooled pseudo-population: {}'.format(e))

    return {'per_session_results': per_session_results, 'pooled_result': pooled_result,
            'bin_centers': bin_centers, 'all_conditions': all_conditions}


def figure_classification_time(server, processed_server, sessions, sessions2=None,
                               align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                               before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                               causal_sigma=CAUSAL_SIGMA, avg_window=None,
                               n_folds=N_FOLDS, shuffle_percentile=SHUFFLE_PERCENTILE,
                               only_good=False, min_rate=MIN_RATE_HZ, processes=1,
                               sessions_label=None, sessions2_label=None, name=None,
                               save=True, save_dir=None, seed=0):
    """Classify the condition (`group_column`) through time for one or two session sets.

    Pools per-trial, causally-smoothed, square-rooted activity (pool_trials), then runs
    k-fold cross-validated LDA at every time bin -- once per session and once on a
    condition-matched pseudo-population pooled across each set's sessions.  A chance level
    is estimated for every dataset by shuffling the labels once per bin, pooling those
    accuracies over time and taking their `shuffle_percentile`th percentile.

    When `sessions2` is given (a non-empty list), it is pooled and classified separately
    and drawn on the same axes in a second colour, for comparison; the two sets never
    share neurons or trials.  Plots each set (individual sessions thin with their chance
    lines, the pooled pseudo-population thick with its chance line) plus a single
    theoretical 1 / n_conditions chance line.

    `sessions` is a list (empty -> all sessions); per-session recording/skip_ttl/
    good_neurons come from each session's meta_neural.json.  `sessions_label` /
    `sessions2_label` set the legend label for each set (e.g. the raw --sessions /
    --sessions2 token string); default to the GROUP_STYLES labels when not given.
    `processes` sets the size of the per-time-bin process pool.  The figure is saved by
    default (save=True) into <processed_server>/pooled_figures/figure_classification_time,
    named after `name` (the --sessions/--sessions2 strings; defaults to a stub built
    from `sessions`/`sessions2`); pass save=False to disable or save_dir to override the
    folder.  Returns the list of plotted group dicts.
    """
    def _run(sess):
        return _pool_and_classify(
            server, processed_server, sess, align_timepoint, group_column, before, after,
            bin_width, causal_sigma, avg_window, n_folds, shuffle_percentile, only_good,
            min_rate, processes, seed)

    group1 = _run(sessions)
    if group1 is None:
        raise ValueError('No sessions with usable neural data among {}.'.format(sessions))
    group1.update(GROUP_STYLES[0])
    if sessions_label:
        group1['label'] = sessions_label
    groups = [group1]

    if sessions2:
        group2 = _run(sessions2)
        if group2 is None:
            ws('No sessions with usable neural data among --sessions2 {}; plotting the '
               'first set only.'.format(sessions2))
        else:
            group2.update(GROUP_STYLES[1])
            if sessions2_label:
                group2['label'] = sessions2_label
            groups.append(group2)

    # the bin grid is identical across sets (same before/after/bin_width); theoretical
    # chance = 1 / n_conditions over the conditions seen across every plotted set.
    bin_centers = groups[0]['bin_centers']
    all_conditions = set().union(*[g['all_conditions'] for g in groups])
    theoretical_chance = 1.0 / len(all_conditions) if all_conditions else np.nan

    save_dir = resolve_pooled_save_dir(
        processed_server, 'figure_classification_time', save, save_dir)
    name = name or sessions_name_stub(sessions, sessions2)
    plot_classification_time(bin_centers, groups, theoretical_chance, shuffle_percentile,
                             align_timepoint, group_column, name, save_dir)
    return groups
