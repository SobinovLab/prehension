#!python3
# -*- coding: utf-8 -*-
"""
Cross-validated classification of an object condition through time, by depth group.

The depth counterpart of figure_classification_time.  Pools per-trial, causally-smoothed,
square-rooted activity across sessions together with each neuron's depth along the probe
(common.pooling.pool_trials with with_depths), then splits every session's neurons by
depth (common.population.split_by_depth): the 'top' group is the top fraction of the
session's neurons (farthest from the electrode tip, 0.25 by default) and the 'bottom'
group the bottom fraction (nearest the tip, 0.25 by default).  Each group is classified
exactly as in figure_classification_time (classify_sessions_data: per-session LDA through
time plus a condition-matched pooled pseudo-population, each with a shuffle-based chance
level), and both groups are overlaid on one figure (plot_classification_time).

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
from ..tools.logs import ws
from ..tools.cmd_args import sessions_name_stub
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH)
from ..neural_processing.common.population import MIN_RATE_HZ, split_by_depth
from ..neural_plotting.common.pooling import pool_trials, CAUSAL_SIGMA
from ..neural_plotting.common.traces import plot_classification_time, resolve_pooled_save_dir
from .figure_classification_time import (
    classify_sessions_data, report_sustained_above_chance, theoretical_chance,
    N_FOLDS, SHUFFLE_PERCENTILE, SUSTAINED_DURATION_S, GROUP_STYLES)

FIGURE_NAME = 'figure_classification_depth'
TOP_FRACTION = 0.25      # top (farthest from the tip) fraction of each session's neurons
BOTTOM_FRACTION = 0.25   # bottom (nearest the tip) fraction of each session's neurons
DEPTH_GROUPS = ('top', 'bottom')   # plotted in this order, styled by GROUP_STYLES


def _check_fractions(top_fraction, bottom_fraction):
    """Validate the depth-group fractions: each in (0, 1], and the groups do not overlap."""
    for name, fraction in (('top_fraction', top_fraction), ('bottom_fraction', bottom_fraction)):
        if not 0.0 < fraction <= 1.0:
            raise ValueError('{} must be in (0, 1], got {}.'.format(name, fraction))
    if top_fraction + bottom_fraction > 1.0:
        raise ValueError('top_fraction + bottom_fraction = {} > 1: the depth groups would '
                         'share neurons.'.format(top_fraction + bottom_fraction))


def figure_classification_depth(server, processed_server, sessions,
                                top_fraction=TOP_FRACTION, bottom_fraction=BOTTOM_FRACTION,
                                align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                                before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                                causal_sigma=CAUSAL_SIGMA, avg_window=None,
                                n_folds=N_FOLDS, shuffle_percentile=SHUFFLE_PERCENTILE,
                                only_good=False, min_rate=MIN_RATE_HZ, processes=1,
                                drift_correct=True, use_threshold_crossings=False, name=None,
                                sustained_duration=SUSTAINED_DURATION_S,
                                save=True, save_dir=None, seed=None):
    """Classify the condition (`group_column`) through time for the top and bottom depth groups.

    Pools per-trial activity with unit depths (pool_trials, after the `min_rate` activity
    filter), splits each session's neurons into its top `top_fraction` (farthest from the
    electrode tip) and bottom `bottom_fraction` (nearest the tip) by depth (split_by_depth),
    and classifies each group per session and as a pooled pseudo-population with k-fold
    cross-validated LDA at every time bin (classify_sessions_data, as in
    figure_classification_time).  Both groups are drawn on the same axes (top in the first
    GROUP_STYLES colour, bottom in the second), with a single theoretical 1 / n_conditions
    chance line; per group, the sessions above their own shuffle chance for longer than
    `sustained_duration` seconds are logged.

    `sessions` is a list (empty -> all sessions); per-session recording/skip_ttl/
    good_neurons come from each session's meta_neural.json.  Sessions whose NWB carries no
    unit depths are left out.  The figure is saved by default (save=True) into
    <processed_server>/pooled_figures/figure_classification_depth, named after `name` (the
    --sessions string; defaults to a stub built from `sessions`); pass save=False to disable
    or save_dir to override the folder.  Returns the list of plotted group dicts.
    """
    _check_fractions(top_fraction, bottom_fraction)
    sessions_data, bin_centers = pool_trials(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        causal_sigma=causal_sigma, avg_window=avg_window, only_good=only_good,
        min_rate=min_rate, use_threshold_crossings=use_threshold_crossings,
        drift_correct=drift_correct, with_depths=True)
    if not sessions_data:
        raise ValueError('No sessions with usable neural data among {}.'.format(sessions))

    depth_groups = split_by_depth(sessions_data, top_fraction, bottom_fraction)
    fractions = {'top': top_fraction, 'bottom': bottom_fraction}
    groups = []
    for key, style in zip(DEPTH_GROUPS, GROUP_STYLES):
        if not depth_groups[key]:
            ws('No neurons in the {} depth group; skipping it.'.format(key))
            continue
        group = classify_sessions_data(depth_groups[key], bin_centers, n_folds,
                                       shuffle_percentile, processes, seed)
        group.update(style)
        group['label'] = '{} {:.0f}% by depth'.format(key, 100 * fractions[key])
        groups.append(group)
    if not groups:
        raise ValueError('No session among {} has unit depths to split by.'.format(sessions))

    report_sustained_above_chance(groups, sustained_duration)

    save_dir = resolve_pooled_save_dir(processed_server, FIGURE_NAME, save, save_dir)
    name = name or sessions_name_stub(sessions)
    plot_classification_time(bin_centers, groups, theoretical_chance(groups),
                             shuffle_percentile, align_timepoint, group_column, name, save_dir)
    return groups
