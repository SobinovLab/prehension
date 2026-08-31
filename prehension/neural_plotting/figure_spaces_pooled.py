#!python3
# -*- coding: utf-8 -*-
"""
Pooled neural state-space (PCA) figures across multiple sessions.

Pools the per-condition average activity of the selected neurons from a list of
sessions (common.pooling.pool_neurons), then:
  * lightly selects neurons (mean firing rate above min_rate Hz),
  * builds a (condition x time, neuron) matrix of per-condition average activity,
  * runs PCA (via SVD on z-scored neurons),
  * plots the first N PCs through time, 2D trajectories (PC1 vs 2, PC1 vs 3) and a
    3D trajectory (PC1 vs 2 vs 3), one trace per condition (coloured by force).

The reusable pieces now live elsewhere: the pooler in neural_plotting.common.pooling,
the pure aggregation (select_active_neurons, build_condition_matrix) in
neural_processing.common.population, the PCA in tools.stats.run_pca, and the drawers
in neural_plotting.common.traces.

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
from ..tools.logs import rs, ws
from ..tools.cmd_args import sessions_name_stub
from ..tools.stats import run_pca
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA)
from ..neural_processing.common.population import (
    select_active_neurons, build_condition_matrix, MIN_RATE_HZ)
from .common.pooling import pool_neurons
from .common.traces import (
    plot_pcs_through_time, plot_pc_2d, plot_pc_3d, resolve_pooled_save_dir)

N_PCS = 9           # number of principal components to compute / plot through time


def figure_spaces_pooled(server, processed_server, sessions,
                         align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                         before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                         filter_sigma=FILTER_SIGMA, only_good=False,
                         min_rate=MIN_RATE_HZ, n_pcs=N_PCS, drift_correct=True,
                         name=None, save=True, save_dir=None):
    """Pool neurons across sessions, PCA their per-condition averages, and plot the spaces.

    `sessions` is a list (empty -> all sessions on the server); per-session
    recording/skip_ttl/good_neurons come from each session's meta_neural.json.
    Figures are saved by default (save=True) into
    <processed_server>/pooled_figures/figure_spaces_pooled, named after `name` (the
    --sessions string; defaults to a stub built from `sessions`) with a per-figure
    suffix; pass save=False to disable or save_dir to override the folder.
    """
    save_dir = resolve_pooled_save_dir(processed_server, 'figure_spaces_pooled', save, save_dir)
    name = name or sessions_name_stub(sessions)
    entries, bin_centers, max_force = pool_neurons(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        filter_sigma=filter_sigma, only_good=only_good, drift_correct=drift_correct)
    if not entries:
        raise ValueError('No neurons pooled from the requested sessions {}.'.format(sessions))

    entries = select_active_neurons(entries, min_rate)
    if not entries:
        raise ValueError('No neurons remain after the > {} Hz activity filter.'.format(min_rate))

    X, conditions, numbins, labels = build_condition_matrix(entries)
    scores, evr = run_pca(X, n_pcs)
    k = scores.shape[1]
    scores3d = scores.reshape(len(conditions), numbins, k)
    rs('PCA on {} neurons x {} conditions x {} timebins -> {} PC(s); var: {}.'.format(
        len(labels), len(conditions), numbins, k,
        ', '.join('{:.1f}%'.format(100 * v) for v in evr)))

    plot_pcs_through_time(scores3d, bin_centers, conditions, max_force, evr,
                          align_timepoint, name, save_dir)
    plot_pc_2d(scores3d, conditions, max_force, evr, align_timepoint, name, save_dir)
    plot_pc_3d(scores3d, conditions, max_force, evr, align_timepoint, name, save_dir)
    return scores3d, evr, conditions, labels
