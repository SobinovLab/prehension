#!python3
# -*- coding: utf-8 -*-
"""
Pooled neural state-space (demixed PCA) figures across multiple sessions.

A dPCA companion to figure_spaces_pooled.  Pools per-condition average activity
(common.pooling.pool_neurons), lightly selects neurons, then:
  * builds the trial-averaged tensor R (n_neurons, n_conditions, n_time),
  * runs demixed PCA (separating the condition-independent time marginalization
    from the condition-dependent ones),
  * prints the variance carried by each marginalization,
  * outputs the matlab_dpca summary figure, and
  * projects the per-condition averages onto the leading condition-dependent
    demixed components and draws one trace per condition with +/- s.e.m. bands.

demixed PCA is provided by the local 'matlab_dpca' package, imported lazily inside
neural_processing.common.population.run_dpca / the traces drawers.  The tensor build
+ dPCA live in neural_processing.common.population; the drawers in
neural_plotting.common.traces.

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
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA)
from ..neural_processing.common.population import (
    select_active_neurons, MIN_RATE_HZ, build_condition_tensors, run_dpca)
from .common.pooling import pool_neurons
from .common.traces import (
    report_marginalization_variance, plot_dpca_summary, plot_cd_dpc_traces)

N_COMPONENTS = 10   # demixed components kept overall (top 3 condition-dependent are traced)
N_SHOW = 3          # condition-dependent components plotted through time


def figure_spaces_dpca_pooled(server, processed_server, sessions,
                              align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                              before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                              filter_sigma=FILTER_SIGMA, only_good=False,
                              min_rate=MIN_RATE_HZ, n_components=N_COMPONENTS, save_dir=None):
    """Pool neurons across sessions, run demixed PCA on their per-condition averages,
    print the marginalization variance, and plot the matlab_dpca summary figure and
    the leading condition-dependent components through time.

    `sessions` is a list (empty -> all sessions); per-session recording/skip_ttl/
    good_neurons come from each session's meta_neural.json.  save_dir defaults to
    processed_server.  Returns (W, V, which_marg, expl_var, conditions).
    """
    entries, bin_centers, max_force = pool_neurons(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        filter_sigma=filter_sigma, only_good=only_good)
    if not entries:
        raise ValueError('No neurons pooled from the requested sessions {}.'.format(sessions))

    entries = select_active_neurons(entries, min_rate)
    if not entries:
        raise ValueError('No neurons remain after the > {} Hz activity filter.'.format(min_rate))

    R, R_sem, conditions, numbins = build_condition_tensors(entries)
    W, V, which_marg, expl_var = run_dpca(R, n_components)
    rs('dPCA on {} neurons x {} conditions x {} timebins; kept {} components.'.format(
        R.shape[0], len(conditions), numbins, W.shape[1]))
    report_marginalization_variance(expl_var)

    save_dir = processed_server if save_dir is None else save_dir
    plot_dpca_summary(R, W, V, which_marg, expl_var, bin_centers, align_timepoint, save_dir)
    plot_cd_dpc_traces(R, R_sem, W, which_marg, expl_var, conditions, max_force,
                       bin_centers, align_timepoint, save_dir, n_show=N_SHOW)
    return W, V, which_marg, expl_var, conditions
