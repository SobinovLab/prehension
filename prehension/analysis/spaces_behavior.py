#!python3
# -*- coding: utf-8 -*-
"""
Demixed PCA (and plain PCA) of a behavioural signal across sessions.

Pools a behavioural signal's channels into per-condition averages (pool_behavior) and
runs the same decomposition the neural spaces figures use -- demixed PCA
(build_condition_tensors -> run_dpca) or PCA (build_condition_matrix -> run_pca) -- then
draws the neural_plotting spaces figures on the result.  The dPCA / PCA machinery is
feature-agnostic, so the "neurons" are simply the behavioural channels (joint angles,
torques, digit or segment forces).

Figures are written under
<processed_server>/pooled_figures/spaces_behavior_<signal>_<method>/.

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
from ..tools.logs import rs
from ..tools.cmd_args import sessions_name_stub
from ..tools.stats import run_pca
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH)
from ..neural_processing.common.population import (
    build_condition_matrix, build_condition_tensors, run_dpca)
from ..neural_plotting.common.traces import (
    plot_pcs_through_time, plot_pc_2d, plot_pc_3d, report_marginalization_variance,
    plot_dpca_summary, plot_cd_dpc_traces, resolve_pooled_save_dir)
from .behavior_pooling import pool_behavior
from .figure_spaces_pooled import N_PCS
from .figure_spaces_dpca_pooled import N_COMPONENTS, N_SHOW


def spaces_behavior(server, processed_server, sessions, signal, method='dpca',
                    group_column=GROUP_COLUMN, align_timepoint=ALIGN_TIMEPOINT,
                    before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                    n_components=N_COMPONENTS, name=None, save=True, save_dir=None):
    """Pool a behavioural signal across sessions and run demixed PCA (or PCA) on it.

    `signal` is one of behavior_pooling.SIGNAL_SPECS ('joint_angles', 'torques',
    'digit_forces', 'segment_forces').  `method` is 'dpca' (default; demixed PCA via the
    local matlab_dpca) or 'pca'.  Conditions are the `group_column` object property (e.g.
    target force), aligned to `align_timepoint`.  For 'dpca' `n_components` is the number
    of demixed components kept (the leading condition-dependent ones are traced); for
    'pca' it is the number of PCs.  Figures are saved by default (save=True) into
    <processed_server>/pooled_figures/spaces_behavior_<signal>_<method>, named after
    `name` (defaults to the session stub + signal); pass save=False or save_dir to
    override.  Returns the decomposition (dPCA: (W, V, which_marg, expl_var, conditions);
    PCA: (scores3d, evr, conditions, labels)).
    """
    figure_name = 'spaces_behavior_{}_{}'.format(signal, method)
    save_dir = resolve_pooled_save_dir(processed_server, figure_name, save, save_dir)
    name = name or '{}_{}'.format(sessions_name_stub(sessions), signal)

    entries, bin_centers, max_force = pool_behavior(
        server, processed_server, sessions, signal, group_column=group_column,
        align_key=align_timepoint, before=before, after=after, bin_width=bin_width)
    if not entries:
        raise ValueError('No {} channels pooled from sessions {}.'.format(signal, sessions))

    if method == 'pca':
        X, conditions, numbins, labels = build_condition_matrix(entries)
        n = n_components if n_components else N_PCS
        scores, evr = run_pca(X, n)
        k = scores.shape[1]
        scores3d = scores.reshape(len(conditions), numbins, k)
        rs('PCA on {} {} channels x {} conditions x {} bins -> {} PC(s); var: {}.'.format(
            len(labels), signal, len(conditions), numbins, k,
            ', '.join('{:.1f}%'.format(100 * v) for v in evr)))
        plot_pcs_through_time(scores3d, bin_centers, conditions, max_force, evr,
                              align_timepoint, name, save_dir)
        plot_pc_2d(scores3d, conditions, max_force, evr, align_timepoint, name, save_dir)
        plot_pc_3d(scores3d, conditions, max_force, evr, align_timepoint, name, save_dir)
        return scores3d, evr, conditions, labels

    R, R_sem, conditions, numbins = build_condition_tensors(entries)
    W, V, which_marg, expl_var = run_dpca(R, n_components)
    rs('dPCA on {} {} channels x {} conditions x {} bins; kept {} components.'.format(
        R.shape[0], signal, len(conditions), numbins, W.shape[1]))
    report_marginalization_variance(expl_var)
    plot_dpca_summary(R, W, V, which_marg, expl_var, bin_centers, align_timepoint, name, save_dir)
    plot_cd_dpc_traces(R, R_sem, W, which_marg, expl_var, conditions, max_force,
                       bin_centers, align_timepoint, name, save_dir, n_show=N_SHOW)
    return W, V, which_marg, expl_var, conditions
