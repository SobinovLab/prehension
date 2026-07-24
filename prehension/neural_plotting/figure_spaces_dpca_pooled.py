#!python3
# -*- coding: utf-8 -*-
"""
Pooled neural state-space (demixed PCA) figures across multiple sessions.

A dPCA copy of figure_spaces_pooled.  Reuses figure_peth_pooled.pool_neurons to
pool per-condition average activity and figure_spaces_pooled.select_active_neurons
to lightly select neurons, then:
  * builds the trial-averaged tensor R (n_neurons, <condition axes...>, n_time),
    the conditions being defined by averaging (per-condition mean activity),
  * runs demixed PCA, which separates a condition-independent component (the time
    marginalization) from the condition-dependent ones,
  * prints the variance carried by each marginalization and the per-component
    breakdown (in the style of plotDPcaTraces.m),
  * outputs the demixed PCA summary figure straight from the matlab_dpca package
    (dpca_plot: explained variance, per-marginalization variance and the leading
    components through time), and
  * projects the per-condition averages onto the leading condition-dependent
    demixed components and draws one trace per condition (coloured by force) with
    +/- s.e.m. bands, mirroring plotDPcaTraces.m.

demixed PCA is provided by the local 'matlab_dpca' package (a Python port of the
MATLAB dPCA package), installed with `pip install -e .` from its source tree.  It
is imported lazily and is NOT installed or run here.

Multiple conditions: currently there is one condition axis (the pool_neurons
group_column, e.g. targetForce).  The tensor construction and the matlab_dpca
`combined_params` are written so that future simultaneous conditions (several
parameters of object_id) only add condition axes to R and extra 1-based parameter
labels to `combined_params` before the trailing time parameter.

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
import matplotlib as mpl
import matplotlib.pyplot as plt

from ..tools.logs import rs, ws
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA)
from .figure_peth_pooled import pool_neurons
from .figure_spaces_pooled import select_active_neurons, MIN_RATE_HZ, _colors, _save

N_COMPONENTS = 10   # demixed components kept overall (top 3 condition-dependent are traced)
N_SHOW = 3          # condition-dependent components plotted through time

# One condition parameter (label 1) + time (label 2). Group the condition main
# effect with the condition/time interaction ("Object"), and keep the time main
# effect on its own ("Condition-independent"). Extend with more 1-based labels
# for future simultaneous conditions, e.g. [[[1], [1, 3]], [[2], [2, 3]], [[3]]].
COMBINED_PARAMS = [[[1], [1, 2]], [[2]]]
MARG_NAMES = ['Object', 'Condition-independent']
CD_MARG = 0         # condition-dependent marginalization (0-based, matches COMBINED_PARAMS)
CI_MARG = 1         # condition-independent (time) marginalization


# ---------------------------------------------------------------------------
# Tensor building + dPCA
# ---------------------------------------------------------------------------
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


def run_dpca(R, n_components=N_COMPONENTS, combined_params=COMBINED_PARAMS):
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


def report_marginalization_variance(expl_var, marg_names=MARG_NAMES):
    """Log total and per-component variance carried by each marginalization.

    Mirrors the console summary of plotDPcaTraces.m: the total variance in each
    marginalization (as a percentage of the total), then a component-by-component
    table of the per-marginalization variance.
    """
    total = expl_var.total_var
    group_pct = expl_var.total_marginalized_var / total * 100.0
    rs('Total variance in each marginalization: ' + ', '.join(
        '{} {:.2f}%'.format(name, pct) for name, pct in zip(marg_names, group_pct)))

    marg_var = np.atleast_2d(expl_var.marg_var)  # (n_marg, n_components)
    header = 'Component  ' + '  '.join('{:>22s} (%)'.format(n) for n in marg_names)
    lines = [header]
    for i_dpc in range(marg_var.shape[1]):
        cells = '  '.join('{:>24.2f}'.format(marg_var[m, i_dpc]) for m in range(marg_var.shape[0]))
        lines.append('{:>9d}  {}'.format(i_dpc + 1, cells))
    rs('Per-component variance by marginalization:\n' + '\n'.join(lines))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_dpca_summary(R, W, V, which_marg, expl_var, bin_centers, align_key,
                      save_dir, marg_names=MARG_NAMES, time_marg=CI_MARG):
    """The matlab_dpca summary figure (dpca_plot): explained variance, per-margin
    variance and the leading components through time, one trace per condition.

    This is the package's own main figure; the condition-independent (time)
    marginalization is placed on the top row.
    """
    from matlab_dpca import dpca_plot

    fig = dpca_plot(
        R, W, V, whichMarg=which_marg, explained_var=expl_var,
        time=bin_centers, time_events=[0.0], marginalization_names=marg_names,
        time_marginalization=time_marg)
    fig.suptitle('dPCA summary, aligned to {}'.format(align_key))
    # _save(fig, save_dir, 'spaces_dpca_summary_{}.png'.format(align_key))
    return fig


def plot_cd_dpc_traces(R, R_sem, W, which_marg, expl_var, conditions, max_force,
                       bin_centers, align_key, save_dir, cd_marg=CD_MARG, n_show=N_SHOW):
    """Leading condition-dependent demixed components through time (plotDPcaTraces.m style).

    Projects the per-condition average activity onto the top condition-dependent
    decoder axes (W) and draws one trace per condition (coloured by force) with
    +/- s.e.m. bands.  The band for a projected component is the decoder-weighted
    combination of the per-neuron s.e.m.: sqrt(sum_n (W[n, comp] * sem[n, c, t])^2),
    the pooled-pseudo-population analogue of the single-trial spread that
    plotDPcaTraces.m shows (pooled data has no simultaneous single trials).
    """
    components = np.nonzero(which_marg == cd_marg)[0][:n_show]
    if components.size == 0:
        ws('No condition-dependent components among the kept dPCs; skipping trace plot.')
        return None

    cmap, norm, cond_colors = _colors(conditions, max_force)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    Wc = W[:, components]                                   # (N, k)
    means = np.tensordot(Wc, R, axes=([0], [0]))            # (k, n_cond, n_time)
    sems = np.sqrt(np.tensordot(Wc ** 2, R_sem ** 2, axes=([0], [0])))

    k = len(components)
    fig, axs = plt.subplots(k, 1, figsize=(8, 3 * k), squeeze=False, sharex=True)
    for i_comp, comp in enumerate(components):
        ax = axs[i_comp][0]
        for i_c, color in enumerate(cond_colors):
            mu, se = means[i_comp, i_c, :], sems[i_comp, i_c, :]
            ax.fill_between(bin_centers, mu - se, mu + se, color=color, alpha=0.25, linewidth=0)
            ax.plot(bin_centers, mu, color=color, linewidth=1.5)
        ax.axvline(0.0, color='g', linewidth=0.8)
        ax.set_ylabel('dPC{} ({:.2f}%)'.format(comp + 1, expl_var.component_var[comp]))
        ax.tick_params(labelsize=6)
    axs[-1][0].set_xlabel('Time, s')
    fig.suptitle('Condition-dependent dPCs (per-condition average +/- s.e.m.), '
                 'aligned to {}'.format(align_key))
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01).set_label(GROUP_COLUMN)
    # _save(fig, save_dir, 'spaces_dpca_cd_traces_{}.png'.format(align_key))
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
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
