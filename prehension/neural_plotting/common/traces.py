#!python3
# -*- coding: utf-8 -*-
"""
Reusable neural figure drawers shared across the pooled figure modules.

Pure plotting helpers that take arrays / dPCA outputs and draw them (PC traces
through time, 2D/3D population trajectories, the matlab_dpca summary, condition-
dependent dPC traces, and classification accuracy through time).  They are
neural-figure-shaped (force/condition/dPCA structures) so they live in
neural_plotting rather than the generic tools layer; the generic pieces they use
(colour scale, subplot layout) come from tools.plotting.

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
import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from ...tools import plotting
from ...tools.logs import rs, ws
from ...neural_processing.common.spikes import GROUP_COLUMN

# demixed-PCA marginalization labels/indices (match figure_spaces_dpca_pooled)
MARG_NAMES = ['Object', 'Condition-independent']
CD_MARG = 0         # condition-dependent marginalization (0-based)
CI_MARG = 1         # condition-independent (time) marginalization
N_SHOW = 3          # condition-dependent components plotted through time


POOLED_FIGURES_DIRNAME = 'pooled_figures'
PREHENSION_PLOTS_DIRNAME = 'prehension_plots'


def resolve_pooled_save_dir(processed_server, figure_name, save, save_dir=None):
    """Resolve where a pooled figure's PNGs are written.

    An explicit ``save_dir`` always wins.  Otherwise, when ``save`` is True the PNGs
    go into ``<processed_server>/pooled_figures/<figure_name>`` (a per-figure
    subfolder); when ``save`` is False saving is disabled (returns None, so the
    drawers skip writing files).
    """
    if save_dir is not None:
        return save_dir
    if save:
        return os.path.join(processed_server, POOLED_FIGURES_DIRNAME, figure_name)
    return None


def resolve_session_save_dir(processed_server, session, save, save_dir=None):
    """Resolve where a single-session figure's PNGs are written.

    An explicit ``save_dir`` always wins.  Otherwise, when ``save`` is True the PNGs
    go into ``<processed_server>/<session>/prehension_plots``; when ``save`` is False
    saving is disabled (returns None).
    """
    if save_dir is not None:
        return save_dir
    if save:
        return os.path.join(processed_server, session, PREHENSION_PLOTS_DIRNAME)
    return None


def figure_filename(base, suffix=None):
    """PNG filename for a figure: ``<base>.png`` or ``<base>_<suffix>.png``.

    ``suffix`` is a short tag distinguishing the individual figures a script
    produces (e.g. 'pc2d', 'traces'); omit it for a script's sole figure.
    """
    return '{}_{}.png'.format(base, suffix) if suffix else '{}.png'.format(base)


def _save(fig, save_dir, name):
    """Save fig as a single PNG into save_dir (created if needed); no-op if None."""
    if save_dir is None:
        return
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, name)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    rs('Saved {}'.format(out))


# ---------------------------------------------------------------------------
# PCA state-space traces
# ---------------------------------------------------------------------------
def plot_pcs_through_time(scores3d, bin_centers, conditions, max_force, evr, align_key, name,
                          save_dir):
    """First min(k,10) PCs vs time, one trace per condition, one subplot per PC."""
    n_cond, numbins, k = scores3d.shape
    n = min(k, 10)
    cmap, norm, cond_colors = plotting.cmap_norm(conditions, max_force)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    xn, yn = plotting.xy_numsubplots(n)
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9))
    axs = np.atleast_1d(axs).flatten()
    for i_pc, ax in enumerate(axs):
        if i_pc >= n:
            ax.axis('off')
            continue
        for i_c, color in enumerate(cond_colors):
            ax.plot(bin_centers, scores3d[i_c, :, i_pc], color=color, linewidth=1.5)
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title('PC{} ({:.1f}%)'.format(i_pc + 1, 100 * evr[i_pc]), fontsize=9)
        ax.tick_params(labelsize=6)
    fig.suptitle('Population PCs through time, aligned to {}'.format(align_key))
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig.colorbar(sm, ax=axs.tolist(), fraction=0.02, pad=0.01).set_label(GROUP_COLUMN)
    _save(fig, save_dir, figure_filename(name, 'pcs_time'))
    return fig


def plot_pc_2d(scores3d, conditions, max_force, evr, align_key, name, save_dir):
    """2D trajectories PC1 vs PC2 and (if available) PC1 vs PC3; o=start, s=end."""
    k = scores3d.shape[2]
    if k < 2:
        ws('Fewer than 2 PCs available; skipping 2D plots.')
        return None
    pairs = [(0, 1)] + ([(0, 2)] if k >= 3 else [])
    cmap, norm, cond_colors = plotting.cmap_norm(conditions, max_force)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axs = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 6))
    axs = np.atleast_1d(axs).flatten()
    for ax, (px, py) in zip(axs, pairs):
        for i_c, color in enumerate(cond_colors):
            x, y = scores3d[i_c, :, px], scores3d[i_c, :, py]
            ax.plot(x, y, color=color, linewidth=1.5)
            ax.plot(x[0], y[0], 'o', color=color, markersize=6)
            ax.plot(x[-1], y[-1], 's', color=color, markersize=6)
        ax.set_xlabel('PC{} ({:.1f}%)'.format(px + 1, 100 * evr[px]))
        ax.set_ylabel('PC{} ({:.1f}%)'.format(py + 1, 100 * evr[py]))
        ax.set_title('PC{} vs PC{}'.format(px + 1, py + 1))
    fig.suptitle('Population trajectories (2D), aligned to {} (o=start, s=end)'.format(align_key))
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    fig.colorbar(sm, ax=axs.tolist(), fraction=0.02, pad=0.01).set_label(GROUP_COLUMN)
    _save(fig, save_dir, figure_filename(name, 'pc2d'))
    return fig


def plot_pc_3d(scores3d, conditions, max_force, evr, align_key, name, save_dir):
    """3D trajectory PC1 vs PC2 vs PC3; o=start, s=end."""
    if scores3d.shape[2] < 3:
        ws('Fewer than 3 PCs available; skipping 3D plot.')
        return None
    cmap, norm, cond_colors = plotting.cmap_norm(conditions, max_force)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='3d')
    for i_c, color in enumerate(cond_colors):
        x, y, z = scores3d[i_c, :, 0], scores3d[i_c, :, 1], scores3d[i_c, :, 2]
        ax.plot(x, y, z, color=color, linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color=color, marker='o', s=40)
        ax.scatter(x[-1], y[-1], z[-1], color=color, marker='s', s=40)
    ax.set_xlabel('PC1 ({:.1f}%)'.format(100 * evr[0]))
    ax.set_ylabel('PC2 ({:.1f}%)'.format(100 * evr[1]))
    ax.set_zlabel('PC3 ({:.1f}%)'.format(100 * evr[2]))
    ax.set_title('Population trajectory (3D), aligned to {} (o=start, s=end)'.format(align_key))
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.1).set_label(GROUP_COLUMN)
    _save(fig, save_dir, figure_filename(name, 'pc3d'))
    return fig


# ---------------------------------------------------------------------------
# demixed-PCA reporting + traces
# ---------------------------------------------------------------------------
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


def plot_dpca_summary(R, W, V, which_marg, expl_var, bin_centers, align_key, name,
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
    _save(fig, save_dir, figure_filename(name, 'summary'))
    return fig


def plot_cd_dpc_traces(R, R_sem, W, which_marg, expl_var, conditions, max_force,
                       bin_centers, align_key, name, save_dir, cd_marg=CD_MARG, n_show=N_SHOW):
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

    cmap, norm, cond_colors = plotting.cmap_norm(conditions, max_force)
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
    _save(fig, save_dir, figure_filename(name, 'cd_traces'))
    return fig


# ---------------------------------------------------------------------------
# Classification-through-time
# ---------------------------------------------------------------------------
def plot_classification_time(bin_centers, groups, theoretical_chance, shuffle_percentile,
                             align_key, group_column, name, save_dir):
    """Plot classification accuracy through time for one or more session sets.

    Each group in `groups` (see figure_classification_time / GROUP_STYLES) is drawn in
    its own colour: individual sessions as thin accuracy lines each with its own dashed
    chance line (the `shuffle_percentile`th percentile of that session's pooled shuffled
    accuracies), and the pooled pseudo-population as a thick accuracy line with its own
    dashed chance line.  A single green solid line marks the theoretical chance level
    (1 / n_conditions).  When more than one group is plotted, legend entries are prefixed
    with the group label.  Saves <name>.png into save_dir and returns the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    def _finite(v):
        return v is not None and np.isfinite(v)

    multi = len(groups) > 1
    for g in groups:
        prefix = '{}: '.format(g['label']) if multi else ''
        s_color, c_color, p_color = g['session_color'], g['chance_color'], g['pooled_color']

        for r in g['per_session_results']:
            ax.plot(bin_centers, r['accuracy'], color=s_color, linewidth=0.8, alpha=0.7)
            if _finite(r.get('chance')):
                ax.axhline(r['chance'], color=c_color, linewidth=0.6, linestyle='--', alpha=0.4)
        if g['per_session_results']:
            ax.plot([], [], color=s_color, linewidth=0.8, alpha=0.7,
                    label='{}individual sessions'.format(prefix))
            ax.plot([], [], color=c_color, linewidth=0.6, linestyle='--', alpha=0.4,
                    label='{}session chance (shuffle {:.0f}th pct)'.format(
                        prefix, shuffle_percentile))

        pooled_result = g['pooled_result']
        if pooled_result is not None:
            accuracy = pooled_result['accuracy']
            chance = pooled_result.get('chance')
            if _finite(chance):
                ax.axhline(chance, color=c_color, linewidth=1.3, linestyle='--',
                           label='{}pooled chance (shuffle {:.0f}th pct)'.format(
                               prefix, shuffle_percentile))
            ax.plot(bin_centers, accuracy, color=p_color, linewidth=2.5,
                    label='{}pooled ({} neurons)'.format(prefix, pooled_result['n_neurons']))

    if _finite(theoretical_chance):
        ax.axhline(theoretical_chance, color='g', linewidth=1.3, linestyle='-',
                   label='theoretical chance ({:.2f})'.format(theoretical_chance))

    ax.axvline(0.0, color='k', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('Classification accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Cross-validated LDA classification of {} through time, aligned to {}'.format(
        group_column, align_key))
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()

    _save(fig, save_dir, figure_filename(name))
    return fig
