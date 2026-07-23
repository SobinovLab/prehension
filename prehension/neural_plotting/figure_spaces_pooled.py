#!python3
# -*- coding: utf-8 -*-
"""
Pooled neural state-space (PCA) figures across multiple sessions.

Reuses figure_peth_pooled.pool_neurons to pool the per-condition average activity
of the selected neurons from a list of sessions, then:
  * lightly selects neurons (mean firing rate above min_rate Hz),
  * builds a (condition x time, neuron) matrix of per-condition average activity,
  * runs PCA (via SVD on z-scored neurons),
  * plots the first N PCs through time, 2D trajectories (PC1 vs 2, PC1 vs 3) and a
    3D trajectory (PC1 vs 2 vs 3), one trace per condition (coloured by force).

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

from ..tools import plotting
from ..tools.logs import rs, ws
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA)
from .figure_peth_pooled import pool_neurons

MIN_RATE_HZ = 1.0   # light activity selection: keep neurons with mean rate above this
N_PCS = 9           # number of principal components to compute / plot through time


# ---------------------------------------------------------------------------
# Selection + matrix building + PCA
# ---------------------------------------------------------------------------
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


def run_pca(X, n_components=N_PCS):
    """Z-score each neuron (column) and run PCA via SVD.

    Returns (scores, explained_variance_ratio) with scores shape (n_samples, k),
    k = min(n_components, rank).
    """
    Xc = X.astype(float)
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xz = (Xc - mean) / std

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    k = int(min(n_components, S.size))
    scores = U[:, :k] * S[:k]
    total = np.sum(S ** 2)
    evr = (S[:k] ** 2) / total if total > 0 else np.zeros(k)
    return scores, evr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _colors(conditions, max_force):
    cmap = plt.get_cmap('autumn_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=max_force if max_force > 0 else 1)
    return cmap, norm, [cmap(norm(c)) for c in conditions]


def _save(fig, save_dir, name):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, name)
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))


def plot_pcs_through_time(scores3d, bin_centers, conditions, max_force, evr, align_key, save_dir):
    """First min(k,10) PCs vs time, one trace per condition, one subplot per PC."""
    n_cond, numbins, k = scores3d.shape
    n = min(k, 10)
    cmap, norm, cond_colors = _colors(conditions, max_force)
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
    # _save(fig, save_dir, 'spaces_pooled_pcs_time_{}.png'.format(align_key))
    return fig


def plot_pc_2d(scores3d, conditions, max_force, evr, align_key, save_dir):
    """2D trajectories PC1 vs PC2 and (if available) PC1 vs PC3; o=start, s=end."""
    k = scores3d.shape[2]
    if k < 2:
        ws('Fewer than 2 PCs available; skipping 2D plots.')
        return None
    pairs = [(0, 1)] + ([(0, 2)] if k >= 3 else [])
    cmap, norm, cond_colors = _colors(conditions, max_force)
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
    # _save(fig, save_dir, 'spaces_pooled_pc2d_{}.png'.format(align_key))
    return fig


def plot_pc_3d(scores3d, conditions, max_force, evr, align_key, save_dir):
    """3D trajectory PC1 vs PC2 vs PC3; o=start, s=end."""
    if scores3d.shape[2] < 3:
        ws('Fewer than 3 PCs available; skipping 3D plot.')
        return None
    cmap, norm, cond_colors = _colors(conditions, max_force)
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
    # _save(fig, save_dir, 'spaces_pooled_pc3d_{}.png'.format(align_key))
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def figure_spaces_pooled(server, processed_server, sessions,
                         align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                         before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                         filter_sigma=FILTER_SIGMA, only_good=False,
                         min_rate=MIN_RATE_HZ, n_pcs=N_PCS, save_dir=None):
    """Pool neurons across sessions, PCA their per-condition averages, and plot the spaces.

    `sessions` is a list (empty -> all sessions on the server); per-session
    recording/skip_ttl/good_neurons come from each session's meta_neural.json.
    save_dir defaults to processed_server.
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

    X, conditions, numbins, labels = build_condition_matrix(entries)
    scores, evr = run_pca(X, n_pcs)
    k = scores.shape[1]
    scores3d = scores.reshape(len(conditions), numbins, k)
    rs('PCA on {} neurons x {} conditions x {} timebins -> {} PC(s); var: {}.'.format(
        len(labels), len(conditions), numbins, k,
        ', '.join('{:.1f}%'.format(100 * v) for v in evr)))

    save_dir = processed_server if save_dir is None else save_dir
    plot_pcs_through_time(scores3d, bin_centers, conditions, max_force, evr,
                          align_timepoint, save_dir)
    plot_pc_2d(scores3d, conditions, max_force, evr, align_timepoint, save_dir)
    plot_pc_3d(scores3d, conditions, max_force, evr, align_timepoint, save_dir)
    return scores3d, evr, conditions, labels
