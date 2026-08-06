#!python3
# -*- coding: utf-8 -*-
"""
Pooled peri-event time histogram (PETH) figure across multiple sessions.

A multi-session copy of figure_peth: instead of one session's units, it pools the
selected neurons from a list of sessions onto a single figure, one subplot per
pooled neuron titled "<session>: <unit id>".  Neuron selection per session is all
units, or -- with only_good -- the unit ids in that session's meta_neural.json
'good_neurons'.  Per-session recording / skip_ttl are read from each session's
meta_neural.json (there is no --units / --recording / --skip_ttl override).

Two stages, as separate functions:
  * pool_neurons(...)  -> pools the selected neurons into a joint structure,
  * plot_pooled(...)   -> plots that structure.
plot_perievent_histograms_pooled(...) runs both.

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

from ..tools import plotting
from ..tools.logs import rs, ws
from ..tools.cmd_args import sessions_name_stub
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA)
from .common.pooling import pool_neurons
from .common.traces import resolve_pooled_save_dir, figure_filename


# ---------------------------------------------------------------------------
# Plot the pooled structure
# ---------------------------------------------------------------------------
def plot_pooled(entries, bin_centers, max_force, group_column, align_key, name, save_dir):
    """Plot pooled per-neuron force-group PETH averages, one subplot per neuron.

    Each subplot is titled '<session>: <unit id>'; force groups are coloured on a
    single shared scale (0..max_force).  Saves <name>.png into save_dir and returns the
    figure.
    """
    cmap = plt.get_cmap('autumn_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=max_force if max_force > 0 else 1)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    xn, yn = plotting.xy_numsubplots(len(entries))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9))
    axs = np.atleast_1d(axs).flatten()
    for i_n, ax in enumerate(axs):
        if i_n >= len(entries):
            ax.axis('off')
            continue
        e = entries[i_n]
        for i_g, gid in enumerate(e['group_ids']):
            color = cmap(norm(gid))
            m, s = e['frs_avg'][i_g, :], e['frs_sem'][i_g, :]
            ax.fill_between(bin_centers, m - s, m + s, color=color, alpha=0.3)
            ax.plot(bin_centers, m, color=color, linewidth=1.8)
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title(e['label'], fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle('Pooled PETH force-group averages (mean +/- SEM), aligned to {}'.format(
        align_key))
    fig.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig.colorbar(sm, ax=axs.tolist(), fraction=0.02, pad=0.01).set_label(group_column)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, figure_filename(name))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def plot_perievent_histograms_pooled(server, processed_server, sessions,
                                     align_timepoint=ALIGN_TIMEPOINT,
                                     group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                                     bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA,
                                     only_good=False, min_rate=None, name=None,
                                     save=True, save_dir=None):
    """Pool the selected neurons across `sessions` and plot them on one figure.

    Arguments mirror figure_peth.plot_perievent_histograms except there is no
    neuron_ids/recording/skip_ttl override: `sessions` is a list (empty -> all
    sessions on the server), and per-session recording/skip_ttl/good_neurons come
    from each session's meta_neural.json.  The figure is saved by default (save=True)
    into <processed_server>/pooled_figures/figure_peth_pooled, named after `name` (the
    --sessions string; defaults to a stub built from `sessions`); pass save=False to
    disable or save_dir to override the folder.
    """
    save_dir = resolve_pooled_save_dir(processed_server, 'figure_peth_pooled', save, save_dir)
    name = name or sessions_name_stub(sessions)
    entries, bin_centers, max_force = pool_neurons(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        filter_sigma=filter_sigma, only_good=only_good)
    if not entries:
        raise ValueError('No neurons pooled from the requested sessions {}.'.format(sessions))
    if min_rate is not None:
        n0 = len(entries)
        entries = [e for e in entries if float(np.mean(e['frs_avg'])) > min_rate]
        rs('Activity filter: kept {} / {} pooled neuron(s) with mean rate > {} Hz.'.format(
            len(entries), n0, min_rate))
        if not entries:
            raise ValueError(
                'No pooled neurons exceed the {} Hz activity threshold.'.format(min_rate))
    rs('Plotting {} pooled neuron(s).'.format(len(entries)))
    return plot_pooled(entries, bin_centers, max_force, group_column, align_timepoint,
                       name, save_dir)
