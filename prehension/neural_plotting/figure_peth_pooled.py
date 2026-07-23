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
import scipy.ndimage

from .. import meta_session
from ..tools import plotting
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA,
    read_nwb_spikes_and_ttl, load_timepoints_into_msession, get_trial_data_spike,
    get_timepoint, get_target_force, resolve_neuron_selection)


# ---------------------------------------------------------------------------
# Stage 1: pool selected neurons across sessions into a joint structure
# ---------------------------------------------------------------------------
def pool_neurons(server, processed_server, sessions, align_key=ALIGN_TIMEPOINT,
                 group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                 bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA, only_good=False):
    """Pool selected neurons across sessions into one joint structure.

    For each session: load its neural NWB + behavioural meta, apply the session's
    meta_neural skip_ttl / recording, window+align spikes to align_key, and compute
    per-neuron force-group PETH averages (mean +/- SEM).  Selection is all units,
    or -- with only_good -- the session's meta_neural 'good_neurons'.  Sessions that
    lack neural data / meta / a matching pulse count are skipped with a warning.

    Returns (entries, bin_centers, max_force) where each entry is
    {'label': '<session>: <unit id>', 'frs_avg': (n_groups, numbins),
    'frs_sem': (n_groups, numbins), 'group_ids': [force, ...]}, and max_force is the
    largest group value across all sessions (for a shared colour scale).
    """
    found = ([s for s in sessions if os.path.isdir(os.path.join(server, s))]
             if sessions else meta_session.find_session_dirs(server))

    # shared bin grid (identical for every session)
    bins = np.arange(-before - bin_width / 2, after + bin_width / 2, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    freq = 1.0 / bin_width
    sigma_bins = filter_sigma / bin_width

    entries = []
    max_force = 0.0
    for session in found:
        try:
            probe_type = npconfig.probe_type_from_meta(server, processed_server, session)
            cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        if only_good:
            neuron_ids = cfg.meta_neural.get('good_neurons') or []
            if not neuron_ids:
                ws("Skipping session {}: only_good set but 'good_neurons' is empty.".format(
                    session))
                continue
        else:
            neuron_ids = None

        skip_ttl = npconfig.resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl', 0)
        skip_ttl_last = npconfig.resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl_last', 0)

        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                cfg.rserv, cfg.pserv)
            load_timepoints_into_msession(msession, mstruct)
            spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(cfg.nwb_path)
        except Exception as e:  # noqa: BLE001
            ws('Skipping session {}: {}'.format(session, e))
            continue

        # positional pulse<->trial offset: skip_ttl >0 drops leading pulses, <0 drops
        # leading trials; skip_ttl_last trims the end (pulses if >0, trials if <0).
        if skip_ttl and skip_ttl > 0:
            events_time = events_time[skip_ttl:]
        elif skip_ttl and skip_ttl < 0:
            msession = msession[-skip_ttl:]
        if skip_ttl_last and skip_ttl_last > 0:
            events_time = events_time[:-skip_ttl_last]
        elif skip_ttl_last and skip_ttl_last < 0:
            msession = msession[:skip_ttl_last]
        if len(events_time) != len(msession):
            ws('Skipping session {}: {} TTL pulses vs {} trials (after skip_ttl={}, '
               'skip_ttl_last={}); inspect with figure_ttl_alignment.'.format(
                   session, len(events_time), len(msession), skip_ttl, skip_ttl_last))
            continue

        session_spikes = get_trial_data_spike(spikes, events_time)
        num_neurons = len(unit_ids)
        for trial_spikes, trial_events in zip(session_spikes, events_time):
            ttl_start = trial_events[0]
            for i_n in range(num_neurons):
                trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ttl_start
        for i_trial, trial in zip(range(len(session_spikes)), msession):
            trial.spikes = session_spikes[i_trial]
        used_msession = [t for t in msession[:len(session_spikes)]
                         if t.success and hasattr(t, 'spikes')]

        try:
            neuron_selection, neuron_labels = resolve_neuron_selection(unit_ids, neuron_ids)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        # trials with a valid alignment timepoint
        trials = [(t, get_timepoint(t, align_key)) for t in used_msession]
        trials = [(t, tp) for t, tp in trials if tp is not None]
        if not trials:
            ws("Skipping session {}: no trials with a valid '{}' timepoint.".format(
                session, align_key))
            continue
        used = [t for t, _ in trials]
        tps = [tp for _, tp in trials]

        group = [get_target_force(mobject, t.object_id, group_column) for t in used]
        group_ids = sorted(set(group))
        if group:
            max_force = max(max_force, max(group))

        for sel_idx, label_uid in zip(neuron_selection, neuron_labels):
            frs = np.zeros((len(used), len(bin_centers)))
            for i_t, (trial, tp) in enumerate(zip(used, tps)):
                s = np.asarray(trial.spikes[sel_idx])
                s = s[(s > tp - before) & (s < tp + after)] - tp
                counts, _ = np.histogram(s, bins=bins)
                frs[i_t, :] = scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)
            frs_avg = np.zeros((len(group_ids), len(bin_centers)))
            frs_sem = np.zeros((len(group_ids), len(bin_centers)))
            for i_g, gid in enumerate(group_ids):
                ingroup = [i for i, g in enumerate(group) if g == gid]
                frs_avg[i_g, :] = np.mean(frs[ingroup, :], axis=0)
                frs_sem[i_g, :] = (np.std(frs[ingroup, :], axis=0)
                                   / np.sqrt(len(ingroup)))
            entries.append({'label': '{}: {}'.format(session, label_uid),
                            'frs_avg': frs_avg, 'frs_sem': frs_sem,
                            'group_ids': group_ids})
        rs('Pooled {} unit(s) from session {}.'.format(len(neuron_selection), session))

    return entries, bin_centers, max_force


# ---------------------------------------------------------------------------
# Stage 2: plot the pooled structure
# ---------------------------------------------------------------------------
def plot_pooled(entries, bin_centers, max_force, group_column, align_key, save_dir):
    """Plot pooled per-neuron force-group PETH averages, one subplot per neuron.

    Each subplot is titled '<session>: <unit id>'; force groups are coloured on a
    single shared scale (0..max_force).  Saves peth_pooled_force_averages_<align>.png
    into save_dir and returns the figure.
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
        out = os.path.join(save_dir, 'peth_pooled_force_averages_{}.png'.format(align_key))
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
                                     only_good=False, min_rate=None, save_dir=None):
    """Pool the selected neurons across `sessions` and plot them on one figure.

    Arguments mirror figure_peth.plot_perievent_histograms except there is no
    neuron_ids/recording/skip_ttl override: `sessions` is a list (empty -> all
    sessions on the server), and per-session recording/skip_ttl/good_neurons come
    from each session's meta_neural.json.  save_dir defaults to processed_server.
    """
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
                       processed_server if save_dir is None else save_dir)
