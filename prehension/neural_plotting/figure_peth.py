#!python3
# -*- coding: utf-8 -*-
"""
Peri-event time histogram (PETH) figure.

Reads the neural product (spikes + TTL sync) from the NWB written by
neural_processing.export_nwb, and behavioural trial timepoints / object forces
from the prehension meta.  Aligns each trial's spikes to a timepoint (default
first_grasp_start) and colour-codes traces by an object property (default
targetForce(N)).  Probe-agnostic: the NWB is uniform for both probe types.

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
from ..tools import io
from ..tools import plotting
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig

# defaults; overridable through the calling function / script
ALIGN_TIMEPOINT = 'first_grasp_start'
GROUP_COLUMN = 'targetForce(N)'
BEFORE = 1.0            # s before the alignment timepoint
AFTER = 1.0            # s after
BIN_WIDTH = 0.02       # s
FILTER_SIGMA = 0.05    # s, Gaussian smoothing of firing-rate traces


# ---------------------------------------------------------------------------
# Neural data (from the NWB)
# ---------------------------------------------------------------------------
def read_nwb_spikes_and_ttl(nwb_path):
    """Read per-unit spike times, unit ids and per-pulse TTL windows from the NWB.

    Returns (spike_per_unit {list of arrays, s}, unit_ids {list}, events_time
    {list of [start, stop] arrays, s}).
    """
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_path), 'r') as fio:
        nwbfile = fio.read()

        udf = nwbfile.units.to_dataframe()
        spike_per_unit = [np.asarray(s, dtype=float) for s in udf['spike_times']]
        if 'unit_id' in udf.columns:
            unit_ids = [u for u in udf['unit_id']]
        else:
            unit_ids = list(udf.index)

        ttl = nwbfile.intervals['ttl_pulses'].to_dataframe()
        events_time = [np.array([float(a), float(b)])
                       for a, b in zip(ttl['start_time'], ttl['stop_time'])]
    rs('NWB: {} units, {} TTL pulses.'.format(len(unit_ids), len(events_time)))
    return spike_per_unit, unit_ids, events_time


def get_trial_data_spike(spike_per_unit, events_time):
    """For each trial window [start, stop], extract per-unit spikes within it."""
    trial_spike = []
    for e in events_time:
        temp = []
        for spike in spike_per_unit:
            idx = np.where((spike > e[0]) & (spike < e[1]))
            temp.append(spike[idx])
        trial_spike.append(temp)
    return trial_spike


# ---------------------------------------------------------------------------
# Behavioural data (prehension meta)
# ---------------------------------------------------------------------------
def load_timepoints_into_msession(msession, mstruct):
    tp_dic = io.import_csv_as_dic(mstruct['timepoint_csv_filename'])
    trial_number_col = tp_dic['trial_number']
    del tp_dic['trial_number']
    # optional occurrence index to disambiguate duplicate recordings of the same trial_number
    dup_col = tp_dic.pop('trial_dup_index', None)
    for trial in msession:
        if dup_col is not None:
            trial_row = None
            for i, tn in enumerate(trial_number_col):
                if (int(float(tn)) == trial.trial_number and
                        int(float(dup_col[i])) == trial.dup_index):
                    trial_row = i
                    break
            if trial_row is None:
                raise ValueError('Trial {} (occurrence {}) not found in timepoints.'.format(
                    trial.trial_number, trial.dup_index))
        else:
            trial_row = trial_number_col.index(trial.trial_number)
        trial.timepoints = {k: v[trial_row] for k, v in tp_dic.items()}


def get_timepoint(trial, key):
    v = trial.timepoints.get(key, None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def get_target_force(mobject, object_id, group_column):
    key = object_id
    if key not in mobject and str(key) in mobject:
        key = str(key)
    return float(mobject[key]['def'][group_column])


def resolve_neuron_selection(unit_ids, neuron_ids):
    """Map requested unit ids to indices into unit_ids.  None/[] -> all."""
    if not neuron_ids:
        return list(range(len(unit_ids))), list(unit_ids)
    id_to_index = {}
    for i, uid in enumerate(unit_ids):
        id_to_index[uid] = i
        id_to_index[str(uid)] = i
    selection, labels, missing = [], [], []
    for uid in neuron_ids:
        idx = id_to_index.get(uid, id_to_index.get(str(uid)))
        if idx is None:
            missing.append(uid)
        else:
            selection.append(idx)
            labels.append(unit_ids[idx])
    if missing:
        ws('Requested unit ids not found and skipped: {}'.format(missing))
    if not selection:
        raise ValueError('None of neuron_ids={} matched {}.'.format(neuron_ids, unit_ids))
    return selection, labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_peth(msession, mobject, neuron_selection, neuron_labels, align_key,
               group_column, before, after, bin_width, filter_sigma, save_dir,
               individual_traces=False):
    """PETH traces aligned to align_key, colour-coded by group_column."""
    trials = [(t, get_timepoint(t, align_key)) for t in msession]
    trials = [(t, tp) for t, tp in trials if tp is not None]
    if not trials:
        raise ValueError("No trials have a valid '{}' timepoint.".format(align_key))
    used = [t for t, _ in trials]
    tps = [tp for _, tp in trials]
    rs('Using {} / {} trials with valid {!r}.'.format(len(used), len(msession), align_key))

    spikes = []
    for neuron_id in neuron_selection:
        per_trial = []
        for trial, tp in zip(used, tps):
            s = np.asarray(trial.spikes[neuron_id])
            per_trial.append(s[(s > tp - before) & (s < tp + after)] - tp)
        spikes.append(per_trial)

    bins = np.arange(-before - bin_width / 2, after + bin_width / 2, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    numbins = len(bin_centers)
    freq = 1.0 / bin_width
    sigma_bins = filter_sigma / bin_width

    frs = np.zeros((len(neuron_selection), len(used), numbins))
    for i_n, neuron_spikes in enumerate(spikes):
        for i_t, nt_spikes in enumerate(neuron_spikes):
            counts, _ = np.histogram(nt_spikes, bins=bins)
            frs[i_n, i_t, :] = scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)

    group = [get_target_force(mobject, t.object_id, group_column) for t in used]
    group_ids = sorted(set(group))
    cmap = plt.get_cmap('autumn_r')
    norm = mpl.colors.Normalize(vmin=0, vmax=max(group) if max(group) > 0 else 1)
    trial_colors = [cmap(norm(g)) for g in group]
    group_colors = [cmap(norm(g)) for g in group_ids]

    frs_avg = np.zeros((len(neuron_selection), len(group_ids), numbins))
    frs_sem = np.zeros((len(neuron_selection), len(group_ids), numbins))
    for i_n in range(len(neuron_selection)):
        for i_g, gid in enumerate(group_ids):
            ingroup = [i for i, g in enumerate(group) if g == gid]
            frs_avg[i_n, i_g, :] = np.mean(frs[i_n, ingroup, :], axis=0)
            frs_sem[i_n, i_g, :] = (np.std(frs[i_n, ingroup, :], axis=0)
                                    / np.sqrt(len(ingroup)))

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    xn, yn = plotting.xy_numsubplots(len(neuron_selection))

    if individual_traces:
        fig1, axs1 = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9))
        axs1 = np.atleast_1d(axs1).flatten()
        for i_n, ax in enumerate(axs1):
            if i_n >= len(neuron_selection):
                ax.axis('off')
                continue
            for i_t, color in enumerate(trial_colors):
                ax.plot(bin_centers, frs[i_n, i_t, :], color=color, linewidth=0.7, alpha=0.7)
            ax.axvline(0.0, color='k', linewidth=0.8, linestyle='--')
            ax.set_title('unit {}'.format(neuron_labels[i_n]), fontsize=8)
            ax.tick_params(labelsize=6)
        fig1.suptitle('PETH per trial, aligned to {}, coloured by {}'.format(
            align_key, group_column))
        fig1.tight_layout(rect=(0, 0, 0.94, 0.96))
        fig1.colorbar(sm, ax=axs1.tolist(), fraction=0.02, pad=0.01).set_label(group_column)

    fig2, axs2 = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9))
    axs2 = np.atleast_1d(axs2).flatten()
    for i_n, ax in enumerate(axs2):
        if i_n >= len(neuron_selection):
            ax.axis('off')
            continue
        for i_g, color in enumerate(group_colors):
            m, e = frs_avg[i_n, i_g, :], frs_sem[i_n, i_g, :]
            ax.fill_between(bin_centers, m - e, m + e, color=color, alpha=0.3)
            ax.plot(bin_centers, m, color=color, linewidth=1.8)
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle='--')
        ax.set_title('unit {}'.format(neuron_labels[i_n]), fontsize=8)
        ax.tick_params(labelsize=6)
    fig2.suptitle('PETH force-group averages (mean +/- SEM), aligned to {}'.format(align_key))
    fig2.tight_layout(rect=(0, 0, 0.94, 0.96))
    fig2.colorbar(sm, ax=axs2.tolist(), fraction=0.02, pad=0.01).set_label(group_column)

    os.makedirs(save_dir, exist_ok=True)
    if individual_traces:
        f1 = os.path.join(save_dir, 'peth_traces_{}.png'.format(align_key))
        fig1.savefig(f1, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(f1))
    f2 = os.path.join(save_dir, 'peth_force_averages_{}.png'.format(align_key))
    fig2.savefig(f2, dpi=150, bbox_inches='tight')
    rs('Saved {}'.format(f2))


def plot_perievent_histograms(server, processed_server, session, probe_type,
                              neuron_ids=None, align_timepoint=ALIGN_TIMEPOINT,
                              group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                              bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA):
    """Plot PETH traces for one session from its NWB and prehension meta.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        neuron_ids {list} --- Unit ids to plot; None/empty -> all units.
        align_timepoint {str} --- Trial timepoint to align to.
        group_column {str} --- Object property to colour-code by.
        before, after {float} --- Window (s) around the alignment timepoint.
        bin_width, filter_sigma {float} --- Binning and smoothing (s).
    """
    cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type)

    # behavioural meta
    mstruct, _, mobject, msession = meta_session.load_meta_information(
        cfg.rserv, cfg.pserv)
    load_timepoints_into_msession(msession, mstruct)

    # neural + TTL windows from the NWB.
    # The pulses carry no trial IDs: pulse i is paired to trial i strictly by position, so
    # msession MUST be in recording (chronological) order - which create_meta now guarantees,
    # including for duplicate-recording trials. With duplicates preserved, the counts should match.
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(cfg.nwb_path)
    if len(events_time) != len(msession):
        ws('WARNING: {} TTL pulses but {} behavioural trials. Since pulses are matched to trials '
           'positionally, a count mismatch means every trial after the discrepancy is likely '
           'misaligned. This is no longer explained by dropped duplicate trials (those are now '
           'preserved), so it indicates a genuinely missed/extra pulse or an ordering problem - '
           'inspect with figure_ttl_alignment before trusting the alignment. Aligning by index up '
           'to the shorter.'.format(len(events_time), len(msession)))

    session_spikes = get_trial_data_spike(spikes, events_time)
    num_neurons = len(unit_ids)
    rs('Total {} trials with {} units.'.format(len(msession), num_neurons))
    for trial_spikes, trial_events in zip(session_spikes, events_time):
        ttl_start = trial_events[0]
        for i_n in range(num_neurons):
            trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ttl_start

    for i_trial, trial in zip(range(len(session_spikes)), msession):
        trial.spikes = session_spikes[i_trial]
    msession = [t for t in msession[:len(session_spikes)]
                if t.success and hasattr(t, 'spikes')]

    neuron_selection, neuron_labels = resolve_neuron_selection(unit_ids, neuron_ids)
    rs('Plotting {} unit(s): {}'.format(len(neuron_selection), neuron_labels))

    _plot_peth(msession, mobject, neuron_selection, neuron_labels, align_timepoint,
               group_column, before, after, bin_width, filter_sigma, cfg.work_folder)
