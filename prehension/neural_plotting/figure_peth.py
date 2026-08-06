#!python3
# -*- coding: utf-8 -*-
"""
Peri-event time histogram (PETH) figure.

Reads the neural product (spikes + TTL sync) from the NWB written by
neural_processing.export_nwb, and behavioural trial timepoints / object forces
from the prehension meta.  Aligns each trial's spikes to a timepoint (default
first_grasp_start) and colour-codes traces by an object property (default
targetForce(N)).  The alignment timepoint may be a column of the timepoints CSV
or a meta_session 'ttl_to_*' offset (e.g. ttl_to_success_grasp) -- both are in
seconds since the trial's TTL pulse.  Probe-agnostic: the NWB is uniform for both
probe types.

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
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA,
    read_nwb_spikes_and_ttl, get_trial_data_spike, resolve_neuron_selection)
from .common.behaviour import (
    load_timepoints_into_msession, get_timepoint, get_target_force)
from .common.traces import resolve_session_save_dir, figure_filename


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _filter_by_rate(msession, neuron_selection, neuron_labels, align_key, before, after,
                    min_rate):
    """Keep neurons whose mean firing rate over the aligned window exceeds min_rate Hz."""
    trials = [(t, get_timepoint(t, align_key)) for t in msession]
    trials = [(t, tp) for t, tp in trials if tp is not None]
    dur = float(before + after)
    keep_sel, keep_lab = [], []
    for sel, lab in zip(neuron_selection, neuron_labels):
        rates = []
        for t, tp in trials:
            s = np.asarray(t.spikes[sel])
            rates.append(np.sum((s > tp - before) & (s < tp + after)) / dur)
        if rates and float(np.mean(rates)) > min_rate:
            keep_sel.append(sel)
            keep_lab.append(lab)
    rs('Activity filter: kept {} / {} unit(s) with mean rate > {} Hz.'.format(
        len(keep_sel), len(neuron_selection), min_rate))
    if not keep_sel:
        raise ValueError('No units exceed the {} Hz activity threshold.'.format(min_rate))
    return keep_sel, keep_lab


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

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        if individual_traces:
            f1 = os.path.join(save_dir, figure_filename('figure_peth', 'traces'))
            fig1.savefig(f1, dpi=150, bbox_inches='tight')
            rs('Saved {}'.format(f1))
        f2 = os.path.join(save_dir, figure_filename('figure_peth'))
        fig2.savefig(f2, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(f2))


def plot_perievent_histograms(server, processed_server, session, probe_type,
                              neuron_ids=None, align_timepoint=ALIGN_TIMEPOINT,
                              group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                              bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA,
                              skip_ttl=None, skip_ttl_last=None, recording=None,
                              only_good=False, min_rate=None, save=True, save_dir=None):
    """Plot PETH traces for one session from its NWB and prehension meta.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        neuron_ids {list} --- Unit ids to plot; None/empty -> all units.
        only_good {bool} --- When True and neuron_ids is empty, plot the unit ids
            listed in meta_neural.json 'good_neurons'.
        min_rate {float} --- Optional average-activity threshold (Hz): drop units
            whose mean firing rate over the aligned window is at or below this.
            None (default) -> meta_neural.json 'min_rate' if present, else no filter.
        align_timepoint {str} --- Trial timepoint to align to. Either a timepoints
            CSV column (e.g. first_grasp_start) or a meta_session 'ttl_to_*' column
            (e.g. ttl_to_success_grasp, ttl_to_reach, ttl_to_force_target_start).
        group_column {str} --- Object property to colour-code by.
        before, after {float} --- Window (s) around the alignment timepoint.
        bin_width, filter_sigma {float} --- Binning and smoothing (s).
        skip_ttl {int} --- Alignment offset for the positional pulse<->trial
            pairing. Positive: drop this many leading TTL pulses (spurious
            sync/setup pulses before the first trial) so pulse skip_ttl pairs to
            trial 0. Negative: drop this many leading behavioural trials so pulse 0
            pairs to trial |skip_ttl|. None -> meta_neural.json 'skip_ttl' then 0.
        skip_ttl_last {int} --- Like skip_ttl but trimming the END: positive drops
            this many trailing TTL pulses, negative drops this many trailing
            behavioural trials. None -> meta_neural.json 'skip_ttl_last' then 0.
        recording {int|str} --- Open Ephys recording within experiment1, 1-based
            (Recording1, Recording2, ...), passed to NeuralConfig. The NWB is
            per-session, so this does not change what is read; kept for a uniform
            interface with the processing pipeline. None -> probe default.
        save {bool} --- Save the figure(s) as PNG (default True). The averages figure
            is <processed_server>/<session>/prehension_plots/figure_peth.png, and the
            optional individual-traces figure adds a '_traces' suffix.
        save_dir {str} --- Explicit output folder overriding the default
            <session>/prehension_plots location. None -> the default (see save).
    """
    cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type,
                                recording=recording)
    # skip_ttl / skip_ttl_last: CLI kwarg > meta_neural.json > 0
    skip_ttl = resolve_meta_arg(skip_ttl, cfg.meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(
        skip_ttl_last, cfg.meta_neural, 'skip_ttl_last', 0)
    min_rate = resolve_meta_arg(min_rate, cfg.meta_neural, 'min_rate', None)

    # --only_good: restrict to the good_neurons listed in meta_neural.json
    if only_good and not neuron_ids:
        neuron_ids = cfg.meta_neural.get('good_neurons') or []
        if not neuron_ids:
            raise ValueError(
                "only_good=True but meta_neural.json 'good_neurons' is empty; fill it "
                "in {} first.".format(npconfig.meta_neural_path(processed_server, session)))
        rs('only_good: plotting {} unit(s) from meta_neural.json good_neurons.'.format(
            len(neuron_ids)))

    # behavioural meta
    mstruct, _, mobject, msession = meta_session.load_meta_information(
        cfg.rserv, cfg.pserv)
    load_timepoints_into_msession(msession, mstruct)

    # neural + TTL windows from the NWB.
    # The pulses carry no trial IDs: pulse i is paired to trial i strictly by position, so
    # msession MUST be in recording (chronological) order - which create_meta now guarantees,
    # including for duplicate-recording trials. With duplicates preserved, the counts should match.
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(cfg.nwb_path)
    if skip_ttl > 0:
        # drop leading TTL pulses: pulse skip_ttl pairs to trial 0
        if skip_ttl >= len(events_time):
            raise ValueError('skip_ttl={} but only {} TTL pulse(s) available.'.format(
                skip_ttl, len(events_time)))
        rs('Skipping the first {} TTL pulse(s): {} -> {}.'.format(
            skip_ttl, len(events_time), len(events_time) - skip_ttl))
        events_time = events_time[skip_ttl:]
    elif skip_ttl < 0:
        # drop leading behavioural trials: pulse 0 pairs to trial |skip_ttl|
        n_drop = -skip_ttl
        if n_drop >= len(msession):
            raise ValueError(
                'skip_ttl={} but only {} behavioural trial(s) available.'.format(
                    skip_ttl, len(msession)))
        rs('Skipping the first {} behavioural trial(s): {} -> {}.'.format(
            n_drop, len(msession), len(msession) - n_drop))
        msession = msession[n_drop:]
    if skip_ttl_last > 0:
        # drop trailing TTL pulses
        if skip_ttl_last >= len(events_time):
            raise ValueError('skip_ttl_last={} but only {} TTL pulse(s) available.'.format(
                skip_ttl_last, len(events_time)))
        rs('Skipping the last {} TTL pulse(s): {} -> {}.'.format(
            skip_ttl_last, len(events_time), len(events_time) - skip_ttl_last))
        events_time = events_time[:-skip_ttl_last]
    elif skip_ttl_last < 0:
        # drop trailing behavioural trials
        n_drop = -skip_ttl_last
        if n_drop >= len(msession):
            raise ValueError(
                'skip_ttl_last={} but only {} behavioural trial(s) available.'.format(
                    skip_ttl_last, len(msession)))
        rs('Skipping the last {} behavioural trial(s): {} -> {}.'.format(
            n_drop, len(msession), len(msession) - n_drop))
        msession = msession[:skip_ttl_last]
    if len(events_time) != len(msession):
        raise ValueError(
            '{} TTL pulses but {} behavioural trials. Pulses are matched to trials strictly by '
            'position, so a count mismatch misaligns every trial after the discrepancy and '
            'flattens the PETH. Duplicate-recording trials are now preserved, so this indicates a '
            'genuinely missed/extra pulse or an ordering problem. Inspect with figure_ttl_alignment '
            'and fix the pulse<->trial correspondence before plotting.'.format(
                len(events_time), len(msession)))

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
    if min_rate is not None:
        neuron_selection, neuron_labels = _filter_by_rate(
            msession, neuron_selection, neuron_labels, align_timepoint, before, after, min_rate)
    rs('Plotting {} unit(s): {}'.format(len(neuron_selection), neuron_labels))

    save_dir = resolve_session_save_dir(processed_server, session, save, save_dir)
    _plot_peth(msession, mobject, neuron_selection, neuron_labels, align_timepoint,
               group_column, before, after, bin_width, filter_sigma, save_dir)
