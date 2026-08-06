#!python3
# -*- coding: utf-8 -*-
"""
Cross-session pooling of neural activity for the pooled figure modules.

pool_neurons  --- per-neuron force-group PETH averages (mean +/- SEM), pooled.
pool_trials   --- per-trial causally-smoothed sqrt-rate activity tensors, pooled.

Both load each session's NWB + behavioural meta, apply the session's meta_neural
skip_ttl / recording, pair TTL pulses to trials positionally, keep the successful
trials with a valid alignment timepoint, and select neurons.  They construct a
NeuralConfig and use the prehension behavioural helpers, so they live in
neural_plotting rather than the reusable neural_processing.common layer; the pure
downstream aggregation lives in neural_processing.common.population.

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
import multiprocessing

import numpy as np
import scipy.ndimage
import tqdm

from ... import meta_session
from ...tools import filters, forces, stats
from ...tools.cmd_args import resolve_meta_arg
from ...tools.logs import rs, ws
from ...neural_processing import config
from ...neural_processing.common import probe
from ...neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA,
    read_nwb_spikes_and_ttl, get_trial_data_spike, resolve_neuron_selection)
from ...neural_processing.common.population import MIN_RATE_HZ
from .behaviour import load_timepoints_into_msession, get_timepoint, get_target_force

CAUSAL_SIGMA = 0.05        # s, SD of the causal half-gaussian rate filter (pool_trials default)

# pool_cross_correlations defaults
PRE_LAG = 1              # s, cross-correlation lag explored before zero
POST_LAG = 1             # s, cross-correlation lag explored after zero
FORCE_ACTIVE_FRACTION = 0.05   # active-grasp threshold, fraction of the trial's peak force
MIN_CORE_BINS = 3          # skip a trial whose active-force period spans fewer bins


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
            probe_type = probe.probe_type_from_meta(server, processed_server, session)
            cfg = config.NeuralConfig(server, processed_server, session, probe_type)
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

        skip_ttl = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl', 0)
        skip_ttl_last = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl_last', 0)

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


def pool_trials(server, processed_server, sessions, align_key=ALIGN_TIMEPOINT,
                group_column=GROUP_COLUMN, before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                causal_sigma=CAUSAL_SIGMA, avg_window=None, only_good=False,
                min_rate=MIN_RATE_HZ):
    """Build the per-session, per-trial activity tensors used for classification.

    Mirrors pool_neurons' session-handling (probe type, skip_ttl, positional
    pulse<->trial pairing, successful-trial selection) but, instead of per-condition
    averages, keeps the single-trial activity.  For every kept neuron and successful
    trial, the spikes in [tp - before, tp + after] are binned, turned into a firing
    rate, smoothed with a causal half-gaussian (SD `causal_sigma`) and square-rooted.
    When `avg_window` (s) is given, the activity is additionally averaged over a centred
    moving window of that width at each time bin.  Neurons are then kept if their mean
    firing rate over the window exceeds `min_rate` Hz.

    Returns (sessions_data, bin_centers) where each entry of sessions_data is
    {'session': str, 'X': (n_trials, n_neurons, n_time) sqrt-rate activity,
    'labels': (n_trials,) condition per trial, 'neuron_labels': [unit id, ...]}.
    Sessions lacking neural data / meta / a matching pulse count / active neurons /
    a valid alignment timepoint are skipped with a warning.
    """
    found = ([s for s in sessions if os.path.isdir(os.path.join(server, s))]
             if sessions else meta_session.find_session_dirs(server))

    # shared bin grid + filters (identical for every session so tensors align)
    bins = np.arange(-before - bin_width / 2, after + bin_width / 2, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    numbins = len(bin_centers)
    freq = 1.0 / bin_width
    duration = float(before + after)
    kernel = filters.causal_halfgaussian_kernel(causal_sigma, bin_width)
    win_bins = max(1, int(round(avg_window / bin_width))) if avg_window else None

    sessions_data = []
    for session in found:
        try:
            probe_type = probe.probe_type_from_meta(server, processed_server, session)
            cfg = config.NeuralConfig(server, processed_server, session, probe_type)
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

        skip_ttl = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl', 0)
        skip_ttl_last = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl_last', 0)

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

        # successful trials with a valid alignment timepoint
        trials = [(t, get_timepoint(t, align_key)) for t in used_msession]
        trials = [(t, tp) for t, tp in trials if tp is not None]
        if not trials:
            ws("Skipping session {}: no trials with a valid '{}' timepoint.".format(
                session, align_key))
            continue
        used = [t for t, _ in trials]
        tps = [tp for _, tp in trials]
        labels = np.array([get_target_force(mobject, t.object_id, group_column) for t in used])

        # per-trial, per-neuron sqrt(causal-smoothed rate); mean rate kept for selection
        n_trials, n_sel = len(used), len(neuron_selection)
        activity = np.zeros((n_trials, n_sel, numbins))
        mean_rate = np.zeros(n_sel)
        for i_t, (trial, tp) in enumerate(zip(used, tps)):
            for j, sel in enumerate(neuron_selection):
                s = np.asarray(trial.spikes[sel])
                s = s[(s > tp - before) & (s < tp + after)] - tp
                counts, _ = np.histogram(s, bins=bins)
                mean_rate[j] += len(s) / duration
                rate = filters.apply_causal_filter(counts * freq, kernel)
                activity[i_t, j, :] = np.sqrt(np.clip(rate, 0.0, None))
        mean_rate /= max(n_trials, 1)

        if win_bins:
            activity = scipy.ndimage.uniform_filter1d(
                activity, size=win_bins, axis=2, mode='nearest')

        keep = np.nonzero(mean_rate > min_rate)[0]
        rs('Session {}: activity filter kept {} / {} unit(s) with mean rate > {} Hz.'.format(
            session, len(keep), n_sel, min_rate))
        if keep.size == 0:
            ws('Skipping session {}: no unit exceeds the {} Hz activity threshold.'.format(
                session, min_rate))
            continue

        sessions_data.append({
            'session': session,
            'X': activity[:, keep, :],
            'labels': labels,
            'neuron_labels': [neuron_labels[j] for j in keep],
        })
        rs('Session {}: {} trials, {} neurons, conditions {}.'.format(
            session, n_trials, len(keep), sorted(set(labels.tolist()))))

    return sessions_data, bin_centers


# Neuron-independent per-trial cross-correlation context, shared with the worker
# processes through a pool initializer (set once per worker, not pickled per neuron).
_XCORR_CTX = None


def _xcorr_pool_init(ctx):
    """multiprocessing.Pool initializer: stash the shared per-trial context."""
    global _XCORR_CTX
    _XCORR_CTX = ctx


def _xcorr_neuron_worker(neuron_trial_spikes):
    """Cross-correlate one neuron's rate with the force over every trial's active span.

    `neuron_trial_spikes` is the neuron's spike-time array per usable trial (already
    windowed to the trial's extended grid).  Using the shared context, bins each
    trial's spikes into a firing rate, symmetrically smooths it, cross-correlates it
    with that trial's force over the active span, and averages the per-trial
    correlation curves (nanmean).  Returns (xcorr_row (n_lags,), spike_count) with
    spike_count the neuron's total spikes inside the active spans (for the mean-rate
    activity filter).  Top-level and picklable so it runs under spawn (Windows).
    """
    (trial_edges, trial_force_core, trial_t0, trial_ncore,
     n_pre, n_post, freq, sigma_bins, dt) = _XCORR_CTX
    n_lags = n_pre + n_post + 1
    xcorr_sum = np.zeros(n_lags)
    xcorr_count = np.zeros(n_lags)
    spike_count = 0
    for s, edges, force_core, t0, n_core in zip(
            neuron_trial_spikes, trial_edges, trial_force_core, trial_t0, trial_ncore):
        s = np.asarray(s)
        spike_count += int(np.sum((s >= t0) & (s < t0 + n_core * dt)))
        counts, _ = np.histogram(s, bins=edges)
        rate = scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)
        xc = stats.lagged_crosscorrelation(rate, force_core, n_pre, n_post)
        valid = np.isfinite(xc)
        xcorr_sum[valid] += xc[valid]
        xcorr_count[valid] += 1
    with np.errstate(invalid='ignore'):
        row = np.where(xcorr_count > 0, xcorr_sum / xcorr_count, np.nan)
    return row, spike_count


def pool_cross_correlations(server, processed_server, sessions, bin_width=BIN_WIDTH,
                            filter_sigma=FILTER_SIGMA, pre_lag=PRE_LAG, post_lag=POST_LAG,
                            force_fraction=FORCE_ACTIVE_FRACTION, only_good=False,
                            min_rate=MIN_RATE_HZ, processes=1):
    """Cross-correlate each neuron's rate with the summed grasp force, per session.

    Mirrors pool_neurons / pool_trials for the session handling (probe type, skip_ttl,
    positional pulse<->trial pairing, successful-trial selection, neuron selection),
    but instead of aligning to a timepoint it works within each trial's active-force
    period.  For every successful trial the summed pressure-sensor force is read (in
    seconds since the TTL pulse, the same frame as the spikes) with
    forces.load_summed_force_trace, and its continuous active-grasp span is found with
    forces.active_period_bounds (the stable-grasp threshold of
    matching.process_and_align_data).  Each neuron's spike train is binned and
    symmetrically smoothed into a firing rate; the rate is then cross-correlated with
    the force over that active span for integer-bin lags in [-pre_lag, +post_lag]
    (stats.lagged_crosscorrelation).  The neuron's rate window is widened by the lag
    range so every lag is evaluated over the full active-grasp span (the lags adjust
    the edges, not the correlation window).  Per neuron the per-trial correlation
    curves are averaged (nanmean).

    Within a session the per-neuron cross-correlations are independent, so they are
    farmed out to a multiprocessing.Pool of ``processes`` workers (one neuron per
    process; serial when processes <= 1) and reported with a tqdm progress bar as each
    neuron finishes.  The neuron-independent per-trial context (grid edges, force over
    the active span) is built once and shared with the workers through a pool
    initializer.

    Selection is all units, or -- with only_good -- the session's meta_neural
    'good_neurons'; neurons whose mean rate over the active periods is at or below
    ``min_rate`` Hz are dropped.  Sessions lacking neural data / meta / a matching
    pulse count / any usable trial / active neurons are skipped with a warning.

    Returns (sessions_results, lag_times) where each entry of sessions_results is
    {'session': str, 'region': str, 'burr_hole': str, 'xcorr': (n_neurons, n_lags)
    mean Pearson cross-correlation curves, 'neuron_labels': [unit id, ...],
    'n_trials': int}, and lag_times is the shared lag axis (s, negative = neuron leads
    the force, positive = neuron lags it).
    """
    found = ([s for s in sessions if os.path.isdir(os.path.join(server, s))]
             if sessions else meta_session.find_session_dirs(server))

    # shared lag grid (identical for every session / trial)
    dt = bin_width
    n_pre = int(round(pre_lag / dt))
    n_post = int(round(post_lag / dt))
    lag_times = np.arange(-n_pre, n_post + 1) * dt
    freq = 1.0 / dt
    sigma_bins = filter_sigma / dt

    sessions_results = []
    for session in found:
        try:
            probe_type = probe.probe_type_from_meta(server, processed_server, session)
            cfg = config.NeuralConfig(server, processed_server, session, probe_type)
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

        skip_ttl = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl', 0)
        skip_ttl_last = resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl_last', 0)

        try:
            _, _, _, msession = meta_session.load_meta_information(cfg.rserv, cfg.pserv)
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

        # First pass (serial, cheap): the neuron-independent per-trial context over
        # each usable trial's active-grasp span -- grid edges and the force sampled on
        # the core grid.  The grid is widened by the lag range so every lag is
        # evaluated over the full active span (bin edge 0 sits at the span start).
        n_sel = len(neuron_selection)
        kept_trials, trial_edges, trial_force_core = [], [], []
        trial_t0, trial_ncore = [], []
        core_duration = 0.0
        for trial in used_msession:
            if not trial.do_pre_ps_files_exist():
                continue
            try:
                ftimes, force = forces.load_summed_force_trace(
                    list(trial.get_pre_ps_filenames().values()))
            except Exception as e:  # noqa: BLE001
                ws('Session {} trial {}: could not read force ({}); skipping trial.'.format(
                    session, trial.trial_number, e))
                continue

            ap_start, ap_end = forces.active_period_bounds(force, fraction=force_fraction)
            n_core = int(round((ftimes[ap_end - 1] - ftimes[ap_start]) / dt)) if ap_end > ap_start \
                else 0
            if n_core < MIN_CORE_BINS:
                continue

            t0 = ftimes[ap_start]
            ext_start = t0 - n_pre * dt
            n_bins = n_pre + n_core + n_post
            edges = ext_start + np.arange(n_bins + 1) * dt
            centers = edges[:-1] + dt / 2
            core_centers = centers[n_pre:n_pre + n_core]

            kept_trials.append(trial)
            trial_edges.append(edges)
            trial_force_core.append(np.interp(core_centers, ftimes, force))
            trial_t0.append(t0)
            trial_ncore.append(n_core)
            core_duration += n_core * dt

        n_used_trials = len(kept_trials)
        if n_used_trials == 0:
            ws('Skipping session {}: no trial with a usable active-force period.'.format(session))
            continue

        # One task per neuron: its spikes across the kept trials, windowed to each
        # trial's extended grid (keeps the pickled payload small).
        tasks = []
        for sel in neuron_selection:
            per_trial = []
            for trial, edges in zip(kept_trials, trial_edges):
                s = np.asarray(trial.spikes[sel])
                per_trial.append(s[(s >= edges[0]) & (s <= edges[-1])])
            tasks.append(per_trial)

        # Split the neurons across a process pool (serial when processes <= 1); the
        # neuron-independent per-trial context is shared through the pool initializer.
        # tqdm reports progress as each neuron finishes (imap yields in task order).
        ctx = (trial_edges, trial_force_core, trial_t0, trial_ncore,
               n_pre, n_post, freq, sigma_bins, dt)
        desc = 'x-corr {} ({} neurons)'.format(session, n_sel)
        if processes and processes > 1:
            with multiprocessing.Pool(processes=processes, initializer=_xcorr_pool_init,
                                      initargs=(ctx,)) as pool:
                results = list(tqdm.tqdm(pool.imap(_xcorr_neuron_worker, tasks),
                                         total=n_sel, desc=desc, ncols=100))
        else:
            _xcorr_pool_init(ctx)
            results = [_xcorr_neuron_worker(task)
                       for task in tqdm.tqdm(tasks, total=n_sel, desc=desc, ncols=100)]

        xcorr = np.array([r[0] for r in results])
        spike_count = np.array([r[1] for r in results], dtype=float)
        mean_rate = spike_count / core_duration if core_duration > 0 else np.zeros(n_sel)
        keep = np.nonzero(mean_rate > min_rate)[0]
        rs('Session {}: activity filter kept {} / {} unit(s) with mean rate > {} Hz over '
           '{} trial(s).'.format(session, len(keep), n_sel, min_rate, n_used_trials))
        if keep.size == 0:
            ws('Skipping session {}: no unit exceeds the {} Hz activity threshold.'.format(
                session, min_rate))
            continue

        sessions_results.append({
            'session': session,
            'region': cfg.meta_neural.get('region', '') or '',
            'burr_hole': cfg.meta_neural.get('burr_hole', '') or '',
            'xcorr': xcorr[keep, :],
            'neuron_labels': [neuron_labels[j] for j in keep],
            'n_trials': n_used_trials,
        })

    return sessions_results, lag_times
