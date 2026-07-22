#!python3
# -*- coding: utf-8 -*-
"""
Plot the neural TTL sync pulses (rising and falling edges) through time and,
underneath, the behavioural trial windows (from the sent-sync-message log), to
check and set the pulse<->trial alignment.

The two are aligned by the first pulse and the first trial; a skip option shifts
which pulse (skip >= 0) or trial (skip < 0) counts as the first, so an offset in
the pulse/trial correspondence can be found by eye.

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
import matplotlib.pyplot as plt

from .. import meta_session
from ..tools import io
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig
from ..neural_processing import ttl_sync

# target number of index labels per subplot (kept readable on long sessions)
MAX_LABELS = 40

# max gap for a trial start to count as having a matching rising TTL pulse
TTL_MATCH_TOL_S = 0.010  # 10 ms


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def read_trial_sync_times(cfg):
    """Read per-trial start/end sync-message times (s) from the behavioural log.

    Uses the same sent-sync-message columns as create_meta:
    'log_sent_start_sync_messages(ms)' for trial start and
    'log_sent_end_sync_messages(ms)' for trial end.  Returns
    (trial_num, dup_index, start_s, end_s) in behavioural-log appearance (recording)
    order - NOT sorted by trial number - so the rows line up positionally with the
    TTL pulses. ``dup_index`` is the 0-based occurrence index of each row within its
    trial number (>0 for duplicate recordings).
    """
    mstruct = meta_session.import_meta_structure(
        os.path.join(cfg.pserv, 'meta_structure.json'),
        raw_dir=cfg.rserv, proc_dir=cfg.pserv)
    if len(mstruct['auto_log']) == 0:
        raise ValueError('Session {} has no auto (behavioural) log.'.format(cfg.session))

    # concatenate all auto logs, mirroring create_meta.import_logs
    col_names, values = io.import_csv(mstruct['auto_log'][0])
    data = np.array(values).transpose()
    for al in mstruct['auto_log'][1:]:
        _, v = io.import_csv(al)
        data = np.concatenate((data, np.array(v).transpose()), axis=0)

    trial_num = data[:, col_names.index('trial_num')].astype(int)
    start_s = data[:, col_names.index('log_sent_start_sync_messages(ms)')] / 1000.0
    end_s = data[:, col_names.index('log_sent_end_sync_messages(ms)')] / 1000.0

    # occurrence index per row within its trial number, in appearance (recording) order
    seen = {}
    dup_index = np.empty(len(trial_num), dtype=int)
    for i, t in enumerate(trial_num):
        t = int(t)
        dup_index[i] = seen.get(t, 0)
        seen[t] = dup_index[i] + 1

    # preserve recording order (do NOT sort by trial number) so rows align with TTL pulses
    return trial_num, dup_index, start_s, end_s


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _interval_step(starts, stops):
    """Build x/y arrays of a 0/1 step signal that is high during each interval."""
    edges = [(float(s), 1) for s in starts]
    edges += [(float(e), -1) for e in stops if np.isfinite(e)]
    edges.sort(key=lambda p: (p[0], -p[1]))
    if not edges:
        return np.array([0.0]), np.array([0])

    xs, ys, level = [edges[0][0]], [0], 0
    for t, d in edges:
        xs.append(t)
        ys.append(level)
        level = max(0, min(1, level + d))  # clamp for overlapping intervals
        xs.append(t)
        ys.append(level)
    xs.append(edges[-1][0])
    ys.append(level)
    return np.array(xs), np.array(ys)


def _plot_track(ax, starts, stops, indices, ylabel, up_label, down_label):
    """Draw a 0/1 track high during each [start, stop], with index labels."""
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    xs, ys = _interval_step(starts, stops)
    ax.plot(xs, ys, color='k', linewidth=1.0)
    ax.plot(starts, np.ones_like(starts), '|', color='tab:green', markersize=10,
            label=up_label)
    finite_stops = stops[np.isfinite(stops)]
    ax.plot(finite_stops, np.zeros_like(finite_stops), '|', color='tab:cyan',
            markersize=10, label=down_label)
    ax.axvline(0.0, color='tab:blue', linestyle='--', linewidth=0.8)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['low', 'high'])
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', fontsize=8)

    label_every = max(1, int(np.ceil(len(starts) / MAX_LABELS)))
    for k, (s, idx) in enumerate(zip(starts, indices)):
        if k % label_every == 0:
            ax.annotate(str(idx), xy=(s, 1.0), xytext=(s, 1.12), ha='center',
                        va='bottom', fontsize=7, color='tab:green', rotation=90)


def _missing_pulse_mask(trial_starts, rising, tol=TTL_MATCH_TOL_S):
    """Boolean mask over trial_starts: True where no rising TTL is within tol (s).

    Both inputs are in the aligned time frame (see plot_ttl_trial_alignment), so a
    True entry marks a trial start that has no corresponding TTL pulse.
    """
    trial_starts = np.asarray(trial_starts, dtype=float)
    rising = np.sort(np.asarray(rising, dtype=float))
    if trial_starts.size == 0:
        return np.zeros(0, dtype=bool)
    if rising.size == 0:
        return np.ones(trial_starts.size, dtype=bool)
    idx = np.searchsorted(rising, trial_starts)
    idx_hi = np.clip(idx, 0, len(rising) - 1)
    idx_lo = np.clip(idx - 1, 0, len(rising) - 1)
    nearest = np.minimum(np.abs(rising[idx_hi] - trial_starts),
                         np.abs(rising[idx_lo] - trial_starts))
    return nearest > tol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_ttl_trial_alignment(server, processed_server, session, probe_type, skip=0,
                             recording=None):
    """Plot TTL pulses over trial windows, aligned by the first pulse-trial.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        skip {int} --- Pulses to skip for alignment; if negative, that many trials
            are skipped instead.  The reference (t=0) becomes rising[skip] and
            trial_start[0] (skip >= 0) or rising[0] and trial_start[-skip] (skip < 0).
        recording {int|str} --- Open Ephys recording within experiment1 to read,
            1-based (Recording1, Recording2, ...); selects which recording's TTL
            events are read. None -> probe default.
    """
    cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type,
                                recording=recording)
    cfg.ensure_work_folder()

    # neural TTL edges (times only; no recording load needed)
    edges = ttl_sync.extract_ttl_edge_times(cfg, verbose=True)
    rising = (np.asarray(edges['rising_times_s'], dtype=float)
              if edges['rising_times_s'] is not None else np.array([]))
    falling = (np.asarray(edges['falling_times_s'], dtype=float)
               if edges['falling_times_s'] is not None else np.array([]))

    # behavioural trial windows (in recording order, aligned positionally with the pulses)
    trial_num, dup_index, start_s, end_s = read_trial_sync_times(cfg)
    rs('{} TTL pulses; {} behavioural trials; skip={}.'.format(
        len(rising), len(start_s), skip))
    if len(rising) == 0 or len(start_s) == 0:
        raise ValueError('Need at least one TTL pulse and one trial to align.')

    # alignment reference: pulse[skip]<->trial[0] (skip>=0) or pulse[0]<->trial[-skip]
    pulse_ref = skip if skip >= 0 else 0
    trial_ref = 0 if skip >= 0 else -skip
    if not 0 <= pulse_ref < len(rising):
        raise ValueError('Pulse reference index {} out of range [0, {}).'.format(
            pulse_ref, len(rising)))
    if not 0 <= trial_ref < len(start_s):
        raise ValueError('Trial reference index {} out of range [0, {}).'.format(
            trial_ref, len(start_s)))
    t0_pulse = rising[pulse_ref]
    t0_trial = start_s[trial_ref]
    if not np.isfinite(t0_trial) or t0_trial == 0:
        ws('Reference trial {} has no start-sync time; alignment may be off.'.format(
            trial_ref))

    rising_a = rising - t0_pulse
    falling_a = falling - t0_pulse

    # keep only trials with valid start and end sync times (0 means not sent)
    valid = (np.isfinite(start_s) & np.isfinite(end_s) & (start_s != 0) &
             (end_s != 0) & (end_s >= start_s))
    trial_idx = np.where(valid)[0]
    start_a = start_s[valid] - t0_trial
    end_a = end_s[valid] - t0_trial
    if len(trial_idx) < len(start_s):
        ws('Dropped {} trials without valid start/end sync times.'.format(
            len(start_s) - len(trial_idx)))

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(16, 6))
    _plot_track(ax1, rising_a, falling_a, np.arange(len(rising)), 'TTL pulses',
                'rising', 'falling')
    ax1.set_title('TTL pulses (n={}) vs trials (n={}); aligned by pulse {} <-> '
                  'trial {} (skip={})'.format(len(rising), len(start_s), pulse_ref,
                                              trial_ref, skip))
    # label trials by their composite id (trial number + '_1', ... for duplicate recordings)
    trial_labels = ['{}{}'.format(trial_num[i], '' if dup_index[i] == 0 else '_%d' % dup_index[i])
                    for i in trial_idx]
    _plot_track(ax2, start_a, end_a, trial_labels, 'trials', 'trial start', 'trial end')
    ax2.set_xlabel('time from alignment (s)')

    # Missing-pulse check: trial starts with no rising TTL within TTL_MATCH_TOL_S,
    # marked with a red vertical line in both subplots.
    missing = _missing_pulse_mask(start_a, rising_a)
    n_missing = int(np.count_nonzero(missing))
    if n_missing:
        missing_labels = [trial_labels[k] for k in np.flatnonzero(missing)]
        ws('{} trial start(s) with no TTL pulse within {:.0f} ms: {}'.format(
            n_missing, TTL_MATCH_TOL_S * 1000, missing_labels))
        for j, x in enumerate(start_a[missing]):
            lbl = 'missing pulse' if j == 0 else None
            ax1.axvline(x, color='red', linewidth=1.2, alpha=0.8, label=lbl)
            ax2.axvline(x, color='red', linewidth=1.2, alpha=0.8, label=lbl)
        ax1.legend(loc='upper right', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    else:
        rs('All {} trial start(s) have a TTL pulse within {:.0f} ms.'.format(
            len(start_a), TTL_MATCH_TOL_S * 1000))

    fig.tight_layout()
    out = os.path.join(cfg.work_folder, 'ttl_trial_alignment.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    rs('Saved {}'.format(out))
