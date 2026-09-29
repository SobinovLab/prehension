#!python3
# -*- coding: utf-8 -*-
"""
Plot the neural TTL sync pulses (rising and falling edges) through time and,
underneath, the behavioural trial windows (from the sent-sync-message log), to
check and set the pulse<->trial alignment.

The two are aligned by the first pulse and the first trial; a skip option shifts
which pulse (skip >= 0) or trial (skip < 0) counts as the first, so an offset in
the pulse/trial correspondence can be found by eye.

The configured intermediate skips are flagged over both tracks so the pairing the
analysis / plotting scripts will apply is visible: the meta_neural
'skip_ttl_intermediate' pulses in purple and the meta_structure 'skip_trials'
trials in magenta.  When an intermediate pulse is skipped, the kept pulses are drawn
as separate re-anchored traces in the TTL subplot -- each post-skip segment is shifted
so its next kept pulse lands on the next unclaimed, unskipped trial -- so the corrected
correspondence lines up with the trial track below.  The meta_neural 'reanchor_ttl' list
starts the same re-anchored trace at given pulse indices WITHOUT dropping any pulse (to
realign across a drift or a recording gap); both this and the skip re-anchoring honour a
non-zero leading skip_ttl.  The alignment check then runs against the re-anchored pulses:
a trial start still left without a nearby drawn pulse is marked misaligned in red, while
skipped trials are excluded (shown as skipped, not misaligned).

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
from ..tools import misc
from ..tools import plotting
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig
from ..neural_processing.common import events
from .common.traces import resolve_session_save_dir, figure_filename

# max gap for a trial start to count as having a matching rising TTL pulse
# Larger value does not matter because each trial is aligned separately
# this needs to be small enough to not confuse multiple trials, which are >1 s long
TTL_MATCH_TOL_S = 0.100  # 100 ms


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


# distinct step-line colours per re-anchored pulse segment; chosen to avoid the green/cyan edge
# ticks, the blue alignment line and the purple/magenta skip flags used elsewhere in the figure
_SEGMENT_COLORS = ('black', 'tab:orange', 'tab:brown', 'tab:olive', 'tab:gray')


def _pair_pulse_segments(n_pulses, ignore, pulse_ref, trial_ref, trial_nums,
                         skip_ttl_intermediate, reanchor_ttl, skip_trials):
    """Group kept TTL pulses into re-anchor segments for the pulse<->trial pairing.

    Walks the ``n_pulses`` pulses (0-based; original index = ignore + j) in order, pairing each
    from the alignment reference onward (pulse ``pulse_ref`` <-> trial ``trial_ref``; earlier
    pulses are leading extras that claim no trial) to the next unclaimed, unskipped trial
    (``trial_nums`` are all trials in recording order, minus those in ``skip_trials``).  A segment
    boundary is started by either a skipped intermediate pulse (original index in
    ``skip_ttl_intermediate``, which is dropped and claims no trial) or a re-anchor pulse (original
    index in ``reanchor_ttl``, which is kept but begins a fresh segment).  Returns a list of
    (kept_js, anchor_ti, dropped): the segment's kept pulse indices, the (full) trial index its
    first kept pulse pairs with (None if none / trials run out; used to shift the segment onto its
    paired trial), and the skipped intermediate pulses that belong to this segment -- those dropped
    just before it -- so they are drawn as its leading pulses in the same frame, sitting before that
    trial (like the leading skip_ttl pulses).  Honouring pulse_ref / trial_ref makes the segments
    correct for a non-zero leading skip.
    """
    segments = []
    seg_js, seg_anchor_ti, seg_dropped = [], None, []
    pending_dropped = []   # skipped pulses buffered until the NEXT segment starts
    ti, n_trials = trial_ref, len(trial_nums)
    for j in range(n_pulses):
        oi = ignore + j
        dropped = oi in skip_ttl_intermediate
        # a dropped intermediate pulse or a re-anchor pulse both close the current segment first
        if (dropped or oi in reanchor_ttl) and seg_js:
            segments.append([seg_js, seg_anchor_ti, seg_dropped])
            seg_js, seg_anchor_ti, seg_dropped = [], None, []
        if dropped:
            pending_dropped.append(j)   # belongs to the segment that starts after it
            continue
        if j < pulse_ref:
            paired_ti = None   # leading extra pulse before the reference: claims no trial
        else:
            while ti < n_trials and trial_nums[ti] in skip_trials:
                ti += 1
            paired_ti = ti if ti < n_trials else None
        if not seg_js:   # first kept pulse of a new segment
            seg_dropped = pending_dropped   # the buffered drops precede this segment's trial
            pending_dropped = []
            seg_anchor_ti = paired_ti
        seg_js.append(j)
        if paired_ti is not None:
            ti += 1
    if seg_js:
        segments.append([seg_js, seg_anchor_ti, seg_dropped])
    # skipped pulses after the last kept pulse have no following segment: keep them with the last
    if pending_dropped:
        if segments:
            segments[-1][2] = segments[-1][2] + pending_dropped
        else:
            segments.append([[], None, pending_dropped])   # degenerate: every pulse skipped
    return segments


def _plot_pulse_segments(ax, segments, ylabel, max_labels=40):
    """Draw the TTL pulse track as one step trace per re-anchored segment.

    ``segments`` is a list of (starts, stops, indices) arrays (one per segment, already shifted
    onto their paired trials, and including the segment's skipped pulses drawn as leading pulses
    before its first kept one).  Each segment is a separately-coloured step trace with green rising
    / cyan falling ticks; the caller marks the skipped pulses with identifying lines.  Axis
    cosmetics and the legend are set once, mirroring plotting.plot_interval_track.
    """
    total = sum(len(s[0]) for s in segments)
    label_every = max(1, int(np.ceil(total / max_labels)))
    drawn = 0
    for si, (starts, stops, indices) in enumerate(segments):
        starts = np.asarray(starts, dtype=float)
        stops = np.asarray(stops, dtype=float)
        color = _SEGMENT_COLORS[si % len(_SEGMENT_COLORS)]
        xs, ys = plotting.interval_step(starts, stops)
        ax.plot(xs, ys, color=color, linewidth=1.0, label='pulses (segment {})'.format(si + 1))
        ax.plot(starts, np.ones_like(starts), '|', color='tab:green', markersize=10,
                label='rising' if si == 0 else None)
        finite_stops = stops[np.isfinite(stops)]
        ax.plot(finite_stops, np.zeros_like(finite_stops), '|', color='tab:cyan',
                markersize=10, label='falling' if si == 0 else None)
        for k, (s, idx) in enumerate(zip(starts, indices)):
            if (drawn + k) % label_every == 0:
                ax.annotate(str(idx), xy=(s, 1.0), xytext=(s, 1.12), ha='center',
                            va='bottom', fontsize=7, color=color, rotation=90)
        drawn += len(starts)
    ax.axvline(0.0, color='tab:blue', linestyle='--', linewidth=0.8)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['low', 'high'])
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', fontsize=8)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_ttl_trial_alignment(server, processed_server, session, probe_type, skip=None,
                             ignore=None, recording=None, save=True, save_dir=None):
    """Plot TTL pulses over trial windows, aligned by the first pulse-trial.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        skip {int} --- Pulses to skip for alignment; if negative, that many trials
            are skipped instead.  The reference (t=0) becomes rising[skip] and
            trial_start[0] (skip >= 0) or rising[0] and trial_start[-skip] (skip < 0).
            None -> meta_neural.json 'skip_ttl' then 0.
        ignore {int} --- Drop the first N TTL pulses and the first N trial starts
            outright before aligning (applied before `skip`), for sessions whose
            leading pulses/trials are spurious.  None -> meta_neural.json 'ignore'
            then 0.
        recording {int|str} --- Open Ephys recording within experiment1 to read,
            1-based (Recording1, Recording2, ...); selects which recording's TTL
            events are read. None -> probe default.
        save {bool} --- Save the figure as PNG (default True) to
            <processed_server>/<session>/prehension_plots/figure_ttl_alignment.png.
        save_dir {str} --- Explicit output folder overriding the default
            <session>/prehension_plots location. None -> the default (see save).
    """
    cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type,
                                recording=recording)
    # skip: CLI kwarg > meta_neural.json 'skip_ttl' > 0
    skip = resolve_meta_arg(skip, cfg.meta_neural, 'skip_ttl', 0)
    # ignore: CLI kwarg > meta_neural.json 'ignore' > 0
    ignore = resolve_meta_arg(ignore, cfg.meta_neural, 'ignore', 0)

    # Configured intermediate skips to flag (read from the meta files only, as the analysis /
    # plotting scripts do): meta_neural 'skip_ttl_intermediate' pulses (by their original index)
    # and meta_structure 'skip_trials' trials (by trial number).
    skip_ttl_intermediate = set(
        int(i) for i in (resolve_meta_arg(
            None, cfg.meta_neural, 'skip_ttl_intermediate', None) or []))
    mstruct = meta_session.import_meta_structure(
        os.path.join(cfg.pserv, 'meta_structure.json'), raw_dir=cfg.rserv, proc_dir=cfg.pserv)
    skip_trials = set(int(t) for t in (mstruct.get('skip_trials') or []))
    # reanchor_ttl: pulse indices (original numbering) at which to re-anchor the plotted
    # pulse<->trial correspondence WITHOUT dropping any pulse (unlike skip_ttl_intermediate), to
    # realign the traces across a drift or a recording gap.
    reanchor_ttl = set(
        int(i) for i in (resolve_meta_arg(None, cfg.meta_neural, 'reanchor_ttl', None) or []))

    # neural TTL edges (times only; no recording load needed)
    edges = events.extract_ttl_edge_times(cfg, verbose=True)
    rising = (np.asarray(edges['rising_times_s'], dtype=float)
              if edges['rising_times_s'] is not None else np.array([]))
    falling = (np.asarray(edges['falling_times_s'], dtype=float)
               if edges['falling_times_s'] is not None else np.array([]))

    # behavioural trial windows (in recording order, aligned positionally with the pulses)
    trial_num, dup_index, start_s, end_s = read_trial_sync_times(cfg)

    # ignore: drop the first N pulses and first N trials outright before aligning
    if ignore and ignore > 0:
        rising = rising[ignore:]
        falling = falling[ignore:]
        trial_num = trial_num[ignore:]
        dup_index = dup_index[ignore:]
        start_s = start_s[ignore:]
        end_s = end_s[ignore:]

    rs('{} TTL pulses; {} behavioural trials; skip={}; ignore={}.'.format(
        len(rising), len(start_s), skip, ignore))
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
    # Pulse track, labelled by ORIGINAL index (offset by ignore) so the numbering matches the
    # behavioural trial numbers.  The kept pulses are split into re-anchored segments at each
    # skipped intermediate pulse (skip_ttl_intermediate, dropped) and each re-anchor pulse
    # (reanchor_ttl, kept): every post-first segment is shifted so its first kept pulse sits on the
    # next unclaimed, unskipped trial, each drawn as its own trace.  The walk honours the leading
    # skip (pulse_ref / trial_ref), so this is correct for skip != 0.  With no boundaries it stays
    # the single original track.
    seg_defs = _pair_pulse_segments(len(rising), ignore, pulse_ref, trial_ref, trial_num,
                                    skip_ttl_intermediate, reanchor_ttl, skip_trials)
    reanchored = len(seg_defs) > 1 and len(falling) == len(rising)
    # per-segment re-anchor shift: 0 for the first segment (and the single-track fallback), else the
    # offset that puts the segment's first KEPT pulse onto its paired trial.  The segment's skipped
    # pulses are drawn with the same shift, so they sit just before that trial.
    deltas = [((start_s[anchor_ti] - t0_trial) - rising_a[js[0]])
              if reanchored and si > 0 and anchor_ti is not None and valid[anchor_ti] and js
              else 0.0
              for si, (js, anchor_ti, _dropped) in enumerate(seg_defs)]
    if reanchored:
        segments, kept_drawn, skipped_pulse_x = [], [], []
        for (js, _anchor, dropped), delta in zip(seg_defs, deltas):
            kept = np.asarray(js, dtype=int)
            # draw the segment's skipped pulses (before its trial) and its kept pulses together in
            # one frame; only the kept pulses feed the alignment check.
            all_js = np.array(sorted(dropped + list(js)), dtype=int)
            segments.append((rising_a[all_js] + delta, falling_a[all_js] + delta, ignore + all_js))
            if kept.size:
                kept_drawn.append(rising_a[kept] + delta)
            skipped_pulse_x += [rising_a[j] + delta for j in dropped]
        _plot_pulse_segments(ax1, segments, 'TTL pulses')
        rising_drawn = np.concatenate(kept_drawn) if kept_drawn else np.array([])
        seg_note = '; {} re-anchored segments'.format(len(segments))
    else:
        if len(seg_defs) > 1:
            ws('Not re-anchoring pulse segments ({} rising vs {} falling edges unequal); '
               'showing a single track.'.format(len(rising), len(falling)))
        plotting.plot_interval_track(ax1, rising_a, falling_a,
                                     np.arange(ignore, ignore + len(rising)),
                                     'TTL pulses', 'rising', 'falling')
        rising_drawn = rising_a
        skipped_pulse_x = [rising_a[j] for (_js, _a, dropped) in seg_defs for j in dropped]
        seg_note = ''
    ax1.set_title('TTL pulses (n={}) vs trials (n={}); aligned by pulse {} <-> '
                  'trial {} (skip={}{})'.format(len(rising), len(start_s), pulse_ref,
                                                trial_ref, skip, seg_note))
    # label trials by their composite id (trial number + '_1', ... for duplicate recordings)
    trial_labels = ['{}{}'.format(trial_num[i], '' if dup_index[i] == 0 else '_%d' % dup_index[i])
                    for i in trial_idx]
    plotting.plot_interval_track(ax2, start_a, end_a, trial_labels, 'trials',
                                 'trial start', 'trial end')
    ax2.set_xlabel('time from alignment (s)')

    # Identifying lines through the skips on both tracks, in the display (re-anchored) frame:
    # skipped intermediate pulses (purple) -- now drawn as leading pulses of the segment they
    # belong to, so the line passes through them just before that segment's trial -- and skipped
    # trials (magenta) at their trial time.  Both are solid vertical lines on both subplots.
    skipped_trial_x = [start_a[m] for m, i in enumerate(trial_idx)
                       if int(trial_num[i]) in skip_trials]
    for xs, color, name in ((skipped_pulse_x, 'purple', 'skipped pulse'),
                            (skipped_trial_x, 'magenta', 'skipped trial')):
        if xs:
            rs('Flagging {} {}(s).'.format(len(xs), name))
            plotting.plot_event_lines([ax1, ax2], xs, color=color, linestyle='-',
                                      linewidth=1.2, label=name)

    # Alignment check, run AFTER re-anchoring: a trial start with no rising TTL within
    # TTL_MATCH_TOL_S of a *drawn* (re-anchored) pulse is marked misaligned with a red vertical
    # line in both subplots.  Skipped trials (skip_trials) are excluded from the check -- they are
    # shown as skipped (magenta), not misaligned.
    skipped_trial_mask = np.array(
        [int(trial_num[i]) in skip_trials for i in trial_idx], dtype=bool)
    missing = misc.unmatched_mask(start_a, rising_drawn, TTL_MATCH_TOL_S) & ~skipped_trial_mask
    n_checked = int(np.count_nonzero(~skipped_trial_mask))
    n_missing = int(np.count_nonzero(missing))
    if n_missing:
        missing_labels = [trial_labels[k] for k in np.flatnonzero(missing)]
        ws('{} trial start(s) misaligned (no TTL pulse within {:.0f} ms after re-anchoring): '
           '{}'.format(n_missing, TTL_MATCH_TOL_S * 1000, missing_labels))
        for j, x in enumerate(start_a[missing]):
            lbl = 'misaligned trial' if j == 0 else None
            ax1.axvline(x, color='red', linewidth=1.2, alpha=0.8, label=lbl)
            ax2.axvline(x, color='red', linewidth=1.2, alpha=0.8, label=lbl)
    else:
        rs('All {} checked trial start(s) have a TTL pulse within {:.0f} ms after '
           're-anchoring.'.format(n_checked, TTL_MATCH_TOL_S * 1000))

    # refresh the legends whenever extra markers (skips or misaligned trials) were added
    if skipped_pulse_x or skipped_trial_x or n_missing:
        ax1.legend(loc='upper right', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)

    fig.tight_layout()
    save_dir = resolve_session_save_dir(processed_server, session, save, save_dir)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, figure_filename('figure_ttl_alignment'))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
