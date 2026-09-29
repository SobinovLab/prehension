#!python3
# -*- coding: utf-8 -*-
"""
Per-session diagnostic overview figure through (continuous) recording time.

Reads the neural product (spikes + TTL sync) from the session NWB and the behavioural
trial data from the prehension meta, and stacks whatever is available onto a shared time
axis:

  * a spike raster (a tick per spike), units sorted by depth along the probe;
  * the population mean firing rate (Hz) through time;
  * the total grasp force, one trace per pressure sensor (medial / lateral);
  * the joint angles, selected the same way as the encoding models
    (tools.filters.filter_ra_dofs: the right-arm independent DOFs of a joint group).

Each trial's behavioural signals are stored in seconds since that trial's TTL pulse, so
they are shifted back to their recording time (by the pulse time paired to the trial) and
stitched -- with gaps between trials -- into one continuous trace per channel; the spikes
are already continuous.  The number of subplots depends on which data a session has: the
neural panels are drawn only when the NWB has units, the force panel only when a trial has
pressure-sensor files, and the joint-angle panel only when a trial has processed kinematics.

The timeline is split into separate figures covering a fixed number of minutes each (10 by
default), one figure per page, so a long recording stays readable.  The unit selection
(depth order and the activity threshold) is computed once over the whole session, so the
same units occupy the same raster rows on every page.

TTL pulses are drawn as vertical red dashed lines and the trial reward/end times
(meta_session 'ttl_to_reward') as vertical green dashed lines, across every subplot.  The
pulse<->trial pairing is positional (meta_neural skip_ttl / skip_ttl_last / skip_ttl_intermediate
and meta_structure skip_trials), as in the other neural figures; inspect a count mismatch with
figure_ttl_alignment.  Skipped items are flagged too: spurious pulses (skip_ttl_intermediate) in
purple and skipped trials (skip_trials) in magenta.  When meta_neural 'reanchor_ttl' lists a pulse
index, the empty recording span before it (a gap / drift) is collapsed on the continuous axis --
the whole timeline (spikes, behaviour, event lines) is shifted, so real recording time is no
longer preserved across a re-anchor.

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
import scipy.ndimage

from .. import meta_session
from ..tools import io, forces, filters, plotting
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..neural_processing.common.spikes import (
    BIN_WIDTH, FILTER_SIGMA, read_nwb_spikes_and_ttl, read_nwb_unit_depths)
from .common.pooling import session_neural_context, pair_pulses_to_trials
from .common.behaviour import load_timepoints_into_msession, get_timepoint
from .common.traces import resolve_session_save_dir, figure_filename

# meta_session offset (seconds since the trial's TTL pulse) of the trial reward / end
# (create_meta 'ttl_to_reward'); drawn as the green event lines.
REWARD_KEY = 'ttl_to_reward'
DEFAULT_JOINT_GROUP = filters.DEFAULT_JOINT_GROUP

# padding (s) added on each side of the trials' span for the plotted time window
WINDOW_PAD_S = 0.5

# default width (minutes) of each figure; the session timeline is tiled into pages of this
# length, one figure per page, so a long recording stays readable
DEFAULT_MINUTES_PER_FIGURE = 10.0


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------
def _select_trials(msession, trials):
    """Indices of the trials to plot: those whose trial_number is in `trials`, else all."""
    if not trials:
        return list(range(len(msession)))
    wanted = set(int(t) for t in trials)
    sel = [i for i, t in enumerate(msession) if t.trial_number in wanted]
    if not sel:
        raise ValueError('None of the requested trials {} are in the session.'.format(
            sorted(wanted)))
    return sel


def _gather_force_segments(msession, events_time, selection, t0):
    """Per-sensor continuous grasp-force segments, in display time (recording time - t0).

    For every selected trial with filtered pressure-sensor files, reads each sensor's summed
    force (forces.load_per_sensor_force_traces) and shifts its seconds-since-TTL times to
    recording time by the trial's TTL pulse.  Returns {sensor_name: [(x, y), ...]} with one
    (times, force) segment per contributing trial, in recording order.
    """
    segments = {}
    for i in selection:
        trial = msession[i]
        if not trial.do_pre_ps_files_exist():
            continue
        try:
            traces = forces.load_per_sensor_force_traces(trial.get_pre_ps_filenames())
        except Exception as e:  # noqa: BLE001
            ws('Trial {}: could not read force ({}); skipping its force.'.format(
                trial.trial_number, e))
            continue
        ttl_start = float(events_time[i][0])
        for name, (times, force) in traces.items():
            segments.setdefault(name, []).append((times + ttl_start - t0, force))
    return segments


def _gather_joint_segments(msession, events_time, selection, t0, joint_group):
    """Per-DOF continuous joint-angle segments, in display time (recording time - t0).

    For every selected trial with processed kinematics, loads the joint-angle CSV, keeps the
    right-arm independent DOFs of `joint_group` (tools.filters.filter_ra_dofs, the same
    selection the encoding models use) and shifts the seconds-since-TTL times to recording
    time by the trial's TTL pulse.  Returns (segments, dof_names) where segments is
    {dof_name: [(x, y), ...]} in recording order and dof_names preserves the DOF order.
    """
    segments = {}
    dof_names = None
    for i in selection:
        trial = msession[i]
        if not trial.does_post_kin_file_exist():
            continue
        try:
            times, names, values = io.import_timed_csv(trial.post_kinematic_filename_csv)
            names, values = filters.filter_ra_dofs(names, values, joint_group)
        except Exception as e:  # noqa: BLE001
            ws('Trial {}: could not read joint angles ({}); skipping its kinematics.'.format(
                trial.trial_number, e))
            continue
        times = np.asarray(times, dtype=float)
        ttl_start = float(events_time[i][0])
        dof_names = list(names) if dof_names is None else dof_names
        for name, row in zip(names, values):
            segments.setdefault(name, []).append((times + ttl_start - t0, row))
    return segments, (dof_names or [])


def _event_times(msession, events_time, selection, t0, reward_key):
    """Display-time TTL pulse times and reward times over the selected trials.

    Returns (ttl_times, reward_times): the TTL pulse (rising-edge) time of each selected
    trial, and, where the trial has a finite `reward_key` offset (seconds since the pulse),
    the reward time -- both in display time (recording time - t0).
    """
    ttl_times = np.array([float(events_time[i][0]) - t0 for i in selection])
    reward_times = []
    for i in selection:
        r = get_timepoint(msession[i], reward_key)
        if r is not None:
            reward_times.append(float(events_time[i][0]) + r - t0)
    return ttl_times, np.array(reward_times)


def _span(segments_dicts, ttl_times, ttl_stops, reward_times):
    """[lo, hi] display-time window spanning the trials and all gathered behaviour."""
    lo_candidates = [ttl_times]
    hi_candidates = [ttl_times, ttl_stops, reward_times]
    for segs in segments_dicts:
        for seg_list in segs.values():
            for x, _ in seg_list:
                if x.size:
                    lo_candidates.append(x[:1])
                    hi_candidates.append(x[-1:])
    lo = min(float(np.min(c)) for c in lo_candidates if np.size(c))
    hi = max(float(np.max(c)) for c in hi_candidates if np.size(c))
    return lo - WINDOW_PAD_S, hi + WINDOW_PAD_S


def _population_rate(spikes_disp, n_units, window, bin_width, filter_sigma):
    """Population mean firing rate (Hz) through time over the display window.

    Histograms all units' (display-time) spikes into `bin_width` bins across `window`,
    divides by (n_units * bin_width) to get the per-unit mean rate, and Gaussian-smooths it.
    Returns (bin_centers, rate).
    """
    lo, hi = window
    bins = np.arange(lo, hi + bin_width, bin_width)
    centers = bins[:-1] + bin_width / 2
    allspikes = np.concatenate(spikes_disp) if spikes_disp else np.array([])
    counts, _ = np.histogram(allspikes, bins=bins)
    rate = counts / (max(n_units, 1) * bin_width)
    if filter_sigma > 0:
        rate = scipy.ndimage.gaussian_filter1d(rate, filter_sigma / bin_width)
    return centers, rate


def _kept_pulse_trial_indices(n_pulses, msession, skip_ttl, skip_ttl_last,
                              skip_ttl_intermediate, skip_trials):
    """Original pulse / trial indices kept by pair_pulses_to_trials, in trimmed order.

    Mirrors the trimming in neural_plotting.common.pooling.pair_pulses_to_trials so each trimmed
    pulse / trial can be mapped back to its original index (used for re-anchoring and skip
    flagging): intermediate pulses and skip_trials are dropped, then the leading/trailing skip_ttl
    offset.  Returns (kept_pulse_idx, kept_trial_idx), each a list of original indices with the
    same length as the paired (trimmed) events_time / msession.
    """
    drop_p = set(int(i) for i in (skip_ttl_intermediate or []))
    drop_t = set(int(t) for t in (skip_trials or []))
    kept_pulse_idx = [i for i in range(n_pulses) if i not in drop_p]
    kept_trial_idx = [k for k, t in enumerate(msession) if t.trial_number not in drop_t]
    if skip_ttl and skip_ttl > 0:
        kept_pulse_idx = kept_pulse_idx[skip_ttl:]
    elif skip_ttl and skip_ttl < 0:
        kept_trial_idx = kept_trial_idx[-skip_ttl:]
    if skip_ttl_last and skip_ttl_last > 0:
        kept_pulse_idx = kept_pulse_idx[:-skip_ttl_last]
    elif skip_ttl_last and skip_ttl_last < 0:
        kept_trial_idx = kept_trial_idx[:skip_ttl_last]
    return kept_pulse_idx, kept_trial_idx


def _build_reanchor_remap(sel_starts, sel_stops, reanchor_mask, pad):
    """Piecewise time-shift that closes the empty span before each re-anchor boundary.

    ``sel_starts`` / ``sel_stops`` are the selected trials' TTL start / stop times (display time,
    recording - t0) in plotting order; ``reanchor_mask[q]`` marks a trial whose pulse is a
    reanchor_ttl point (a new segment starts there).  At each boundary the empty span between the
    previous trial's stop and this trial's start beyond ``pad`` is removed from everything after
    it, so a big recording gap collapses.  Returns (remap, n_closed): remap(times) -> shifted
    times (identity when nothing is re-anchored), applicable to any display-time array (spikes,
    behaviour, events), and the number of gaps closed.
    """
    boundary_times, gaps = [], []
    for q in range(1, len(sel_starts)):
        if reanchor_mask[q]:
            gap = float(sel_starts[q] - sel_stops[q - 1]) - pad
            if gap > 0:
                boundary_times.append(float(sel_starts[q]))
                gaps.append(gap)
    if not boundary_times:
        return (lambda t: np.asarray(t, dtype=float)), 0
    boundary_times = np.array(boundary_times)
    cum = np.concatenate([[0.0], np.cumsum(gaps)])

    def remap(t):
        t = np.asarray(t, dtype=float)
        return t - cum[np.searchsorted(boundary_times, t, side='right')]
    return remap, len(boundary_times)


def _remap_segments(segments, remap):
    """Apply the display-time remap to the x of every (x, y) trace in a segment dict."""
    return {name: [(remap(x), y) for x, y in seg_list] for name, seg_list in segments.items()}


def _skipped_trial_marker_times(msession_full, skip_trials, kept_trial_idx, selection, sel_starts):
    """Display-time markers for skip_trials trials (which have no neural pulse of their own).

    Each skipped trial is placed at the pulse of the next selected (kept) trial in recording order
    -- an approximate 'a trial was skipped here' marker -- or the last selected trial when none
    follows.  Returns a 1-D array (empty when nothing is skipped / selected).
    """
    drop_t = set(int(t) for t in (skip_trials or []))
    if not drop_t or len(sel_starts) == 0:
        return np.array([])
    sel_full = np.array([kept_trial_idx[i] for i in selection])
    xs = []
    for k, trial in enumerate(msession_full):
        if trial.trial_number in drop_t:
            pos = int(np.searchsorted(sel_full, k))
            xs.append(sel_starts[pos] if pos < len(sel_starts) else sel_starts[-1])
    return np.array(xs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot_raster(ax, spikes_disp, row_labels, window):
    """Depth-sorted spike raster on ax (rows already ordered tip-at-bottom)."""
    plotting.plot_spike_raster(ax, spikes_disp, row_labels=row_labels)
    ax.set_xlim(window)
    ax.set_ylabel('unit (by depth,\ntip at bottom)', fontsize=8)


def _plot_rate(ax, centers, rate):
    """Population mean firing-rate trace on ax."""
    ax.plot(centers, rate, color='k', linewidth=0.8)
    ax.set_ylabel('mean rate (Hz)', fontsize=8)


def _plot_force(ax, force_segments):
    """One continuous grasp-force trace per pressure sensor on ax."""
    for name in sorted(force_segments):
        x, y = plotting.stitch_traces_with_gaps(force_segments[name])
        ax.plot(x, y, linewidth=0.8, label=name)
    ax.set_ylabel('grasp force\n(a.u.)', fontsize=8)
    ax.legend(loc='upper right', fontsize=7, ncol=len(force_segments))


def _plot_joints(ax, joint_segments, dof_names, joint_group):
    """One continuous trace per selected joint-angle DOF on ax, coloured along the group."""
    cmap = plt.get_cmap('viridis')
    n = max(len(dof_names), 1)
    for j, name in enumerate(dof_names):
        x, y = plotting.stitch_traces_with_gaps(joint_segments[name])
        ax.plot(x, y, linewidth=0.7, color=cmap(j / n), label=name)
    ax.set_ylabel('joint angle', fontsize=8)
    ax.legend(loc='upper right', fontsize=5, ncol=max(1, len(dof_names) // 6),
              title='{} DOFs'.format(joint_group), title_fontsize=6)


def _page_suffix(i_page, n_pages):
    """Filename suffix for a time page: '' for a single page, else 'pN' (1-based)."""
    return '' if n_pages <= 1 else 'p{}'.format(i_page + 1)


def _draw_overview_figure(session, panels, page_window, spikes_shown, row_labels, n_shown,
                          n_units, min_rate, force_segments, joint_segments, dof_names,
                          joint_group, ttl_times, reward_times, skipped_pulse_x, skipped_trial_x,
                          bin_width, filter_sigma, use_threshold_crossings, reanchored,
                          i_page, n_pages):
    """Draw one time page (a `page_window` slice) of the session overview into a new figure.

    The panels and the (depth-sorted, min_rate-filtered) unit set are fixed by the caller, so
    every page shares the same layout and raster rows; here the spikes and event lines are
    clipped to this page's window and the panels are drawn.  Returns the figure.
    """
    p_lo, p_hi = page_window

    def _in_page(x):
        x = np.asarray(x, dtype=float)
        return x[(x >= p_lo) & (x <= p_hi)]

    # spikes / events restricted to this page (rows already depth-sorted; same set every page)
    page_spikes = [s[(s >= p_lo) & (s <= p_hi)] for s in spikes_shown]
    ttl_page = _in_page(ttl_times)
    reward_page = _in_page(reward_times)
    skipped_pulse_page = _in_page(skipped_pulse_x)
    skipped_trial_page = _in_page(skipped_trial_x)

    heights = {'raster': 3.0, 'rate': 1.6, 'force': 1.8, 'joints': 2.2}
    fig, axs = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                            figsize=(16, sum(heights[p] for p in panels)),
                            gridspec_kw={'height_ratios': [heights[p] for p in panels]})
    axs = axs.ravel()
    ax_by_panel = dict(zip(panels, axs))

    if 'raster' in ax_by_panel:
        _plot_raster(ax_by_panel['raster'], page_spikes, row_labels, page_window)
    if 'rate' in ax_by_panel:
        centers, rate = _population_rate(page_spikes, n_shown, page_window, bin_width, filter_sigma)
        _plot_rate(ax_by_panel['rate'], centers, rate)
    if 'force' in ax_by_panel:
        _plot_force(ax_by_panel['force'], force_segments)
    if 'joints' in ax_by_panel:
        _plot_joints(ax_by_panel['joints'], joint_segments, dof_names, joint_group)

    # TTL pulses (red) and reward times (green) as dashed vertical lines on every subplot;
    # skipped pulses (purple) and skipped trials (magenta) as solid lines where present
    plotting.plot_event_lines(axs, ttl_page, color='red', label='TTL pulse')
    plotting.plot_event_lines(axs, reward_page, color='green', label='reward')
    if skipped_pulse_page.size:
        plotting.plot_event_lines(axs, skipped_pulse_page, color='purple', linestyle='-',
                                  label='skipped pulse')
    if skipped_trial_page.size:
        plotting.plot_event_lines(axs, skipped_trial_page, color='magenta', linestyle='-',
                                  label='skipped trial')
    axs[0].set_xlim(page_window)
    axs[0].legend(loc='upper left', fontsize=7)
    axs[-1].set_xlabel('{}time from first TTL pulse (s)'.format(
        're-anchored ' if reanchored else ''))

    units_note = ('{} units'.format(n_shown) if min_rate is None or n_shown == n_units
                  else '{}/{} units > {} Hz'.format(n_shown, n_units, min_rate))
    page_note = ' (page {}/{})'.format(i_page + 1, n_pages) if n_pages > 1 else ''
    fig.suptitle('Session overview -- {}{} [{:.1f}-{:.1f} min]{} ({})'.format(
        session, ' (threshold crossings)' if use_threshold_crossings else '',
        p_lo / 60.0, p_hi / 60.0, page_note, units_note))
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    return fig


def plot_session_overview(server, processed_server, session, trials=None,
                          joint_group=DEFAULT_JOINT_GROUP, bin_width=BIN_WIDTH,
                          filter_sigma=FILTER_SIGMA, min_rate=None,
                          minutes_per_figure=DEFAULT_MINUTES_PER_FIGURE, reward_key=REWARD_KEY,
                          skip_ttl=None, skip_ttl_last=None, use_threshold_crossings=False,
                          save=True, save_dir=None):
    """Draw the per-session diagnostic overview through continuous recording time.

    Assembles, on one shared time axis, whichever of these a session has: the depth-sorted
    spike raster and population mean firing rate (when the NWB has units), the per-sensor
    total grasp force, and the joint angles (right-arm independent DOFs of `joint_group`,
    selected as in the encoding models).  Behaviour is aligned to recording time by each
    trial's TTL pulse and stitched -- with gaps between trials -- into a continuous trace;
    the time axis is measured from the first plotted pulse.  TTL pulses (red) and reward
    times (green) are drawn as dashed vertical lines on every subplot.

    The timeline is tiled into pages of `minutes_per_figure` minutes, one figure per page, so
    a long recording stays readable.  The unit selection (depth order and the `min_rate`
    activity filter) is computed once over the whole session, so the same units appear at the
    same raster rows on every page.  Returns the list of figures (one per page).

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        trials {list} --- Trial numbers to plot; None/empty -> every trial.
        joint_group {str} --- Right-arm DOF group for the joint-angle panel: 'hand'
            (distal), 'proximal' (shoulder + elbow) or 'all' (tools.filters.JOINT_GROUPS).
        bin_width {float} --- Bin width (s) of the population firing-rate trace.
        filter_sigma {float} --- SD (s) of the Gaussian smoothing of the rate trace.
        min_rate {float} --- Optional activity threshold (Hz): drop units whose mean firing
            rate over the whole session window is at or below this (so the same units appear
            on every page). None -> meta_neural.json 'min_rate' if present, else no filter
            (as in figure_peth).
        minutes_per_figure {float} --- Width (minutes) of each figure; the timeline is split
            into pages of this length, one figure per page. Default: 10 min.
        reward_key {str} --- meta_session offset column (s since the pulse) for the reward
            lines; default 'ttl_to_reward'.
        skip_ttl {int} --- Positional pulse<->trial offset (positive drops leading pulses,
            negative leading trials). None -> meta_neural.json 'skip_ttl' then 0.
        skip_ttl_last {int} --- Like skip_ttl but trimming the end. None ->
            meta_neural.json 'skip_ttl_last' then 0.
        use_threshold_crossings {bool} --- Read the threshold-crossing product instead of
            the sorted neural.nwb (the sorted file is used by default, or the threshold
            crossings with a warning when it is missing).
        save {bool} --- Save each page as PNG (default True), into
            <processed_server>/<session>/prehension_plots/figure_session_overview.png (a
            single page) or figure_session_overview_pN.png (multiple pages).
        save_dir {str} --- Explicit output folder overriding the default. None -> default.
    """
    nwb_path, meta_neural, rserv, pserv = session_neural_context(
        server, processed_server, session, use_threshold_crossings)
    skip_ttl = resolve_meta_arg(skip_ttl, meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(skip_ttl_last, meta_neural, 'skip_ttl_last', 0)
    skip_ttl_intermediate = resolve_meta_arg(None, meta_neural, 'skip_ttl_intermediate', None)
    reanchor_ttl = set(int(i) for i in (resolve_meta_arg(
        None, meta_neural, 'reanchor_ttl', None) or []))
    min_rate = resolve_meta_arg(min_rate, meta_neural, 'min_rate', None)

    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(nwb_path)
    unit_depths = read_nwb_unit_depths(nwb_path)

    mstruct, _, _, msession = meta_session.load_meta_information(rserv, pserv)
    # best-effort: the default reward key ('ttl_to_reward') lives on meta_session (trial
    # other_info), so a missing / partial timepoints CSV must not break this diagnostic;
    # get_timepoint falls back to other_info when a trial has no attached timepoints.
    try:
        load_timepoints_into_msession(msession, mstruct)
    except Exception as e:  # noqa: BLE001
        ws('Could not load timepoints for {} ({}); using meta_session offsets only.'.format(
            session, e))

    # pair pulses to trials (dropping skipped pulses / trials); keep the full arrays and the kept
    # original indices so the skipped items can be flagged and the reanchor_ttl pulses located.
    skip_trials = mstruct.get('skip_trials', [])
    events_time_full, msession_full = events_time, msession
    kept_pulse_idx, kept_trial_idx = _kept_pulse_trial_indices(
        len(events_time_full), msession_full, skip_ttl, skip_ttl_last,
        skip_ttl_intermediate, skip_trials)
    events_time, msession = pair_pulses_to_trials(
        events_time, msession, skip_ttl, skip_ttl_last,
        skip_ttl_intermediate=skip_ttl_intermediate, skip_trials=skip_trials)
    if len(events_time) != len(msession):
        raise ValueError(
            '{} TTL pulses but {} behavioural trials (after skip_ttl={}, skip_ttl_last={}). '
            'Pulses are paired to trials by position; inspect the correspondence with '
            'figure_ttl_alignment before plotting.'.format(
                len(events_time), len(msession), skip_ttl, skip_ttl_last))
    if not events_time:
        raise ValueError('Session {} has no TTL pulses to align to.'.format(session))

    selection = _select_trials(msession, trials)
    t0 = float(events_time[selection[0]][0])

    # Re-anchor: at each selected trial whose pulse is a reanchor_ttl point, collapse the empty
    # span (recording gap / drift) before it on the continuous axis.  remap() then applies the
    # same piecewise shift to every display-time array below (spikes, behaviour, event lines).
    sel_starts = np.array([float(events_time[i][0]) - t0 for i in selection])
    sel_stops = np.array([float(events_time[i][1]) - t0 for i in selection])
    reanchor_mask = np.array([kept_pulse_idx[i] in reanchor_ttl for i in selection], dtype=bool)
    remap, n_closed = _build_reanchor_remap(sel_starts, sel_stops, reanchor_mask, WINDOW_PAD_S)
    if n_closed:
        rs('Re-anchoring: closed {} recording gap(s) at reanchor_ttl point(s).'.format(n_closed))
    ttl_stops = remap(sel_stops)

    # behavioural segments (recording time - t0, then re-anchored), stitched per channel with gaps
    force_segments = _remap_segments(
        _gather_force_segments(msession, events_time, selection, t0), remap)
    joint_segments, dof_names = _gather_joint_segments(
        msession, events_time, selection, t0, joint_group)
    joint_segments = _remap_segments(joint_segments, remap)
    ttl_times, reward_times = _event_times(msession, events_time, selection, t0, reward_key)
    ttl_times, reward_times = remap(ttl_times), remap(reward_times)

    # skip flags (re-anchored): spurious pulses (purple) at their own recording time; skipped
    # trials (magenta) placed at the next selected trial (they carry no neural pulse of their own).
    skipped_pulse_x = remap(np.array(
        [float(events_time_full[oi][0]) - t0
         for oi in sorted(set(int(i) for i in (skip_ttl_intermediate or [])))
         if oi < len(events_time_full)]))
    skipped_trial_x = remap(_skipped_trial_marker_times(
        msession_full, skip_trials, kept_trial_idx, selection, sel_starts))

    n_units = len(unit_ids)
    window = _span([force_segments, joint_segments], ttl_times, ttl_stops, reward_times)
    lo, hi = window
    dur = max(hi - lo, 1e-9)

    # Global unit selection (shared by every time page so the raster rows are stable): spikes in
    # display time sorted by depth (tip / smallest at bottom); with min_rate set, drop units whose
    # mean rate over the WHOLE session window is at or below it.
    spikes_shown, row_labels = [], []
    if n_units:
        order = sorted(range(n_units),
                       key=lambda i: (np.isnan(unit_depths.get(unit_ids[i], np.nan)),
                                      unit_depths.get(unit_ids[i], np.nan)))
        for i in order:
            s = remap(np.asarray(spikes[i], dtype=float) - t0)
            s = s[(s >= lo) & (s <= hi)]
            if min_rate is not None and s.size / dur <= min_rate:
                continue
            spikes_shown.append(s)
            row_labels.append(str(unit_ids[i]))
        if min_rate is not None:
            rs('Activity filter: keeping {} / {} unit(s) with mean rate > {} Hz over the '
               'full session window.'.format(len(spikes_shown), n_units, min_rate))
    n_shown = len(spikes_shown)

    # panels, top to bottom, present (on every page) only when their data exists
    panels = []
    if n_shown:
        panels.append('raster')
        panels.append('rate')
    if force_segments:
        panels.append('force')
    if joint_segments:
        panels.append('joints')
    if not panels:
        raise ValueError('Session {} has neither neural units (above the activity threshold) '
                         'nor behavioural signals to plot.'.format(session))

    # tile the timeline into pages of `minutes_per_figure`, one figure each
    chunk = max(float(minutes_per_figure) * 60.0, bin_width)
    n_pages = max(1, int(np.ceil((hi - lo) / chunk)))
    rs('Session {}: {} trials, {} of {} units shown; panels: {}; {} figure(s) of {:g} min.'.format(
        session, len(selection), n_shown, n_units, ', '.join(panels), n_pages,
        minutes_per_figure))

    save_dir = resolve_session_save_dir(processed_server, session, save, save_dir)
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    figures = []
    for i_page in range(n_pages):
        page_window = (lo + i_page * chunk, min(lo + (i_page + 1) * chunk, hi))
        fig = _draw_overview_figure(
            session, panels, page_window, spikes_shown, row_labels, n_shown, n_units, min_rate,
            force_segments, joint_segments, dof_names, joint_group, ttl_times, reward_times,
            skipped_pulse_x, skipped_trial_x, bin_width, filter_sigma,
            use_threshold_crossings, n_closed > 0, i_page, n_pages)
        if save_dir is not None:
            out = os.path.join(save_dir, figure_filename(
                'figure_session_overview', _page_suffix(i_page, n_pages)))
            fig.savefig(out, dpi=150, bbox_inches='tight')
            rs('Saved {}'.format(out))
        figures.append(fig)
    return figures
