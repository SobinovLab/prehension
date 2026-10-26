#!python3
# -*- coding: utf-8 -*-
"""
Aligned neural activity by depth along the probe, for one session.

For one session, every unit's firing rate is aligned to a trial timepoint (default
first_grasp_start), Gaussian-smoothed and averaged over the successful trials
(neural_plotting.common.pooling.load_session_trials / session_unit_rates).  The units
are then grouped by their depth along the probe: the span of unit depths is split into
N equal-width bins (200 by default) and the trial-averaged rates of the units in each
bin are averaged (tools.stats.binned_mean_rows).  Each depth bin is normalized to its
own peak through time (tools.stats.normalize_rows_to_max), so the colour runs from 0 to
1, where 1 is the maximum activity at that depth.  The figure is a heatmap of time vs
depth (electrode tip at the bottom); depth bins without units are drawn in grey.  With
stack_by_depth the precise depth is ignored: every unit gets its own row, stacked in depth
order and normalized to its own peak, with the depths shown as the tick labels.

Recording / skip_ttl / good_neurons come from the session's meta_neural.json, as in
the other neural figures; units without a depth (older NWB products) are left out.

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

from ..tools import plotting, stats
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA, read_nwb_unit_depths)
from ..neural_plotting.common.pooling import load_session_trials, session_unit_rates
from ..neural_plotting.common.traces import resolve_session_save_dir, figure_filename

FIGURE_NAME = 'figure_depth_activity'
N_DEPTH_BINS = 200      # equal-width bins over the span of unit depths
MAX_DEPTH_TICKS = 10    # depth tick labels on the stacked (stack_by_depth) y-axis, at most


def _unit_depths(nwb_path, neuron_labels):
    """Depth (um along the probe) of each unit in neuron_labels; NaN where unknown."""
    depth_by_unit = read_nwb_unit_depths(nwb_path)
    return np.array([depth_by_unit.get(lbl, np.nan) for lbl in neuron_labels], dtype=float)


def _filter_by_rate(rates, depths, min_rate):
    """Keep the units whose mean trial-averaged rate exceeds min_rate Hz (None -> all)."""
    if min_rate is None:
        return rates, depths
    keep = np.mean(rates, axis=1) > min_rate
    rs('Activity filter: kept {} / {} unit(s) with mean rate > {} Hz.'.format(
        int(np.sum(keep)), len(keep), min_rate))
    if not keep.any():
        raise ValueError('No units exceed the {} Hz activity threshold.'.format(min_rate))
    return rates[keep], depths[keep]


def _stack_by_depth(rates, depths):
    """One row per unit, ordered by depth (smallest depth, the tip, at the bottom).

    Units without a depth are left out.  Returns (row_edges (n_units + 1,), rows
    (n_units, n_time), sorted_depths (n_units,)): rows sit on unit-wide bins 0..n_units,
    in the same layout as stats.binned_mean_rows, so the normalization and plot are shared.
    """
    finite = np.isfinite(depths)
    order = np.argsort(depths[finite], kind='stable')
    rows = rates[finite][order]
    return np.arange(rows.shape[0] + 1, dtype=float), rows, depths[finite][order]


def plot_depth_activity(bin_centers, depth_edges, activity, align_key, title, save_dir,
                        row_depths=None):
    """Heatmap of the depth-normalized activity: time on x, depth on y (tip at bottom).

    ``activity`` is (n_rows, n_time) in [0, 1] (NaN for empty depth bins) on the
    ``depth_edges`` row bins (um).  With ``row_depths`` (the depth of each row) the rows
    are units stacked in depth order instead: the y-axis is the unit's rank, its sparse
    tick labels (at most MAX_DEPTH_TICKS) give the depth, and the figure is saved with a '_stacked' suffix.  Saves
    figure_depth_activity_<align_key>[_stacked].png into save_dir (when not None) and
    returns the figure.
    """
    stacked = row_depths is not None
    fig, ax = plt.subplots(figsize=(8, 9))
    mesh = plotting.plot_binned_heatmap(ax, bin_centers, depth_edges, activity,
                                        vmin=0.0, vmax=1.0)
    ax.axvline(0.0, color='w', linewidth=0.8, linestyle='--')
    # the stored depth increases from the electrode tip upward (read_nwb_unit_depths), so
    # ascending limits put the tip at the bottom
    ax.set_ylim(depth_edges[0], depth_edges[-1])
    ax.set_xlabel('Time from {}, s'.format(align_key))
    if stacked:
        step = max(1, int(np.ceil(len(row_depths) / MAX_DEPTH_TICKS)))
        rows = np.arange(len(row_depths))[::step]
        ax.set_yticks(rows + 0.5)
        ax.set_yticklabels(['{:.0f}'.format(row_depths[i]) for i in rows], fontsize=6)
        ax.set_ylabel('Units in depth order (tick: depth along probe, um; tip at bottom)')
    else:
        ax.set_ylabel('Depth along probe, um (tip at bottom)')
    ax.set_title(title, fontsize=9)
    fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.02).set_label(
        'Activity (fraction of the {} maximum)'.format('unit' if stacked else 'depth-bin'))
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = '{}_stacked'.format(align_key) if stacked else align_key
        out = os.path.join(save_dir, figure_filename(FIGURE_NAME, suffix))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


def figure_depth_activity(server, processed_server, session, align_timepoint=ALIGN_TIMEPOINT,
                          before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                          filter_sigma=FILTER_SIGMA, n_depth_bins=N_DEPTH_BINS,
                          stack_by_depth=False, only_good=False, min_rate=None,
                          drift_correct=True, use_threshold_crossings=False, save=True,
                          save_dir=None):
    """Plot one session's aligned, trial-averaged activity binned by unit depth.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        align_timepoint {str} --- Trial timepoint to align to: a timepoints CSV column
            (e.g. first_grasp_start) or a meta_session 'ttl_to_*' column.
        before, after {float} --- Window (s) around the alignment timepoint.
        bin_width, filter_sigma {float} --- Time binning and Gaussian smoothing (s).
        n_depth_bins {int} --- Number of equal-width bins over the span of unit depths.
        stack_by_depth {bool} --- Instead of binning by the precise depth, draw one row per
            unit, stacked in depth order (each row normalized to that unit's own peak);
            n_depth_bins is then unused.
        only_good {bool} --- Use only meta_neural.json 'good_neurons' (else all units).
        min_rate {float} --- Optional activity threshold (Hz): drop units whose mean
            trial-averaged rate is at or below this. None -> meta_neural.json 'min_rate'
            if present, else no filter.
        drift_correct {bool} --- Subtract each unit's linear session drift (default True).
        use_threshold_crossings {bool} --- Read the threshold-crossing product instead
            of the sorted neural.nwb.
        save {bool} --- Save the figure (default True) into
            <processed_server>/<session>/prehension_plots/figure_depth_activity_<align>.png
            (figure_depth_activity_<align>_stacked.png with stack_by_depth).
        save_dir {str} --- Explicit output folder overriding the default. None -> default.

    Returns (fig, depth_edges, activity, counts): the row-bin edges (um; the unit ranks
    0..n_units with stack_by_depth), the normalized activity (n_rows, n_time) and the
    number of units per row (all ones with stack_by_depth).
    """
    session_data = load_session_trials(
        server, processed_server, session, align_key=align_timepoint, only_good=only_good,
        use_threshold_crossings=use_threshold_crossings)
    min_rate = resolve_meta_arg(min_rate, session_data['meta_neural'], 'min_rate', None)

    rates, bin_centers = session_unit_rates(
        session_data, before=before, after=after, bin_width=bin_width,
        filter_sigma=filter_sigma, drift_correct=drift_correct)
    depths = _unit_depths(session_data['nwb_path'], session_data['neuron_labels'])
    rates, depths = _filter_by_rate(rates, depths, min_rate)

    n_no_depth = int(np.sum(~np.isfinite(depths)))
    if n_no_depth == len(depths):
        raise ValueError('Session {}: the NWB carries no unit depths; re-export it to add '
                         'them.'.format(session))
    if n_no_depth:
        ws('Leaving out {} unit(s) without a depth.'.format(n_no_depth))

    n_units = len(depths) - n_no_depth
    row_depths = None
    if stack_by_depth:
        depth_edges, binned, row_depths = _stack_by_depth(rates, depths)
        counts = np.ones(n_units, dtype=int)
        layout = 'units stacked in depth order'
        rs('{} unit(s) stacked in depth order.'.format(n_units))
    else:
        depth_edges, binned, counts = stats.binned_mean_rows(rates, depths, n_depth_bins)
        layout = '{} depth bins'.format(n_depth_bins)
        rs('{} unit(s) in {} / {} occupied depth bins of {:.1f} um.'.format(
            n_units, int(np.count_nonzero(counts)), n_depth_bins,
            depth_edges[1] - depth_edges[0]))
    activity = stats.normalize_rows_to_max(binned)

    title = '{}{}: activity by depth, aligned to {}\n({} units, {} trials, {})'.format(
        session, ' (threshold crossings)' if use_threshold_crossings else '',
        align_timepoint, n_units, len(session_data['trials']), layout)
    save_dir = resolve_session_save_dir(processed_server, session, save, save_dir)
    fig = plot_depth_activity(bin_centers, depth_edges, activity, align_timepoint, title,
                              save_dir, row_depths=row_depths)
    return fig, depth_edges, activity, counts
