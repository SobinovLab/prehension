#!python3
# -*- coding: utf-8 -*-
"""
Neuron rate vs summed grasp force cross-correlation figure across sessions.

For every requested session, cross-correlates each neuron's firing rate with the
summed pressure-sensor force within the same trial, restricted to the continuous
active-grasp period (the stable-grasp threshold used in
matching.process_and_align_data).  The neuron's rate and the force live in the same
reference frame (seconds since the trial's TTL pulse), so no alignment timepoint is
needed.  The lag is explored from -pre_lag to +post_lag seconds; the neuron window is
widened by the lag range so every lag is evaluated over the full active span (the
lags adjust the correlation edges, not the span).  Per session the per-trial
correlation curves are averaged per neuron.  Curves are shown and summarised as r^2,
so a neuron's peak lag is the lag of its strongest correlation regardless of sign.
Sessions are drawn on separate subplots titled with the session, region and burr hole
(from meta_neural.json).

Three figures are produced:
  * plot_cross_correlation(...)    -> the per-neuron r^2 curves, one subplot per
    session, with the median peak-r^2 lag marked,
  * plot_peak_lag_histogram(...)   -> a density histogram of the per-neuron peak-r^2
    lag, one subplot per session, and
  * plot_depth_stacked_cross_correlation(...) -> each neuron's peak-normalized
    cross-correlation curve stacked at its depth along the probe, one subplot per
    session, oriented with the electrode tip at the bottom.

Stages, as separate functions:
  * pool_cross_correlations(...)  -> per-session per-neuron mean cross-correlations (r)
    and each neuron's depth along the probe,
  * plot_cross_correlation(...) / plot_peak_lag_histogram(...) /
    plot_depth_stacked_cross_correlation(...) -> the three figures.
figure_cross_correlation(...) runs them all.

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

from ..tools import plotting
from ..tools.logs import rs, ws
from ..tools.cmd_args import sessions_name_stub
from ..neural_processing.common.spikes import BIN_WIDTH, FILTER_SIGMA
from ..neural_processing.common.population import MIN_RATE_HZ
from .common.pooling import (
    pool_cross_correlations, PRE_LAG, POST_LAG, FORCE_ACTIVE_FRACTION)
from .common.traces import resolve_pooled_save_dir, figure_filename


def _peak_lags(xcorr_r2, lag_times):
    """Per-neuron peak lag: the lag of each neuron's maximum r^2.

    Using r^2 makes the peak the lag of strongest correlation regardless of sign.
    Neurons whose curve is all-NaN (no usable trial overlap) are skipped.  Returns a
    1-D array of the surviving neurons' peak lags (empty when none are defined).
    """
    peaks = [lag_times[np.nanargmax(row)] for row in xcorr_r2 if not np.all(np.isnan(row))]
    return np.asarray(peaks, dtype=float)


def _median_peak_lag(xcorr_r2, lag_times):
    """Median across neurons of the per-neuron peak lag (see _peak_lags).

    Returns np.nan when no neuron has a defined peak.
    """
    peaks = _peak_lags(xcorr_r2, lag_times)
    return float(np.median(peaks)) if peaks.size else np.nan


# ---------------------------------------------------------------------------
# Plot the per-session cross-correlations
# ---------------------------------------------------------------------------
def plot_cross_correlation(sessions_results, lag_times, pre_lag, post_lag, name, save_dir):
    """Plot per-neuron neuron-force cross-correlations (r^2), one subplot per session.

    Each subplot draws every neuron's mean cross-correlation curve as r^2 (thin), the
    across-neuron mean r^2 (thick), a dotted line at zero lag, and an annotated dashed
    line at the median peak-r^2 lag across the session's neurons.  The subplot title
    carries the session, its region and burr hole (from meta_neural.json) and the
    neuron count.  Saves <name>_curves.png into save_dir and returns the figure.
    """
    xn, yn = plotting.xy_numsubplots(len(sessions_results))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    for i_s, ax in enumerate(axs):
        if i_s >= len(sessions_results):
            ax.axis('off')
            continue
        res = sessions_results[i_s]
        r2 = res['xcorr'] ** 2
        for row in r2:
            ax.plot(lag_times, row, color='0.6', linewidth=0.7, alpha=0.6)
        with np.errstate(invalid='ignore'):
            mean_curve = np.nanmean(r2, axis=0)
        ax.plot(lag_times, mean_curve, color='k', linewidth=1.8, label='mean')
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle=':')

        median_peak = _median_peak_lag(r2, lag_times)
        if np.isfinite(median_peak):
            plotting.annotated_vbar(
                ax, median_peak, 'median peak {:+.0f} ms'.format(median_peak * 1e3),
                color='tab:red', linestyle='--')

        region = res['region'] or 'n/a'
        burr_hole = res['burr_hole'] or 'n/a'
        ax.set_title('{}\n{}, burr hole {} ({} neurons, {} trials)'.format(
            res['session'], region, burr_hole, r2.shape[0], res['n_trials']), fontsize=8)
        ax.set_xlim(-pre_lag, post_lag)
        ax.set_ylim(bottom=0.0)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Neuron lag re force, s (>0: neuron lags force)', fontsize=7)
        ax.set_ylabel('Cross-correlation r$^2$', fontsize=7)

    fig.suptitle('Neuron rate vs summed grasp-force cross-correlation (r$^2$) over the '
                 'active-grasp period')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, figure_filename(name, 'curves'))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


# ---------------------------------------------------------------------------
# Plot the per-session peak-lag distribution
# ---------------------------------------------------------------------------
def plot_peak_lag_histogram(sessions_results, lag_times, pre_lag, post_lag, name, save_dir):
    """Density histogram of each session's per-neuron peak-r^2 lag, one subplot per session.

    For every session, each neuron's peak lag (the lag of its maximum r^2, see
    _peak_lags) is collected and drawn as a probability-density histogram binned on the
    lag grid, with a dotted line at zero lag and an annotated dashed line at the median
    peak lag.  The subplot title carries the session, its region and burr hole (from
    meta_neural.json) and the neuron count.  Saves <name>_peak_lag_hist.png into
    save_dir and returns the figure.
    """
    # histogram bins aligned to the lag grid (peak lags fall on grid points)
    dt = float(np.median(np.diff(lag_times))) if lag_times.size > 1 else (pre_lag + post_lag) or 1.0
    edges = np.concatenate([lag_times - dt / 2, [lag_times[-1] + dt / 2]])

    xn, yn = plotting.xy_numsubplots(len(sessions_results))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    for i_s, ax in enumerate(axs):
        if i_s >= len(sessions_results):
            ax.axis('off')
            continue
        res = sessions_results[i_s]
        r2 = res['xcorr'] ** 2
        peaks = _peak_lags(r2, lag_times)
        if peaks.size:
            ax.hist(peaks, bins=edges, density=True, color='tab:blue',
                    alpha=0.7, edgecolor='white', linewidth=0.3)
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle=':')

        median_peak = float(np.median(peaks)) if peaks.size else np.nan
        if np.isfinite(median_peak):
            plotting.annotated_vbar(
                ax, median_peak, 'median peak {:+.0f} ms'.format(median_peak * 1e3),
                color='tab:red', linestyle='--')

        region = res['region'] or 'n/a'
        burr_hole = res['burr_hole'] or 'n/a'
        ax.set_title('{}\n{}, burr hole {} ({} neurons, {} trials)'.format(
            res['session'], region, burr_hole, r2.shape[0], res['n_trials']), fontsize=8)
        ax.set_xlim(-pre_lag, post_lag)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Peak lag re force, s (>0: neuron lags force)', fontsize=7)
        ax.set_ylabel('Density', fontsize=7)

    fig.suptitle('Distribution of neuron peak-r$^2$ lag over the active-grasp period')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, figure_filename(name, 'peak_lag_hist'))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


# ---------------------------------------------------------------------------
# Plot the per-session depth-stacked cross-correlations
# ---------------------------------------------------------------------------
def plot_depth_stacked_cross_correlation(sessions_results, lag_times, pre_lag, post_lag,
                                         name, save_dir, gain_fraction=0.9):
    """Depth-stacked, peak-normalized neuron-force cross-correlation curves per session.

    One subplot per session.  Each neuron's mean cross-correlation curve is squared to
    r^2 and normalized to its own peak (r^2 max -> 1), then drawn at a vertical position
    set by the neuron's depth along the probe (um), so the panel reads as a depth map of
    the correlation shape.  A red tick marks each curve's peak r^2, placed on the curve
    itself (at the peak lag, height depth + gain).  Curves whose depth is unknown (NaN --
    older NWB products without the depth column) are skipped.  The per-curve gain is
    ``gain_fraction`` of the median spacing between adjacent neuron depths, so
    neighbouring traces stay legible.  The subplot title carries the session, its region
    and burr hole (from
    meta_neural.json) and the count of neurons with a depth.  Saves <name>_depth_stack.png
    into save_dir and returns the figure.
    """
    xn, yn = plotting.xy_numsubplots(len(sessions_results))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    for i_s, ax in enumerate(axs):
        if i_s >= len(sessions_results):
            ax.axis('off')
            continue
        res = sessions_results[i_s]
        xcorr = res['xcorr']
        depths = np.asarray(res.get('depths'), dtype=float)

        finite = np.isfinite(depths)
        if not finite.any():
            ax.text(0.5, 0.5, 'no unit depths', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='0.4')
            ax.set_title(res['session'], fontsize=8)
            continue

        # Vertical gain: a peak-normalized curve (|max| -> 1) spans gain_fraction of the
        # median spacing between adjacent neuron depths so neighbouring traces just about
        # touch and stay readable regardless of how many neurons the probe recorded.
        sorted_depths = np.sort(depths[finite])
        spacing = np.median(np.diff(sorted_depths)) if sorted_depths.size > 1 else 1.0
        if not np.isfinite(spacing) or spacing <= 0:
            spacing = 1.0
        gain = gain_fraction * spacing

        n_drawn = 0
        for row, depth in zip(xcorr, depths):
            if not np.isfinite(depth) or np.all(np.isnan(row)):
                continue
            r2 = row ** 2
            peak = np.nanmax(r2)
            norm = r2 / peak if peak > 0 else np.zeros_like(r2)
            curve = depth + gain * norm
            ax.plot(lag_times, curve, color='0.3', linewidth=0.7, alpha=0.8)
            # Red tick at the peak r^2 of this curve, sitting on the curve itself
            # (its height is the curve value at the peak lag = depth + gain).
            i_peak = np.nanargmax(r2)
            ax.plot(lag_times[i_peak], curve[i_peak], marker='|', color='tab:red',
                    markersize=5, markeredgewidth=1.0)
            n_drawn += 1
        ax.axvline(0.0, color='k', linewidth=0.8, linestyle=':')

        # Orientation: the stored depth is the unit-location y-coordinate along the
        # probe, which increases from the electrode tip upward (see export_nwb /
        # read_nwb_unit_depths).  Matplotlib's y-axis increases upward by default, so
        # plotting depth directly already puts the electrode TIP AT THE BOTTOM.  Set the
        # limits ascending (low at bottom) explicitly and never invert the axis, so the
        # tip stays at the bottom no matter what order the depths arrive in.
        lo = float(np.nanmin(depths[finite])) - gain
        hi = float(np.nanmax(depths[finite])) + gain
        ax.set_ylim(lo, hi)   # ascending: tip (smallest probe y) at the bottom

        region = res['region'] or 'n/a'
        burr_hole = res['burr_hole'] or 'n/a'
        ax.set_title('{}\n{}, burr hole {} ({} neurons w/ depth, {} trials)'.format(
            res['session'], region, burr_hole, n_drawn, res['n_trials']), fontsize=8)
        ax.set_xlim(-pre_lag, post_lag)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('Neuron lag re force, s (>0: neuron lags force)', fontsize=7)
        ax.set_ylabel('Depth along probe, um (tip at bottom)', fontsize=7)

    fig.suptitle('Depth-stacked peak-normalized neuron rate vs summed grasp-force '
                 'cross-correlation (r$^2$; tip at bottom)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, figure_filename(name, 'depth_stack'))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def figure_cross_correlation(server, processed_server, sessions, bin_width=BIN_WIDTH,
                             filter_sigma=FILTER_SIGMA, pre_lag=PRE_LAG, post_lag=POST_LAG,
                             force_fraction=FORCE_ACTIVE_FRACTION, only_good=False,
                             min_rate=MIN_RATE_HZ, processes=1, name=None,
                             save=True, save_dir=None):
    """Cross-correlate each neuron's rate with the summed grasp force, per session.

    Runs pool_cross_correlations over `sessions` (empty -> all sessions on the
    server; per-session recording/skip_ttl/good_neurons come from each session's
    meta_neural.json) and plots one subplot per session.  `pre_lag` / `post_lag` are
    the cross-correlation lags explored before/after zero (seconds, default 0.5);
    `force_fraction` is the active-grasp threshold as a fraction of each trial's peak
    force; `min_rate` drops neurons whose mean rate over the active periods is at or
    below this (Hz), defaulting to the same MIN_RATE_HZ activity floor as the dPCA figure.
    `processes` sets the size of the per-session neuron process pool (one neuron per
    process; serial when <= 1).  Three figures are produced -- the per-neuron r^2
    curves, the per-session peak-lag density histogram, and the depth-stacked
    peak-normalized cross-correlation curves (each neuron placed at its probe depth,
    tip at the bottom) -- and saved by default (save=True) into
    <processed_server>/pooled_figures/figure_cross_correlation, named after `name`
    (the --sessions string; defaults to a stub built from `sessions`) with the
    '_curves' / '_peak_lag_hist' / '_depth_stack' suffixes; pass save=False to disable
    or save_dir to override the folder.  Returns
    (curves_fig, peak_lag_hist_fig, depth_stack_fig).
    """
    save_dir = resolve_pooled_save_dir(
        processed_server, 'figure_cross_correlation', save, save_dir)
    name = name or sessions_name_stub(sessions)
    sessions_results, lag_times = pool_cross_correlations(
        server, processed_server, sessions, bin_width=bin_width, filter_sigma=filter_sigma,
        pre_lag=pre_lag, post_lag=post_lag, force_fraction=force_fraction,
        only_good=only_good, min_rate=min_rate, processes=processes)
    if not sessions_results:
        raise ValueError(
            'No sessions with usable neural + force data among {}.'.format(sessions))
    rs('Plotting neuron-force cross-correlations for {} session(s).'.format(
        len(sessions_results)))
    curves_fig = plot_cross_correlation(
        sessions_results, lag_times, pre_lag, post_lag, name, save_dir)
    hist_fig = plot_peak_lag_histogram(
        sessions_results, lag_times, pre_lag, post_lag, name, save_dir)
    depth_stack_fig = plot_depth_stacked_cross_correlation(
        sessions_results, lag_times, pre_lag, post_lag, name, save_dir)
    return curves_fig, hist_fig, depth_stack_fig
