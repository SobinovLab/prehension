#!python3
# -*- coding: utf-8 -*-
"""
Wiener / Kalman filter decoding of the total grasp force from neural activity, per session.

For a session, assembles a continuous per-trial design -- the binned, smoothed firing rate of
every unit as the neural observation, and the total grasp force (summed over all sensels and both
filtered pressure sensors, forces.load_summed_force_trace on the filtered pre_ps files) resampled
onto the same bins as the state -- and cross-validates a linear force decoder holding out whole
trials (5-fold by default).  The decoder is a Wiener / linear-FIR filter by default
(tools.decoding.WienerFilterDecoder) or a Kalman filter (tools.decoding.KalmanFilterDecoder,
Wu et al. 2006).  Each trial is held out exactly once and scored with a per-trial pseudo-R2
(coefficient of determination of the decoded vs actual force).  With --n_pcs the neural
observations are first reduced to their top principal components (fit per fold on the training
trials only).

Two figures are drawn per session and saved to <session>/prehension_plots/:

  * the held-out actual-vs-decoded force traces, one small panel per held-out trial;
  * per-trial decoding-quality metrics over the held-out trials -- pseudo-R2 (coefficient of
    determination, clipped at -1), NRMSE (root mean squared error normalized by the session's
    force range) and the Pearson correlation r -- each as two swarms, the real decoding beside a
    neural<->force time-alignment shuffle (a chance null, tools.decoding.shuffle_trial_alignment).

Decoding is per session because the neural observation dimension (the sorted units) is
session-specific.  The trial assembly reuses analysis.decoding.pool_decoding_trials with the
grasp-force target; by default only the active-touch period (active_grasp: first grasp -> release)
is decoded, matching the encoding force period, but --period exposes the other sub-periods.

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
from ..tools import plotting
from ..tools.logs import rs, ws
from ..tools.decoding import (
    make_decoder, decode_cv_per_trial, shuffle_trial_alignment, coefficient_of_determination,
    rmse, pearson_r, DECODE_METHODS)
from .decoding import pool_decoding_trials
from .encoding import PERIOD_ACTIVE_GRASP, _period_suffix

# The decoded target is the scalar summed grasp force (forces.load_summed_force_trace), assembled
# by analysis.decoding.pool_decoding_trials under this target key.
FORCE_TARGET = 'grasp_force'

DEFAULT_METHOD = 'wiener'                 # Wiener / linear-FIR filter (see tools.decoding)
DEFAULT_PERIOD = PERIOD_ACTIVE_GRASP      # active-touch period: first grasp -> release
DEFAULT_N_PCS = 25                        # top PCs used when --n_pcs is passed without a count
DEFAULT_WIENER_HISTORY_S = 0.25           # Wiener neural history (s) -> preceding bins per session
MAX_TRACE_TRIALS = 49                     # cap on the number of trace panels in one figure
PR2_FLOOR = -1.0                          # per-trial pseudo-R2 clipped to this floor for display
NRMSE_CEIL = 2.0                          # NRMSE ceiling; higher values grouped here as outliers

METHOD_LABELS = {'wiener': 'Wiener filter', 'kalman': 'Kalman filter'}


def _method_label(method):
    """Human-readable decoder name for figure titles."""
    return METHOD_LABELS.get(method, method)


def _pca_label(n_pcs):
    """Title fragment noting the PCA reduction, or '' when decoding from the raw units."""
    return ', top {} PCs'.format(n_pcs) if n_pcs else ''


def _fig_suffix(method, period, n_pcs, use_threshold_crossings):
    """Filename suffix encoding the decoder, period, PCA reduction and neural source.

    The period reuses encoding._period_suffix (PERIOD_ALL -> ''); the '_tx' source tag stays last,
    matching the encoding / decoding outputs.
    """
    suffix = '_{}{}'.format(method, _period_suffix(period))
    if n_pcs:
        suffix += '_pca{}'.format(n_pcs)
    if use_threshold_crossings:
        suffix += '_tx'
    return suffix


def _per_trial_metrics(trues, preds, force_range):
    """Per-trial decoding-quality metrics over held-out (true, pred) trials.

    Returns a dict with, per trial: 'r2' the pseudo-R2 (coefficient of determination), 'nrmse' the
    root mean squared error normalized by `force_range` (the session-wide force max - min, a single
    scalar so trials share one scale), and 'r' the Pearson correlation between decoded and actual
    force.  A non-positive / non-finite `force_range` yields NaN NRMSE.
    """
    denom = force_range if force_range and force_range > 0 else np.nan
    return {
        'r2': np.array([coefficient_of_determination(t, p) for t, p in zip(trues, preds)]),
        'nrmse': np.array([rmse(t, p) / denom for t, p in zip(trues, preds)]),
        'r': np.array([pearson_r(t, p) for t, p in zip(trues, preds)]),
    }


def _select_trace_trials(order, max_trials, session):
    """Indices into the held-out lists to plot, in trial order, capped at `max_trials`.

    Every trial is held out exactly once, so this is the full set sorted by trial index; when
    there are more than `max_trials` an evenly spaced subset is kept (and a warning logged) so
    one figure stays readable.
    """
    sort = np.argsort(np.asarray(order))
    if sort.size > max_trials:
        pick = np.unique(np.linspace(0, sort.size - 1, max_trials).round().astype(int))
        ws('Session {}: {} held-out trials; plotting {} of them in the trace figure.'.format(
            session, sort.size, pick.size))
        sort = sort[pick]
    return sort


def _draw_trace_figure(session, trues, preds, order, metrics, bin_width, method, period, n_pcs,
                       use_threshold_crossings, save_dir, max_trials, save):
    """One panel per held-out trial: actual vs decoded grasp-force trace, titled with its metrics.

    `metrics` is the per-trial dict from _per_trial_metrics ('r2', 'nrmse', 'r'); each subplot
    title reports that trial's pseudo-R2, NRMSE and Pearson r.
    """
    sort = _select_trace_trials(order, max_trials, session)
    xn, yn = plotting.xy_numsubplots(sort.size)
    fig, axs = plt.subplots(nrows=yn, ncols=xn, squeeze=False,
                            figsize=(1.0 + 3.0 * xn, 1.0 + 2.5 * yn))
    axs = axs.flatten()
    for ax, k in zip(axs, sort):
        t = np.arange(trues[k].shape[0]) * bin_width
        ax.plot(t, np.asarray(trues[k]).ravel(), color='k', linewidth=1.2, label='actual')
        ax.plot(t, np.asarray(preds[k]).ravel(), color='tab:red', linewidth=1.0,
                linestyle='--', label='decoded')
        ax.set_title('trial {}\nR$^2$={:.2f}, NRMSE={:.2f}, r={:.2f}'.format(
            int(order[k]) + 1, metrics['r2'][k], metrics['nrmse'][k], metrics['r'][k]),
            fontsize=7)
        ax.tick_params(labelsize=6)
    axs[0].legend(loc='upper right', fontsize=6)
    for ax in axs[sort.size:]:
        ax.axis('off')
    fig.supxlabel('time in {} period (s)'.format(period), fontsize=9)
    fig.supylabel('summed grasp force (N)', fontsize=9)
    fig.suptitle('Grasp-force decoding traces (held-out trials) -- {} [{}], {}{}{}'.format(
        session, period, _method_label(method), _pca_label(n_pcs),
        ' (threshold crossings)' if use_threshold_crossings else ''))
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.96))
    if save:
        plotting.savefig(save_dir, 'figure_decode_force_traces' + _fig_suffix(
            method, period, n_pcs, use_threshold_crossings), fig=fig)
        rs('Saved grasp-force decoding traces for session {}.'.format(session))
    return fig


# Per-trial metric panels drawn in the metrics figure: (key, y-axis label, display floor, display
# ceiling, whether to draw a 0 reference line).  pseudo-R2 is clipped up to PR2_FLOOR and NRMSE
# down to NRMSE_CEIL (its higher outliers grouped at the cap); the Pearson r ([-1, 1]) is as-is.
# The NRMSE panel also gets a right-hand axis in Newtons (RMSE = NRMSE * session force range).
_METRIC_PANELS = (
    ('r2', 'per-trial pseudo-R$^2$ (clipped at {:.0f})'.format(PR2_FLOOR), PR2_FLOOR, None, True),
    ('nrmse', 'per-trial NRMSE (by force range, capped at {:.0f})'.format(NRMSE_CEIL),
     None, NRMSE_CEIL, False),
    ('r', 'per-trial Pearson r', None, None, True),
)

_SWARM_GROUPS = ('held-out', 'time-shuffled')
_SWARM_POINT_COLORS = ('tab:blue', '0.6')
_SWARM_MEDIAN_COLORS = ('tab:blue', '0.3')


def _swarm_panel(ax, sns, real, shuf, floor, ceiling, zero_line):
    """Draw one metric panel: held-out vs time-shuffled swarms with per-group median markers.

    Non-finite values are dropped; when `floor` / `ceiling` are given the values are clipped into
    that range (so extreme outliers -- pseudo-R2 << 0, or a large NRMSE -- pile up at the bound
    instead of dominating the axis) and a dashed line marks the active bound(s).  `zero_line` adds
    a dotted reference at 0.  Point counts go in the x tick labels.
    """
    real = np.asarray(real, dtype=float)
    shuf = np.asarray(shuf, dtype=float)
    real = real[np.isfinite(real)]
    shuf = shuf[np.isfinite(shuf)]
    if floor is not None or ceiling is not None:
        real = np.clip(real, floor, ceiling)
        shuf = np.clip(shuf, floor, ceiling)

    for i, vals in enumerate((real, shuf)):
        if vals.size:
            sns.swarmplot(x=[_SWARM_GROUPS[i]] * vals.size, y=vals, order=_SWARM_GROUPS, ax=ax,
                          color=_SWARM_POINT_COLORS[i], size=3, alpha=0.7)
            med = float(np.median(vals))
            ax.plot([i - 0.4, i + 0.4], [med, med], color=_SWARM_MEDIAN_COLORS[i], linewidth=2.5)
            ax.annotate('med {:.2f}'.format(med), (i, med), textcoords='offset points',
                        xytext=(6, 2), fontsize=8, color=_SWARM_MEDIAN_COLORS[i])
    if zero_line:
        ax.axhline(0.0, color='0.6', linewidth=0.8, linestyle=':')
    if floor is not None:
        ax.axhline(floor, color='0.8', linewidth=0.8, linestyle='--')
        ax.set_ylim(bottom=floor - 0.1)
    if ceiling is not None:
        ax.axhline(ceiling, color='0.8', linewidth=0.8, linestyle='--')
        ax.set_ylim(top=ceiling + 0.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['{}\n(n={})'.format(_SWARM_GROUPS[0], real.size),
                        '{}\n(n={})'.format(_SWARM_GROUPS[1], shuf.size)])
    ax.set_xlim(-0.6, 1.6)


def _draw_metrics_figure(session, real_metrics, shuf_metrics, force_range, method, period, n_pcs,
                         use_threshold_crossings, save_dir, save):
    """Per-trial metrics (pseudo-R2, NRMSE, Pearson r) as held-out vs time-shuffled swarms.

    One panel per metric (_METRIC_PANELS): the real held-out per-trial scores beside the
    time-alignment shuffle null, with per-group medians marked.  The pseudo-R2 is clipped at
    PR2_FLOOR and the NRMSE capped at NRMSE_CEIL for display; the NRMSE panel also carries a
    right-hand axis in Newtons (RMSE = NRMSE * `force_range`, the session force max - min).
    """
    import seaborn as sns

    fig, axs = plt.subplots(1, len(_METRIC_PANELS), figsize=(4.2 * len(_METRIC_PANELS), 5))
    axs = np.atleast_1d(axs)
    for ax, (key, ylabel, floor, ceiling, zero_line) in zip(axs, _METRIC_PANELS):
        _swarm_panel(ax, sns, real_metrics[key], shuf_metrics[key], floor, ceiling, zero_line)
        ax.set_ylabel(ylabel)
        # NRMSE = RMSE / force_range, so a linearly scaled twin axis shows the un-normalized RMSE
        # in Newtons (only when the session force range is a usable scale)
        if key == 'nrmse' and np.isfinite(force_range) and force_range > 0:
            secax = ax.secondary_yaxis('right', functions=(
                lambda y, r=force_range: y * r, lambda y, r=force_range: y / r))
            secax.set_ylabel('RMSE (N)')
    fig.suptitle('Grasp-force decoding per-trial metrics (held-out vs time-shuffled) -- '
                 '{} [{}], {}{}{}'.format(
                     session, period, _method_label(method), _pca_label(n_pcs),
                     ' (threshold crossings)' if use_threshold_crossings else ''))
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        plotting.savefig(save_dir, 'figure_decode_force_metrics' + _fig_suffix(
            method, period, n_pcs, use_threshold_crossings), fig=fig)
        rs('Saved grasp-force decoding metrics for session {}.'.format(session))
    return fig


def decode_force(server, processed_server, sessions, method=DEFAULT_METHOD, n_folds=5,
                 n_pcs=None, wiener_history=DEFAULT_WIENER_HISTORY_S, bin_width=None,
                 use_threshold_crossings=False, period=DEFAULT_PERIOD, drift_correct=True,
                 successful_only=True, seed=None, max_trace_trials=MAX_TRACE_TRIALS, save=True):
    """Cross-validated Wiener / Kalman decoding of the total grasp force per session, with figures.

    For every session (empty `sessions` -> all sessions under processed_server): assembles per-
    trial (neural rate, summed grasp force) time series (analysis.decoding.pool_decoding_trials,
    grasp-force target, restricted to `period`), then trial-wise k-fold cross-validates a force
    decoder holding out whole trials (tools.decoding.decode_cv_per_trial with a `method` decoder
    from make_decoder; `n_pcs`, when set, reduces the neural observations to their top principal
    components per fold).  Each held-out trial gets a pseudo-R2 (coefficient of determination of
    decoded vs actual force).  The identical pipeline is rerun on a neural<->force time-alignment
    shuffle (shuffle_trial_alignment) for a chance distribution.

    Each held-out trial also gets an NRMSE (root mean squared error normalized by the session-wide
    force range, max - min over every trial) and a Pearson correlation r.

    Draws and (unless save is False) saves two figures per session into
    <session>/prehension_plots/: the held-out actual-vs-decoded traces (one panel per trial, at
    most `max_trace_trials`) and a per-trial metrics figure -- pseudo-R2 (clipped at -1), NRMSE
    (capped at 2, with a twin RMSE-in-Newtons axis) and Pearson r, each as held-out vs
    time-shuffled swarms.  `successful_only` (default True) decodes only successful trials.
    `period` defaults to the active-touch window (active_grasp); `wiener_history` sets the
    Wiener filter's neural-history length in seconds, converted to preceding bins per session as
    round(wiener_history / bin_width) (ignored by the Kalman decoder); `seed` makes the fold split
    and the shuffle reproducible.  Returns the list of figures (two per decoded session).
    """
    if method not in DECODE_METHODS:
        raise ValueError('Unknown decode method {!r}; expected one of {}.'.format(
            method, DECODE_METHODS))

    found = sessions if sessions else meta_session.find_session_dirs(processed_server)

    figures = []
    for session in found:
        try:
            trials_obs, trials_state, unit_ids, _, bw, _ = pool_decoding_trials(
                server, processed_server, session, FORCE_TARGET, bin_width=bin_width,
                use_threshold_crossings=use_threshold_crossings, period=period,
                drift_correct=drift_correct, successful_only=successful_only)
        except Exception as e:  # noqa: BLE001
            ws('Skipping {} [{}]: {}'.format(session, period, e))
            continue

        # the Wiener history is given in seconds -> preceding bins at this session's bin width
        # (the Kalman decoder ignores taps); build the decoder factory per session accordingly
        wiener_taps = int(round(wiener_history / bw)) if bw else 0
        factory = make_decoder(method, wiener_taps=wiener_taps)
        method_desc = method if method != 'wiener' else 'wiener ({:.0f} ms / {} taps)'.format(
            wiener_history * 1e3, wiener_taps)

        trues, preds, order = decode_cv_per_trial(
            trials_state, trials_obs, factory, n_folds=n_folds, n_pcs=n_pcs, seed=seed)
        if not trues:
            ws('Skipping {} [{}]: too few trials to cross-validate.'.format(session, period))
            continue

        # NRMSE is normalized by the session-wide force range (max - min over every trial's actual
        # force), a single scale shared by all trials and by the shuffle null
        all_force = np.concatenate([np.asarray(s, dtype=float).ravel() for s in trials_state])
        force_range = float(all_force.max() - all_force.min()) if all_force.size else np.nan
        real_metrics = _per_trial_metrics(trues, preds, force_range)

        # matched time-alignment null: same folds / PCA, states with their within-trial alignment
        # to the neural rates shuffled (a distinct but reproducible shuffle seed when seed is set)
        shuffled_state = shuffle_trial_alignment(
            trials_state, seed=None if seed is None else seed + 1)
        s_trues, s_preds, _ = decode_cv_per_trial(
            shuffled_state, trials_obs, factory, n_folds=n_folds, n_pcs=n_pcs, seed=seed)
        shuf_metrics = _per_trial_metrics(s_trues, s_preds, force_range)

        rs('Decoded grasp force {} [{}] with {}{}{}: {} units -> {} held-out trials, median '
           'per-trial R2={:.3f} / NRMSE={:.3f} / r={:.3f} (time-shuffled R2={:.3f}).'.format(
               session, period, method_desc, _pca_label(n_pcs),
               ' (threshold crossings)' if use_threshold_crossings else '',
               len(unit_ids), len(trues), np.nanmedian(real_metrics['r2']),
               np.nanmedian(real_metrics['nrmse']), np.nanmedian(real_metrics['r']),
               np.nanmedian(shuf_metrics['r2'])))

        save_dir = os.path.join(processed_server, session, 'prehension_plots')
        figures.append(_draw_trace_figure(
            session, trues, preds, order, real_metrics, bw, method, period, n_pcs,
            use_threshold_crossings, save_dir, max_trace_trials, save))
        figures.append(_draw_metrics_figure(
            session, real_metrics, shuf_metrics, force_range, method, period, n_pcs,
            use_threshold_crossings, save_dir, save))
    return figures
