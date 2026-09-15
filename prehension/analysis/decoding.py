#!python3
# -*- coding: utf-8 -*-
"""
Kalman filter decoding of behaviour from neural activity through time, per session.

For a session, assembles a continuous per-trial design -- the binned, smoothed firing rate of
every unit as the observation, and a behavioural target (joint angles, joint torques, or the
summed grasp force) resampled onto the same bins as the latent state -- and fits a
cross-validated Kalman filter decoder (tools.decoding.KalmanFilterDecoder, Wu et al. 2006) that
estimates the target through time from the neural rates.  Whole trials are held out (trial-wise
k-fold), and the held-out decoded state is scored per dimension with the coefficient of
determination (R2) and the Pearson correlation.

The performance is reported as a scatter figure per session: one panel per target, a
decoded-vs-actual scatter of the held-out samples (z-scored per dimension so multi-dimensional
targets pool comparably) with the identity line and the median per-dimension R2 / correlation.

The joint-angle and torque targets use the same right-arm independent joint DOFs as the encoding
models (analysis.encoding._filter_ra_dofs), and each target is decoded on the same trial
sub-period the encoding models use for that signal.  Decoding is per session because the neural
observation dimension (the sorted units) is session-specific.  Behaviour + neural assembly
mirrors analysis.encoding._pool_encoding_trials with the roles swapped.

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
import scipy.ndimage
import matplotlib.pyplot as plt

from .. import meta_session
from ..tools import plotting, forces
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..tools.decoding import kalman_decode_cv
from ..neural_processing.common.spikes import (
    FILTER_SIGMA, read_nwb_spikes_and_ttl, get_trial_data_spike, fit_session_drift, drift_offset)
from ..neural_plotting.common.pooling import session_neural_context
from ..neural_plotting.common.behaviour import load_timepoints_into_msession
from .encoding import (
    _load_predictor, _component_exists, _trial_period_window, DEFAULT_PERIODS, PERIOD_ALL,
    PERIOD_ACTIVE_GRASP)

# Behavioural targets the decoder estimates.  joint_angles / torques are the multi-DOF encoding
# signals (joint_angles restricted to the right-arm independent DOFs, as in encoding);
# grasp_force is the scalar summed pressure-sensor force (forces.load_summed_force_trace).
DECODE_TARGETS = ('joint_angles', 'torques', 'grasp_force')
TARGET_LABELS = {
    'joint_angles': 'joint angles',
    'torques': 'joint torques',
    'grasp_force': 'grasp force',
}

MAX_SCATTER_POINTS = 5000   # subsample the decoded-vs-actual scatter to at most this many points

# Each target is decoded on the same trial sub-period the encoding models use for that signal
# (analysis.encoding.DEFAULT_PERIODS): kinematics / torques over the movement; grasp_force is a
# force, so it follows the encoding force period (active_grasp).  Used when no period is requested.
DECODE_DEFAULT_PERIODS = {
    'joint_angles': DEFAULT_PERIODS['joint_angles'],
    'torques': DEFAULT_PERIODS['torques'],
    'grasp_force': PERIOD_ACTIVE_GRASP,
}


def _target_period(target, period):
    """Effective sub-period for a decode target: the requested `period`, or -- when it is falsy --
    the target's encoding default (DECODE_DEFAULT_PERIODS; PERIOD_ALL if it has none)."""
    return period or DECODE_DEFAULT_PERIODS.get(target, PERIOD_ALL)


def _target_exists(trial, target):
    """Whether a trial has the file(s) needed for a decode target."""
    if target == 'grasp_force':
        return trial.do_pre_ps_files_exist()
    return _component_exists(trial, target)


def _load_decode_target(trial, target):
    """Load one trial's decode target as (times, channel_names, values (n_channels, n_times)).

    joint_angles / torques reuse the encoding loader (joint_angles restricted to the right-arm
    independent DOFs); grasp_force is the summed pressure-sensor force as a single channel.
    Times are seconds since the trial's TTL pulse (the same frame as the zeroed spikes).
    """
    if target == 'grasp_force':
        times, force = forces.load_summed_force_trace(
            list(trial.get_pre_ps_filenames().values()))
        return (np.asarray(times, dtype=float), ['grasp_force'],
                np.asarray(force, dtype=float)[np.newaxis, :])
    return _load_predictor(trial, target)


def pool_decoding_trials(server, processed_server, session, target, bin_width=None,
                         filter_sigma=FILTER_SIGMA, use_threshold_crossings=False,
                         period=PERIOD_ALL, drift_correct=True):
    """Per-trial (neural rate, target signal) segments for a session, on a shared bin grid.

    Mirrors analysis.encoding._pool_encoding_trials but with the decoding roles: the neural
    firing rate is the observation and `target` (joint_angles / torques / grasp_force) is the
    latent state.  Reads the neural source, pairs TTL pulses to trials positionally (meta_neural
    skip_ttl / skip_ttl_last), bins + Gaussian-smooths each unit's rate (subtracting the linear
    session drift when drift_correct), and resamples the target channels onto the same bin
    centres (seconds-since-TTL frame); `period` optionally crops each trial to a sub-period.
    Returns (trials_obs, trials_state, unit_ids, target_names, bin_width, fps): per-trial lists of
    (n_bins_i, n_units) and (n_bins_i, n_target).
    """
    nwb_path, meta_neural, rserv, pserv = session_neural_context(
        server, processed_server, session, use_threshold_crossings)
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(nwb_path)
    skip_ttl = resolve_meta_arg(None, meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(None, meta_neural, 'skip_ttl_last', 0)

    mstruct, _, _, msession = meta_session.load_meta_information(rserv, pserv)
    if period != PERIOD_ALL:
        load_timepoints_into_msession(msession, mstruct)

    fps = float(mstruct.get('fps') or 0) or None
    if bin_width is None:
        if not fps:
            raise ValueError("bin_width not given and no 'fps' in meta_structure for {}.".format(
                session))
        bin_width = 1.0 / fps

    # positional pulse<->trial pairing (as in neural_plotting.common.pooling)
    if skip_ttl and skip_ttl > 0:
        events_time = events_time[skip_ttl:]
    elif skip_ttl and skip_ttl < 0:
        msession = msession[-skip_ttl:]
    if skip_ttl_last and skip_ttl_last > 0:
        events_time = events_time[:-skip_ttl_last]
    elif skip_ttl_last and skip_ttl_last < 0:
        msession = msession[:skip_ttl_last]
    if len(events_time) != len(msession):
        raise ValueError('{} TTL pulses vs {} trials (skip_ttl={}, skip_ttl_last={}).'.format(
            len(events_time), len(msession), skip_ttl, skip_ttl_last))

    # spikes per trial, zeroed to each trial's TTL start -> seconds since TTL
    session_spikes = get_trial_data_spike(spikes, events_time)
    n_units = len(unit_ids)
    for trial_spikes, ev in zip(session_spikes, events_time):
        for i_n in range(n_units):
            trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ev[0]
    slopes, t_ref = fit_session_drift(spikes, events_time) if drift_correct else (None, 0.0)

    freq = 1.0 / bin_width
    sigma_bins = filter_sigma / bin_width

    trials_obs, trials_state, names = [], [], None
    for trial, tspk, ev in zip(msession, session_spikes, events_time):
        if not trial.success or not _target_exists(trial, target):
            continue
        try:
            times_b, ch_names, values = _load_decode_target(trial, target)
        except Exception as e:  # noqa: BLE001
            ws('Session {} trial {}: could not read {} ({}); skipping.'.format(
                session, trial.trial_number, target, e))
            continue
        if times_b.size < 2:
            continue

        bins = np.arange(float(times_b[0]), float(times_b[-1]) + bin_width, bin_width)
        if bins.size < 3:
            continue
        centers = bins[:-1] + bin_width / 2

        rate = np.zeros((centers.size, n_units))
        for i_n in range(n_units):
            counts, _ = np.histogram(tspk[i_n], bins=bins)
            rate[:, i_n] = (scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)
                            - drift_offset(slopes, t_ref, i_n, float(ev[0])))
        state = np.column_stack([np.interp(centers, times_b, v) for v in values])
        names = ch_names if names is None else names

        # crop to the requested sub-period (whole trial for PERIOD_ALL); drop trials whose
        # window is undefined (missing timepoints) or falls outside the trial's bins
        if period != PERIOD_ALL:
            window = _trial_period_window(trial, period)
            if window is None:
                continue
            keep = (centers >= window[0]) & (centers <= window[1])
            if not np.any(keep):
                continue
            rate = rate[keep]
            state = state[keep]
        trials_obs.append(rate)
        trials_state.append(state)

    if not trials_obs:
        raise ValueError('No usable trials for target {} in session {}.'.format(target, session))
    return trials_obs, trials_state, list(unit_ids), names, bin_width, fps


def _per_dim_r2(true, pred):
    """Per-dimension coefficient of determination R2 (NaN for a constant true dimension)."""
    ss_res = np.sum((true - pred) ** 2, axis=0)
    ss_tot = np.sum((true - true.mean(axis=0)) ** 2, axis=0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, np.nan)


def _per_dim_cc(true, pred):
    """Per-dimension Pearson correlation between decoded and true (NaN when either is constant)."""
    cc = np.full(true.shape[1], np.nan)
    for j in range(true.shape[1]):
        t, p = true[:, j], pred[:, j]
        if t.std() > 0 and p.std() > 0:
            cc[j] = float(np.corrcoef(t, p)[0, 1])
    return cc


def _standardize_pair(true, pred):
    """Z-score decoded and true by the true mean/std per dimension (identity line at 45 deg)."""
    mean = true.mean(axis=0)
    std = true.std(axis=0)
    std[std == 0] = 1.0
    return (true - mean) / std, (pred - mean) / std


def _draw_decoding_figure(session, targets, results, save_dir, use_threshold_crossings,
                          max_points, seed, save):
    """One panel per target: decoded-vs-actual scatter of held-out samples with the identity line."""
    xn, yn = plotting.xy_numsubplots(len(targets))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, squeeze=False,
                            figsize=(1.0 + 4.2 * xn, 1.0 + 4.2 * yn))
    axs = axs.flatten()
    rng = np.random.RandomState(seed)
    for ax, target in zip(axs, targets):
        res = results.get(target)
        if res is None:
            ax.axis('off')
            ax.set_title('{}\n(no data)'.format(TARGET_LABELS[target]), fontsize=9)
            continue
        x = res['true_z'].ravel()
        y = res['pred_z'].ravel()
        if x.size > max_points:
            idx = rng.choice(x.size, max_points, replace=False)
            x, y = x[idx], y[idx]
        ax.scatter(x, y, s=6, color='k', alpha=0.2, edgecolor='none')
        lo = float(min(x.min(), y.min())) if x.size else -1.0
        hi = float(max(x.max(), y.max())) if x.size else 1.0
        ax.plot([lo, hi], [lo, hi], color='0.6', linewidth=0.8, linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('actual (z-scored)', fontsize=8)
        ax.set_ylabel('decoded (z-scored)', fontsize=8)
        ax.set_title('{} ({}d): median R$^2$={:.2f}, cc={:.2f}'.format(
            TARGET_LABELS[target], res['true_z'].shape[1],
            np.nanmedian(res['r2']), np.nanmedian(res['cc'])), fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axs[len(targets):]:
        ax.axis('off')
    fig.suptitle('Kalman decoding performance (held-out, decoded vs actual) -- {}{}'.format(
        session, ' (threshold crossings)' if use_threshold_crossings else ''))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        plotting.savefig(save_dir, 'figure_decoding{}'.format(
            '_tx' if use_threshold_crossings else ''), fig=fig)
        rs('Saved decoding figure for session {}.'.format(session))
    return fig


def decode_targets(server, processed_server, sessions, targets=DECODE_TARGETS, n_folds=5,
                   bin_width=None, use_threshold_crossings=False, period=None,
                   drift_correct=True, seed=None, max_scatter_points=MAX_SCATTER_POINTS, save=True):
    """Kalman-decode each target through time per session and plot the performance scatters.

    For every session (empty `sessions` -> all under processed_server) and target
    (joint_angles / torques / grasp_force), assembles per-trial (neural rate, target) time series
    (pool_decoding_trials), fits a Kalman filter decoder with trial-wise k-fold cross-validation
    (tools.decoding.kalman_decode_cv), and scores the held-out decoded state per dimension (R2 and
    Pearson correlation).  Draws one figure per session with a panel per target: a decoded-vs-
    actual scatter of the held-out samples (z-scored per dimension so multi-dimensional targets
    pool comparably) with the identity line and the median per-dimension R2 / correlation.
    `period` restricts each trial to a sub-period before decoding; None (default) uses each
    target's encoding period (DECODE_DEFAULT_PERIODS -- kinematics/torques over the movement,
    grasp force over the grasp).  Figures are saved to <session>/prehension_plots/ unless save is
    False.  Returns the list of figures.
    """
    found = sessions if sessions else meta_session.find_session_dirs(processed_server)
    tgs = [t for t in DECODE_TARGETS if t in targets]

    figures = []
    for session in found:
        results = {}
        for target in tgs:
            eff_period = _target_period(target, period)
            try:
                trials_obs, trials_state, unit_ids, names, bw, fps = pool_decoding_trials(
                    server, processed_server, session, target, bin_width=bin_width,
                    use_threshold_crossings=use_threshold_crossings, period=eff_period,
                    drift_correct=drift_correct)
            except Exception as e:  # noqa: BLE001
                ws('Skipping {} / {} [{}]: {}'.format(session, target, eff_period, e))
                results[target] = None
                continue
            true, pred = kalman_decode_cv(trials_state, trials_obs, n_folds=n_folds, seed=seed)
            if true.shape[0] == 0:
                ws('Skipping {} / {} [{}]: too few trials to cross-validate.'.format(
                    session, target, eff_period))
                results[target] = None
                continue
            true_z, pred_z = _standardize_pair(true, pred)
            results[target] = {
                'true_z': true_z, 'pred_z': pred_z, 'names': names,
                'r2': _per_dim_r2(true, pred), 'cc': _per_dim_cc(true, pred)}
            rs('Decoded {} / {} [{}]: {} units -> {} dims, {} held-out samples, '
               'median R2={:.3f}, median cc={:.3f}.'.format(
                   session, target, eff_period, len(unit_ids), true.shape[1], true.shape[0],
                   np.nanmedian(results[target]['r2']), np.nanmedian(results[target]['cc'])))

        if not any(results.values()):
            ws('Skipping session {}: no target decoded.'.format(session))
            continue
        save_dir = os.path.join(processed_server, session, 'prehension_plots')
        figures.append(_draw_decoding_figure(
            session, tgs, results, save_dir, use_threshold_crossings, max_scatter_points, seed,
            save))
    return figures
