#!python3
# -*- coding: utf-8 -*-
"""
Poisson GLM encoding models of individual neurons / channels, per session.

For a session, assembles a continuous design across the session's trials -- each unit's
binned, smoothed firing rate as the response, and a behavioural signal (joint angles,
joint velocity, digit or segment forces, torques) resampled onto the same bins as the
covariates -- and fits a cross-validated Poisson GLM per unit (tools.encoding), scoring
each with the deviance-based pseudo-R2 and its predictor-count-adjusted variant.  The
per-unit performance and the full fit specification are written, human-readably, to the
session's ``encoding/`` folder, one file per predictor (and per neural source).

Predictors are the base signals plus combined groups (PREDICTOR_GROUPS -- position+velocity,
position+torque and position+segment_forces -- whose channels are concatenated into one
design).  The per-DOF predictors (position, velocity, torques) use only the right-arm
independent joint DOFs (channel names starting ``ra_`` and not ending ``_d``); see
_filter_ra_dofs.

optimal_lag sweeps the input<->output lag (as in the manuscript), picks each unit's
pR2-maximizing lag, and saves the per-unit lags and their session mode; when that file is
present, encoding_models applies the session mode lag for that predictor.

Each fit can be restricted to a sub-period of every trial (PERIODS): the whole trial
('all'), the reach-grasp-retreat movement ('active_movement') or the grasp itself
('active_grasp'), with the window read per trial from timepoints.csv (create_timepoints).
The period is recorded in each output file and appended to its name ('all' stays
unsuffixed); when no period is requested each predictor uses its DEFAULT_PERIODS entry.

Bins default to the session's kinematic/video rate (1/mstruct['fps']); lags are anti-causal
(positive = the covariate follows the firing rate).

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

from .. import meta_session
from ..tools import io, filters
from ..tools.constants import DISTAL_DOFS, PROXIMAL_DOFS, ALL_DOFS
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..tools.stats import run_pca
from ..tools.encoding import fit_glms_over_units
from ..neural_processing.common.spikes import (
    FILTER_SIGMA, read_nwb_spikes_and_ttl, get_trial_data_spike, resolve_neuron_selection)
from ..neural_processing.import_kilosorted import resolve_neural_nwb_path
from ..neural_processing.config import meta_neural_path
from ..neural_plotting.common.behaviour import load_timepoints_into_msession, get_timepoint
from . import behavior_pooling

# Predictor signals the GLM supports.  joint_velocity is derived from the joint angles
# (tools.filters.joint_velocity); the rest load a per-trial signal CSV via SIGNAL_SPECS.
PREDICTORS = ('joint_angles', 'joint_velocity', 'digit_forces', 'segment_forces', 'torques')

# Combined-signal predictors: a group name -> the base predictors whose channels are
# concatenated into a single design and fit jointly (position == joint angles).
PREDICTOR_GROUPS = {
    'position_velocity': ('joint_angles', 'joint_velocity'),
    'position_torques': ('joint_angles', 'torques'),
    'position_segment_forces': ('joint_angles', 'segment_forces'),
}

# Everything selectable / fit by default: the base signals followed by the combined groups.
ALL_PREDICTORS = PREDICTORS + tuple(PREDICTOR_GROUPS)

# Per-DOF predictors restricted to the right-arm independent joint DOFs -- channel names starting
# 'ra_' and not ending '_d' -- dropping the model's thorax and object (ps_) columns and the
# dependent/coupled '_d' coordinates (see _filter_ra_dofs).  Positions, velocities and torques all
# live on the same joint DOFs, so all three are restricted the same way.
RA_DOF_PREDICTORS = ('joint_angles', 'joint_velocity', 'torques')

# Joint-group selection for the per-DOF predictors (positions, velocities, torques): each group
# is an explicit list of DOF names from tools.constants, applied identically to all three.
#   'hand'     -- the distal DOFs (wrist + thumb + fingers; constants.DISTAL_DOFS)
#   'proximal' -- the shoulder + elbow DOFs (constants.PROXIMAL_DOFS)
#   'all'      -- every independent right-arm DOF (constants.ALL_DOFS); i.e. the previous
#                 behaviour, still ignoring the dependent '_d' and thorax / object columns.
# The 'all' group keeps the historic ra_* (non-_d) pattern filter, so its output is unchanged and
# its files stay unsuffixed (_joint_group_suffix); 'hand' is the default (JOINT_GROUPS order aside).
JOINT_GROUPS = {'hand': DISTAL_DOFS, 'proximal': PROXIMAL_DOFS, 'all': ALL_DOFS}
DEFAULT_JOINT_GROUP = 'hand'

ENCODING_SUBDIR = 'encoding'

# lag sweep (seconds); the step defaults to the bin width (1/fps) so lags fall on bins.
LAG_MIN = -1.0
LAG_MAX = 1.0
LAG_MIN_RATE_HZ = 0.5   # skip the lag search for units below this mean firing rate (Hz)
MIN_ADJ_R2 = 0.025      # skip units whose adjusted pseudo-R2 falls below this (lag mode / scatter)

# Trial sub-periods a fit can be restricted to.  Every non-'all' period is bounded by a pair
# of timepoints.csv columns (seconds since the trial's TTL pulse -- the same frame as the
# binned rate and the covariates); 'all' spans the whole trial and is saved unsuffixed.  See
# create_timepoints / find_event_onsets for how the columns are computed.
PERIOD_ALL = 'all'
PERIOD_ACTIVE_MOVEMENT = 'active_movement'   # earliest limb-movement onset -> hand back at rest
PERIOD_ACTIVE_GRASP = 'active_grasp'         # first grasp (force on) -> release (force off)
PERIODS = (PERIOD_ALL, PERIOD_ACTIVE_MOVEMENT, PERIOD_ACTIVE_GRASP)

# active_movement starts at the earliest available onset (the reach begins with whichever
# joint moves first); these are the movement-onset columns, proximal to distal.
MOVEMENT_ONSET_KEYS = ('shoulder_onset', 'elbow_onset', 'wrist_onset', 'finger_onset')

# Per-predictor default period when none is requested on the command line: kinematics and
# torques are informative throughout the movement, forces only while the object is grasped.
DEFAULT_PERIODS = {
    'joint_angles': PERIOD_ACTIVE_MOVEMENT,
    'joint_velocity': PERIOD_ACTIVE_MOVEMENT,
    'torques': PERIOD_ACTIVE_MOVEMENT,
    'digit_forces': PERIOD_ACTIVE_GRASP,
    'segment_forces': PERIOD_ACTIVE_GRASP,
    # combined groups all default to the movement window: position paired with velocity, torque
    # or the segment forces is evaluated over the whole reach-grasp-retreat movement
    'position_velocity': PERIOD_ACTIVE_MOVEMENT,
    'position_torques': PERIOD_ACTIVE_MOVEMENT,
    'position_segment_forces': PERIOD_ACTIVE_MOVEMENT,
}


def encoding_dir(processed_server, session):
    """Path to a session's encoding/ output folder."""
    return os.path.join(processed_server, session, ENCODING_SUBDIR)


def _suffix(use_threshold_crossings):
    return '_tx' if use_threshold_crossings else ''


def _period_suffix(period):
    """Filename suffix identifying a trial sub-period; PERIOD_ALL -> '' (unsuffixed file)."""
    return '' if not period or period == PERIOD_ALL else '_{}'.format(period)


def _joint_group_suffix(joint_group):
    """Filename suffix identifying the joint group; the default 'all' group -> '' (unsuffixed).

    Keeping 'all' unsuffixed means the previous (all-DOF) outputs keep their names, and only the
    restricted groups ('hand', 'proximal') get a distinguishing suffix so groups never clobber.
    """
    return '' if not joint_group or joint_group == 'all' else '_{}'.format(joint_group)


def resolve_period(predictor, period):
    """Effective period for a predictor: the requested `period`, or -- when it is falsy --
    the predictor's default from DEFAULT_PERIODS (PERIOD_ALL if the predictor has none)."""
    return period or DEFAULT_PERIODS.get(predictor, PERIOD_ALL)


def encoding_json_path(processed_server, session, predictor, use_threshold_crossings=False,
                       period=PERIOD_ALL, joint_group='all'):
    """Path to the saved encoding performance for one predictor (neural source, period, group).

    The period and joint-group suffixes precede the '_tx' neural-source suffix so the '_tx' tag
    stays last (downstream readers detect the source with a trailing '_tx.json' check).
    """
    return os.path.join(encoding_dir(processed_server, session),
                        'encoding_{}{}{}{}.json'.format(
                            predictor, _period_suffix(period), _joint_group_suffix(joint_group),
                            _suffix(use_threshold_crossings)))


def lag_json_path(processed_server, session, predictor, use_threshold_crossings=False,
                  period=PERIOD_ALL, joint_group='all'):
    """Path to the saved optimal-lag analysis for one predictor (neural source, period, group)."""
    return os.path.join(encoding_dir(processed_server, session),
                        '{}_lag{}{}{}.json'.format(
                            predictor, _period_suffix(period), _joint_group_suffix(joint_group),
                            _suffix(use_threshold_crossings)))


def _load_meta_neural(processed_server, session):
    """meta_neural.json for a session, or {} when absent (Utah sessions may lack it)."""
    path = meta_neural_path(processed_server, session)
    return io.load_json(path) if os.path.exists(path) else {}


def _group_components(predictor):
    """Base predictor(s) that make up `predictor`: a group's members, or the predictor itself."""
    return PREDICTOR_GROUPS.get(predictor, (predictor,))


def _component_exists(trial, predictor):
    """Whether a trial has the file(s) needed for one base predictor."""
    if predictor == 'joint_velocity':
        return trial.does_post_kin_file_exist()
    _, exists = behavior_pooling.SIGNAL_SPECS[predictor]
    return getattr(trial, exists)()


def _predictor_exists(trial, predictor):
    """Whether a trial has the file(s) for `predictor` -- every base of a group must be present."""
    return all(_component_exists(trial, c) for c in _group_components(predictor))


def _filter_ra_dofs(predictor, names, values, joint_group='all'):
    """Restrict a per-DOF predictor to the right-arm independent joint DOFs (and a joint group).

    joint_angles / joint_velocity / torques carry the model's thorax and object (ps_) columns and
    the dependent (coupled) coordinates; the encoding uses only the independent right-arm joint
    coordinates -- channel names starting 'ra_' and not ending '_d'.  `joint_group` further
    restricts those to a group from JOINT_GROUPS ('hand' -> the distal DOFs, 'proximal' -> the
    shoulder + elbow DOFs); 'all' keeps every independent ra_ DOF (the historic behaviour).  The
    same group is applied to positions, velocities and torques (they share these DOFs).  `values`
    is (n_channels, n_times).  Predictors not in RA_DOF_PREDICTORS are returned unchanged.
    """
    if predictor not in RA_DOF_PREDICTORS:
        return names, values
    if joint_group not in JOINT_GROUPS:
        raise ValueError('Unknown joint_group {!r}; expected one of {}.'.format(
            joint_group, sorted(JOINT_GROUPS)))
    keep = [i for i, n in enumerate(names) if n.startswith('ra_') and not n.endswith('_d')]
    if joint_group != 'all':
        allowed = set(JOINT_GROUPS[joint_group])
        keep = [i for i in keep if names[i] in allowed]
    if not keep:
        raise ValueError('No ra_* (non-_d) DOFs in group {!r} among the {} {} channels.'.format(
            joint_group, len(names), predictor))
    return [names[i] for i in keep], np.asarray(values)[keep]


def _load_predictor(trial, predictor, joint_group='all'):
    """Load one trial's base predictor as (times, channel_names, values (n_channels, n_times)).

    joint_velocity is the time derivative of the joint angles (tools.filters.joint_velocity,
    unfiltered); the other predictors load their signal CSV directly.  Per-DOF predictors
    (positions, velocities, torques) are restricted to the right-arm independent joint DOFs and
    the requested `joint_group` (_filter_ra_dofs; 'all' by default -- no group restriction).
    Times are seconds since the trial's TTL pulse (the same frame as the zeroed spikes).
    """
    if predictor == 'joint_velocity':
        times, names, values = io.import_timed_csv(trial.post_kinematic_filename_csv)
        times = np.asarray(times, dtype=float)
        values = filters.joint_velocity(np.asarray(values, dtype=float), times)
    else:
        attr, _ = behavior_pooling.SIGNAL_SPECS[predictor]
        times, names, values = io.import_timed_csv(getattr(trial, attr))
        times = np.asarray(times, dtype=float)
        values = np.asarray(values, dtype=float)
    names, values = _filter_ra_dofs(predictor, names, values, joint_group)
    return times, names, values


def _trial_period_window(trial, period):
    """(t_start, t_end) seconds-since-TTL bounding `period` within a trial, or None to skip it.

    PERIOD_ALL spans the whole trial ((-inf, inf); no masking).  PERIOD_ACTIVE_MOVEMENT runs
    from the earliest available limb-movement onset (MOVEMENT_ONSET_KEYS) to hand_retreat_time;
    PERIOD_ACTIVE_GRASP from first_grasp_start to release.  The timepoints come from
    load_timepoints_into_msession (via get_timepoint); when a bounding one is missing / not
    finite or the window is non-positive, returns None so the caller drops that trial.
    """
    if period == PERIOD_ALL:
        return -np.inf, np.inf
    if period == PERIOD_ACTIVE_MOVEMENT:
        onsets = [t for t in (get_timepoint(trial, k) for k in MOVEMENT_ONSET_KEYS)
                  if t is not None]
        start = min(onsets) if onsets else None
        end = get_timepoint(trial, 'hand_retreat_time')
    elif period == PERIOD_ACTIVE_GRASP:
        start = get_timepoint(trial, 'first_grasp_start')
        end = get_timepoint(trial, 'release')
    else:
        raise ValueError('Unknown period {!r} (expected one of {}).'.format(period, PERIODS))
    if start is None or end is None or end <= start:
        return None
    return float(start), float(end)


def _lag_pair(X, Y, lag_bins):
    """Pair each response row Y[t] with covariate row X[t + lag_bins] (trimming edges).

    Positive lag_bins => the covariate follows the firing rate (anti-causal encoding);
    negative => it leads.  Returns the aligned (X, Y) for one trial.
    """
    if lag_bins > 0:
        return X[lag_bins:], Y[:-lag_bins]
    if lag_bins < 0:
        k = -lag_bins
        return X[:-k], Y[k:]
    return X, Y


def _pool_encoding_trials(server, processed_server, session, predictor, bin_width=None,
                          filter_sigma=FILTER_SIGMA, use_threshold_crossings=False,
                          period=PERIOD_ALL, joint_group='all'):
    """Per-trial (design, response) segments for a session, on a shared bin grid.

    Reads the neural source (sorted neural.nwb by default, threshold crossings when
    requested / when the sorted file is missing), pairs TTL pulses to trials positionally
    (meta_neural skip_ttl / skip_ttl_last), and for every successful trial that has the
    predictor: bins + Gaussian-smooths each unit's rate and resamples every predictor
    channel onto the same bin centres (seconds-since-TTL frame).  A combined-group predictor
    (PREDICTOR_GROUPS) concatenates its base predictors' channels, over their overlapping time
    range, into one design.  When `period` is not
    PERIOD_ALL each trial is then cropped to that sub-period's window (_trial_period_window,
    from timepoints.csv); the smoothing runs on the whole trial first, so the crop adds no
    edge artefact, and trials without a valid window are dropped.  Returns
    (trials_X, trials_Y, unit_ids, channel_names, bin_width, fps), where trials_X[i] is
    (n_bins_i, n_channels) and trials_Y[i] is (n_bins_i, n_units); the caller applies the
    lag and concatenates.  bin_width defaults to 1/mstruct['fps'].
    """
    nwb_path = resolve_neural_nwb_path(processed_server, session, use_threshold_crossings)
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(nwb_path)
    meta_neural = _load_meta_neural(processed_server, session)
    skip_ttl = resolve_meta_arg(None, meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(None, meta_neural, 'skip_ttl_last', 0)

    mstruct, _, _, msession = meta_session.load_meta_information(
        os.path.join(server, session), os.path.join(processed_server, session))

    # per-trial timepoints are only needed to crop the trials to a sub-period
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

    freq = 1.0 / bin_width
    sigma_bins = filter_sigma / bin_width

    components = _group_components(predictor)
    trials_X, trials_Y, names = [], [], None
    for trial, tspk in zip(msession, session_spikes):
        if not trial.success or not _predictor_exists(trial, predictor):
            continue
        try:
            loaded = [_load_predictor(trial, c, joint_group) for c in components]
        except Exception as e:  # noqa: BLE001
            ws('Session {} trial {}: could not read {} ({}); skipping.'.format(
                session, trial.trial_number, predictor, e))
            continue
        if any(times_c.size < 2 for times_c, _, _ in loaded):
            continue

        # one bin grid over the components' overlapping time range (identical to the single
        # component's own range when the predictor is not a group)
        t_lo = max(float(times_c[0]) for times_c, _, _ in loaded)
        t_hi = min(float(times_c[-1]) for times_c, _, _ in loaded)
        bins = np.arange(t_lo, t_hi + bin_width, bin_width)
        if bins.size < 3:
            continue
        centers = bins[:-1] + bin_width / 2

        rate = np.zeros((centers.size, n_units))
        for i_n in range(n_units):
            counts, _ = np.histogram(tspk[i_n], bins=bins)
            rate[:, i_n] = scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)

        # concatenate every component's channels, each interpolated onto the shared centres;
        # a group's channels are prefixed by their base predictor to keep the names distinct
        cols, ch_names = [], []
        for (times_c, names_c, values_c), comp in zip(loaded, components):
            for name, channel in zip(names_c, values_c):
                cols.append(np.interp(centers, times_c, channel))
                ch_names.append('{}:{}'.format(comp, name) if len(components) > 1 else name)
        design = np.column_stack(cols)
        names = ch_names if names is None else names

        # crop to the requested sub-period (whole trial for PERIOD_ALL); drop trials whose
        # window is undefined (missing timepoints) or falls outside the trial's bins
        window = _trial_period_window(trial, period)
        if window is None:
            continue
        keep = (centers >= window[0]) & (centers <= window[1])
        if not np.any(keep):
            continue
        trials_X.append(design[keep])
        trials_Y.append(rate[keep])

    if not trials_X:
        raise ValueError('No usable trials for predictor {} in session {}.'.format(
            predictor, session))
    return trials_X, trials_Y, list(unit_ids), names, bin_width, fps


def _assemble(trials_X, trials_Y, lag_bins):
    """Concatenate the per-trial segments into (X, Y) after applying `lag_bins` per trial.

    Returns (None, None) when the lag trims every trial to nothing (a lag wider than the
    trials), so the caller can skip that lag rather than fail.
    """
    Xs, Ys = [], []
    for Xd, Yd in zip(trials_X, trials_Y):
        Xl, Yl = _lag_pair(Xd, Yd, lag_bins)
        if Yl.shape[0] >= 1:
            Xs.append(Xl)
            Ys.append(Yl)
    if not Xs:
        return None, None
    return np.vstack(Xs), np.vstack(Ys)


def _maybe_pca(X, names, n_pcs):
    """Optionally replace X with its top n_pcs principal components (tools.stats.run_pca)."""
    if not n_pcs or n_pcs >= X.shape[1]:
        return X, names, None
    scores, _ = run_pca(X, n_pcs)
    k = scores.shape[1]
    return scores, ['PC{}'.format(i + 1) for i in range(k)], k


def pool_encoding(server, processed_server, session, predictor, bin_width=None,
                  filter_sigma=FILTER_SIGMA, lag=0.0, use_threshold_crossings=False,
                  period=PERIOD_ALL, joint_group='all'):
    """Continuous (X, Y) encoding design for a session at a given input->output lag.

    Returns (X (n_samples, n_channels), Y (n_samples, n_units), unit_ids, channel_names,
    bin_width, fps).  See _pool_encoding_trials for the assembly (and `period`, the trial
    sub-period the segments are cropped to, and `joint_group`, the DOF group the per-DOF
    predictors are restricted to); `lag` (s) is applied as round(lag / bin_width) bins
    (positive = covariate follows firing rate).
    """
    trials_X, trials_Y, unit_ids, names, bw, fps = _pool_encoding_trials(
        server, processed_server, session, predictor, bin_width=bin_width,
        filter_sigma=filter_sigma, use_threshold_crossings=use_threshold_crossings,
        period=period, joint_group=joint_group)
    X, Y = _assemble(trials_X, trials_Y, int(round(lag / bw)))
    if X is None:
        raise ValueError('No samples after applying lag {}s to {} / {}.'.format(
            lag, session, predictor))
    return X, Y, unit_ids, names, bw, fps


def read_optimal_lag(processed_server, session, predictor, use_threshold_crossings=False,
                     period=PERIOD_ALL, joint_group='all'):
    """Session mode optimal lag (s) for a predictor (period, joint group) from encoding/, or 0.0."""
    path = lag_json_path(processed_server, session, predictor, use_threshold_crossings, period,
                         joint_group)
    if not os.path.exists(path):
        return 0.0
    lag = io.load_json(path).get('mode_lag_s', 0.0)
    rs('Using optimal lag {:.3f}s for {} [{}] {{{}}} ({}).'.format(
        lag, predictor, period, joint_group, session))
    return float(lag)


def _save_encoding(path, session, predictor, use_threshold_crossings, period, bin_width, fps,
                   lag, n_folds, alpha, max_predictors, n_pcs, names, unit_ids, results,
                   joint_group='all'):
    """Write the per-unit encoding performance + full fit spec as human-readable JSON."""
    units = [{'unit_id': str(uid), 'pr2': _round(r[0]), 'adj_pr2': _round(r[1])}
             for uid, r in zip(unit_ids, results)]
    n_predictors = int(results[0][3]['n_predictors']) if results else 0
    n_samples = int(results[0][3]['n_samples']) if results else 0
    io.save_json({
        'session': session,
        'predictor': predictor,
        'period': period,
        'joint_group': joint_group,
        'source': 'threshold_crossings' if use_threshold_crossings else 'sorted',
        'bin_width_s': bin_width,
        'fps': fps,
        'filter_sigma_s': FILTER_SIGMA,
        'lag_s': lag,
        'model': 'poisson_glm',
        'regularization': 'l2',
        'alpha': alpha,
        'max_predictors': max_predictors,
        'n_pcs': n_pcs,
        'n_folds': n_folds,
        'n_samples': n_samples,
        'n_predictors': n_predictors,
        'predictor_channels': list(names),
        'metric': 'deviance_pseudo_r2',
        'units': units,
    }, path)
    rs('Wrote encoding performance -> {}'.format(path))


def _round(x, nd=4):
    """Round for compact, human-readable JSON; NaN -> None (valid JSON)."""
    x = float(x)
    return None if not np.isfinite(x) else round(x, nd)


def _mode(values):
    """Most frequent value (the manuscript's per-unit lag mode); ties -> smallest value."""
    vals, counts = np.unique(np.asarray(values, dtype=float), return_counts=True)
    return float(vals[int(np.argmax(counts))])


def encoding_models(server, processed_server, sessions, predictors=ALL_PREDICTORS, n_folds=5,
                    alpha=1e-4, max_predictors=None, n_pcs=None, bin_width=None,
                    units=None, plot_units=None, use_threshold_crossings=False, processes=1,
                    overwrite=False, period=None, joint_group=DEFAULT_JOINT_GROUP):
    """Fit and save Poisson GLM encoding models for each session and predictor.

    For every session and predictor, applies the session optimal lag if the matching
    <predictor>_lag[...] .json is present (else 0), assembles the design (pool_encoding),
    fits a cross-validated Poisson GLM per unit (tools.encoding.fit_glms_over_units) and
    saves the per-unit pR2 / adjusted-pR2 and the full fit specification to
    encoding/encoding_<predictor>[_<period>][_<joint_group>][_tx].json.  When plot_units is
    given, the actual vs predicted firing-rate traces of those units are also plotted (see
    plot_encoding_traces).  A predictor with no usable data in a session is skipped with a
    warning; existing outputs are skipped unless overwrite.

    `period` restricts every trial to a sub-period before fitting (see PERIODS): pass one of
    PERIODS to force it for all predictors, or None (default) to use each predictor's
    DEFAULT_PERIODS entry (resolve_period).  The effective period is recorded in each output
    file and, unless PERIOD_ALL, appended to its name; the lag applied is the one saved for
    the same period.

    `joint_group` restricts the per-DOF predictors (positions, velocities, torques) to a joint
    group (JOINT_GROUPS): 'hand' (the default; distal DOFs), 'proximal' (shoulder + elbow) or
    'all' (every independent right-arm DOF -- the previous behaviour).  The group is recorded in
    each output file and, unless 'all', appended to its name, so groups never clobber; the lag
    read is the one saved for the same group.  Force predictors are unaffected by the group.

    When `units` and/or `plot_units` are given, only those unit ids are fit (the union of the
    two, so every requested / plotted unit has its fit) and the result is reported to the log
    but NOT saved: the encoding JSON is neither checked for existence nor written, so a subset
    of units can be (re)computed, inspected and plotted without regenerating or clobbering the
    full-session file.
    """
    # plot_units needs those units fit, so treat units and/or plot_units as an inspect-only
    # subset: fit just the union of them and neither check nor overwrite the saved file.
    subset = list(dict.fromkeys((units or []) + (plot_units or []))) or None
    for session in sessions:
        if not subset:
            os.makedirs(encoding_dir(processed_server, session), exist_ok=True)
        for predictor in predictors:
            eff_period = resolve_period(predictor, period)
            out = encoding_json_path(processed_server, session, predictor,
                                     use_threshold_crossings, eff_period, joint_group)
            # With an inspect-only unit subset the saved JSON is left untouched (no existence
            # check, no write); otherwise skip predictors whose output already exists.
            if not subset and os.path.exists(out) and not overwrite:
                rs('  {} / {} [{}] {{{}}}: {} exists; skipping (use --overwrite).'.format(
                    session, predictor, eff_period, joint_group, os.path.basename(out)))
                continue
            lag = read_optimal_lag(processed_server, session, predictor,
                                   use_threshold_crossings, eff_period, joint_group)
            try:
                X, Y, unit_ids, names, bin_width_used, fps = pool_encoding(
                    server, processed_server, session, predictor, bin_width=bin_width,
                    lag=lag, use_threshold_crossings=use_threshold_crossings, period=eff_period,
                    joint_group=joint_group)
                if subset:
                    sel, unit_ids = resolve_neuron_selection(unit_ids, subset)
                    Y = Y[:, sel]
            except Exception as e:  # noqa: BLE001
                ws('Skipping {} / {} [{}] {{{}}}: {}'.format(
                    session, predictor, eff_period, joint_group, e))
                continue
            X, names, n_pcs_used = _maybe_pca(X, names, n_pcs)
            rs('Encoding {} / {} [{}] {{{}}}: {} samples, {} predictors, {} units '
               '(lag {:.3f}s).'.format(session, predictor, eff_period, joint_group, X.shape[0],
                                       X.shape[1], Y.shape[1], lag))
            results = fit_glms_over_units(X, Y, n_folds=n_folds, alpha=alpha,
                                          max_predictors=max_predictors, processes=processes)
            if subset:
                for uid, r in zip(unit_ids, results):
                    rs('  {} / {} unit {}: pR2={}, adj_pR2={}.'.format(
                        session, predictor, uid, _round(r[0]), _round(r[1])))
            else:
                _save_encoding(out, session, predictor, use_threshold_crossings, eff_period,
                               bin_width_used, fps, lag, n_folds, alpha, max_predictors,
                               n_pcs_used, names, unit_ids, results, joint_group)
            if plot_units:
                plot_encoding_traces(processed_server, session, predictor, X, Y, unit_ids,
                                     names, plot_units, bin_width_used, alpha,
                                     use_threshold_crossings)


def _mean_rates(trials_Y):
    """Mean firing rate (Hz) per unit over all concatenated samples of trials_Y."""
    total = sum(Yi.shape[0] for Yi in trials_Y)
    if not total:
        return np.zeros(trials_Y[0].shape[1] if trials_Y else 0)
    return sum(Yi.sum(axis=0) for Yi in trials_Y) / total


def _lag_surrogate_r2(trials_X, trials_Y, unit_idx, lags, bw):
    """Cheap OLS-on-sqrt-rate R2 surrogate over all lags -- the lag locator (B2/B3).

    For each lag, assembles the lag-aligned design and, in one multi-output ordinary
    least-squares solve over the selected units, computes the R2 of each unit's variance-
    stabilized rate (sqrt) on the covariates.  This is far cheaper than a Poisson GLM, so it
    can be scanned across every lag to LOCATE each unit's best lag; the exact cross-validated
    Poisson pR2 is then computed only at that lag (see optimal_lag).  The OLS is solved from
    the real per-trial-aligned cross-products (np.linalg.lstsq), so it is exact per lag.
    Returns R2 of shape (len(unit_idx), len(lags)); columns with no usable overlap are NaN.
    """
    r2 = np.full((len(unit_idx), len(lags)), np.nan)
    for j, lag in enumerate(lags):
        X, Y = _assemble(trials_X, trials_Y, int(round(lag / bw)))
        if X is None or X.shape[0] <= X.shape[1] + 1:
            continue
        Xc = X - X.mean(axis=0)                            # centring handles the intercept
        S = np.sqrt(np.clip(Y[:, unit_idx], 0.0, None))    # variance-stabilized rate
        Sc = S - S.mean(axis=0)
        beta, _, _, _ = np.linalg.lstsq(Xc, Sc, rcond=None)
        rss = np.sum((Sc - Xc @ beta) ** 2, axis=0)
        tss = np.sum(Sc ** 2, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            r2[:, j] = np.where(tss > 0, 1.0 - rss / tss, np.nan)
    return r2


def optimal_lag(server, processed_server, sessions, predictors=ALL_PREDICTORS, lag_min=LAG_MIN,
                lag_max=LAG_MAX, lag_step=None, n_folds=5, alpha=1e-4, max_predictors=None,
                n_pcs=None, min_rate=LAG_MIN_RATE_HZ, min_adj_r2=MIN_ADJ_R2, bin_width=None,
                use_threshold_crossings=False, processes=1, overwrite=False, period=None,
                write_encoding=True, joint_group=DEFAULT_JOINT_GROUP):
    """Find and save the pR2-maximizing input->output lag per unit (and its session mode).

    For each session and predictor the per-trial segments are assembled once, then:
      * (C1) only units whose mean firing rate exceeds `min_rate` Hz are searched -- low-rate
        units give a flat, noisy lag profile and only add noise to the mode;
      * (B2/B3) each kept unit's lag is LOCATED with a cheap all-lags OLS-on-sqrt-rate R2
        surrogate (_lag_surrogate_r2), searching only the interior lags (the +/- window edges
        are ignored, since an argmax pinned at the sweep boundary is an unreliable, clipped
        optimum), and the exact cross-validated Poisson pR2 is computed only at that selected
        lag (units that chose the same lag are fit together), so the Poisson GLM is not fit at
        every lag.
    A unit is dropped from the session mode when its confirming adjusted pR2 is below
    `min_adj_r2` (poorly-encoded units carry no meaningful lag).  Each kept unit's optimal
    lag is the surrogate argmax; its saved best_pr2 is the confirming Poisson pR2; the session
    lag is the mode across the kept units.  `n_pcs` / `max_predictors` shape the confirming
    Poisson fit only (the surrogate uses the raw covariates).  `period` restricts every trial
    to a sub-period before the search (see PERIODS / resolve_period; None -> each predictor's
    DEFAULT_PERIODS entry), so the saved lag matches the period encoding_models will fit on.
    Saved to encoding/<predictor>_lag[_<period>][_tx].json.  Unless `write_encoding` is False it
    then also fits and writes the encoding file at the session mode lag from the same pooled
    segments (encoding/encoding_<predictor>[...] .json -- the identical output encoding_models
    would produce), so a single lag run yields both files without a separate encoding pass.
    Existing lag / encoding outputs are skipped unless overwrite (a run can back-fill just the
    missing one -- e.g. write the encoding for a period whose lag already exists).  `joint_group`
    restricts the per-DOF predictors to a JOINT_GROUPS group ('hand' default; see encoding_models);
    it is recorded / appended to the file names so the lag matches the group encoding_models fits.
    """
    import tqdm

    for session in sessions:
        os.makedirs(encoding_dir(processed_server, session), exist_ok=True)
        for predictor in predictors:
            eff_period = resolve_period(predictor, period)
            lag_out = lag_json_path(processed_server, session, predictor,
                                    use_threshold_crossings, eff_period, joint_group)
            enc_out = encoding_json_path(processed_server, session, predictor,
                                         use_threshold_crossings, eff_period, joint_group)
            need_lag = overwrite or not os.path.exists(lag_out)
            need_enc = write_encoding and (overwrite or not os.path.exists(enc_out))
            if not need_lag and not need_enc:
                rs('  {} / {} [{}] {{{}}}: lag and encoding exist; skipping (use '
                   '--overwrite).'.format(session, predictor, eff_period, joint_group))
                continue
            try:
                trials_X, trials_Y, unit_ids, names, bw, fps = _pool_encoding_trials(
                    server, processed_server, session, predictor, bin_width=bin_width,
                    use_threshold_crossings=use_threshold_crossings, period=eff_period,
                    joint_group=joint_group)
            except Exception as e:  # noqa: BLE001
                ws('Skipping {} / {} [{}] {{{}}}: {}'.format(
                    session, predictor, eff_period, joint_group, e))
                continue

            if need_lag:
                step = lag_step or bw
                lags = np.arange(lag_min, lag_max + step / 2, step)

                # C1: only search sufficiently active units
                kept = np.nonzero(_mean_rates(trials_Y) > min_rate)[0]
                rs('Optimal lag {} / {} [{}]: {} lags; {} / {} units above {} Hz.'.format(
                    session, predictor, eff_period, len(lags), kept.size, len(unit_ids),
                    min_rate))
                if kept.size == 0:
                    ws('Skipping {} / {}: no unit above {} Hz.'.format(
                        session, predictor, min_rate))
                    continue

                # B2/B3: locate each kept unit's lag with the cheap OLS surrogate over all lags,
                # ignoring the +/- window edges (a peak pinned at the sweep boundary is clipped /
                # unreliable) by masking the first and last lag before the argmax.
                surrogate = _lag_surrogate_r2(trials_X, trials_Y, kept, lags, bw)
                if surrogate.shape[1] > 2:
                    surrogate[:, 0] = np.nan
                    surrogate[:, -1] = np.nan
                best_j = np.array([int(np.nanargmax(row)) if not np.all(np.isnan(row)) else -1
                                   for row in surrogate])

                # B2: confirm with the exact CV Poisson pR2 only at each unit's selected lag,
                # fitting together the units that chose the same lag (one assemble+fit per lag).
                best_lag = np.full(len(unit_ids), np.nan)
                best_pr2 = np.full(len(unit_ids), np.nan)
                distinct = sorted(set(best_j[best_j >= 0].tolist()))
                for j in tqdm.tqdm(distinct, desc='lag confirm {}/{}'.format(session, predictor),
                                   ncols=100):
                    members = kept[best_j == j]                # global unit indices at lag j
                    X, Y = _assemble(trials_X, trials_Y, int(round(lags[j] / bw)))
                    if X is None:
                        continue
                    if n_pcs:
                        X, _, _ = _maybe_pca(X, None, n_pcs)
                    results = fit_glms_over_units(X, Y[:, members], n_folds=n_folds, alpha=alpha,
                                                  max_predictors=max_predictors,
                                                  processes=processes, progress=False)
                    for gidx, r in zip(members, results):
                        # drop poorly-encoded units (adjusted pR2 below threshold) from the mode
                        if np.isfinite(r[1]) and r[1] >= min_adj_r2:
                            best_lag[gidx] = float(lags[j])
                            best_pr2[gidx] = r[0]

                mode_lag = _save_lag(lag_out, session, predictor, use_threshold_crossings,
                                     eff_period, bw, fps, lags, unit_ids, best_lag, best_pr2,
                                     min_rate, min_adj_r2, joint_group)
            else:
                # lag already computed; reuse its session mode to back-fill the encoding file
                mode_lag = read_optimal_lag(processed_server, session, predictor,
                                            use_threshold_crossings, eff_period, joint_group)

            # populate the encoding file at the session mode lag from the same pooled segments,
            # so a lag run yields the encoding too, without a separate encoding pass
            if need_enc:
                _encode_at_lag_and_save(
                    enc_out, trials_X, trials_Y, unit_ids, names, session, predictor,
                    use_threshold_crossings, eff_period, bw, fps, mode_lag, n_folds, alpha,
                    max_predictors, n_pcs, processes, joint_group)


def _save_lag(path, session, predictor, use_threshold_crossings, period, bin_width, fps, lags,
              unit_ids, best_lag, best_pr2, min_rate, min_adj_r2, joint_group='all'):
    """Write per-unit optimal lags + the session mode lag as human-readable JSON.

    best_lag / best_pr2 are per-unit arrays (NaN for units left out of the search -- below
    the min_rate threshold, with an edge/empty surrogate, or with a confirming adjusted pR2
    below min_adj_r2); those are written as null and excluded from the mode, which is taken
    over the kept units.
    """
    per_unit, kept_lags = [], []
    for uid, bl, bp in zip(unit_ids, best_lag, best_pr2):
        if bl is None or not np.isfinite(bl):
            per_unit.append({'unit_id': str(uid), 'best_lag_s': None, 'best_pr2': None})
        else:
            kept_lags.append(float(bl))
            per_unit.append({'unit_id': str(uid), 'best_lag_s': _round(bl, 4),
                             'best_pr2': _round(bp)})
    mode_lag = _mode(kept_lags) if kept_lags else 0.0
    io.save_json({
        'session': session,
        'predictor': predictor,
        'period': period,
        'joint_group': joint_group,
        'source': 'threshold_crossings' if use_threshold_crossings else 'sorted',
        'bin_width_s': bin_width,
        'fps': fps,
        'lag_grid_s': [round(float(l), 4) for l in lags],
        'mode_lag_s': round(mode_lag, 4),
        'min_rate_hz': min_rate,
        'min_adj_r2': min_adj_r2,
        'n_units': len(unit_ids),
        'n_units_kept': len(kept_lags),
        'lag_method': 'interior_ols_sqrt_rate_surrogate_argmax + cv_poisson_pr2_confirm',
        'metric': 'deviance_pseudo_r2',
        'units': per_unit,
    }, path)
    rs('Wrote optimal lag (mode {:.3f}s from {} unit(s)) -> {}'.format(
        mode_lag, len(kept_lags), path))
    # return the rounded value stored / read back, so a downstream encoding fit at this lag
    # matches whether it was just computed or later read via read_optimal_lag
    return round(mode_lag, 4)


def _encode_at_lag_and_save(path, trials_X, trials_Y, unit_ids, names, session, predictor,
                            use_threshold_crossings, period, bin_width, fps, lag,
                            n_folds, alpha, max_predictors, n_pcs, processes, joint_group='all'):
    """Fit a per-unit Poisson GLM at `lag` on already-pooled trials and save the encoding JSON.

    Lets optimal_lag populate the encoding file at the session mode lag from the segments it
    already pooled -- the identical fit + output as encoding_models -- without a separate
    encoding pass or re-pooling.  Writes nothing and returns False when the lag trims every
    trial away (see _assemble); otherwise returns True.
    """
    X, Y = _assemble(trials_X, trials_Y, int(round(lag / bin_width)))
    if X is None:
        ws('{} / {} [{}]: lag {}s leaves no samples; encoding not written.'.format(
            session, predictor, period, lag))
        return False
    X, names, n_pcs_used = _maybe_pca(X, names, n_pcs)
    results = fit_glms_over_units(X, Y, n_folds=n_folds, alpha=alpha,
                                  max_predictors=max_predictors, processes=processes)
    _save_encoding(path, session, predictor, use_threshold_crossings, period, bin_width, fps,
                   lag, n_folds, alpha, max_predictors, n_pcs_used, names, unit_ids, results,
                   joint_group)
    return True


def plot_encoding_traces(processed_server, session, predictor, X, Y, unit_ids, names,
                         requested, bin_width, alpha, use_threshold_crossings=False,
                         n_samples=600, save=True):
    """Plot actual vs GLM-predicted firing-rate traces for requested units.

    For each requested unit id, refits a Poisson GLM on the full (X, Y) design and plots
    the actual (grey) and predicted (coloured) rate over the first `n_samples` bins, one
    panel per unit.  Saved as <session>/prehension_plots/encoding_traces_<predictor>[_tx].png.
    Returns the figure.
    """
    import warnings
    import matplotlib.pyplot as plt
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import PoissonRegressor
    from ..tools import plotting

    sel, labels = resolve_neuron_selection(unit_ids, requested)
    t = np.arange(min(n_samples, X.shape[0])) * bin_width
    xn, yn = plotting.xy_numsubplots(len(sel))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    for ax, i_u, uid in zip(axs, sel, labels):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            model = PoissonRegressor(alpha=alpha, max_iter=300).fit(X, Y[:, i_u])
        pred = model.predict(X)
        ax.plot(t, Y[:t.size, i_u], color='0.6', linewidth=0.8, label='actual')
        ax.plot(t, pred[:t.size], color='tab:red', linewidth=1.0, label='predicted')
        ax.set_title('unit {}'.format(uid), fontsize=8)
        ax.set_xlabel('Time, s', fontsize=7)
        ax.set_ylabel('Rate, Hz', fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axs[len(sel):]:
        ax.axis('off')
    axs[0].legend(fontsize=7)
    fig.suptitle('{} {} encoding: actual vs predicted firing rate'.format(session, predictor))
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save:
        save_dir = os.path.join(processed_server, session, 'prehension_plots')
        plotting.savefig(save_dir, 'encoding_traces_{}{}'.format(
            predictor, _suffix(use_threshold_crossings)), fig=fig)
        rs('Saved encoding traces for {} unit(s).'.format(len(sel)))
    return fig
