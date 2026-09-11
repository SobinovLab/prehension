#!python3
# -*- coding: utf-8 -*-
"""
Cross-session pooling of a behavioural signal into per-condition averages.

pool_behavior mirrors neural_plotting.common.pooling.pool_neurons, but the "features"
are the channels of a behavioural signal (joint angles, torques, digit or segment
forces) instead of neurons: for every successful trial with a valid alignment
timepoint, each channel of the signal CSV is resampled onto a shared bin grid aligned
to that timepoint (the signals live in the seconds-since-TTL frame, like the spikes),
grouped by the object condition (get_target_force), and averaged.  The returned
``entries`` have the same shape pool_neurons produces, so the population aggregation
(build_condition_matrix / build_condition_tensors -> run_pca / run_dpca) and the
neural_plotting drawers work on them unchanged.

Behaviour-only (no NWB, no NeuralConfig), so it runs on any session with processed
behaviour, including the Utah-array datasets.

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

from .. import meta_session
from ..tools import io
from ..tools.logs import rs, ws
from ..neural_processing.common.spikes import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH)
from ..neural_plotting.common.behaviour import (
    load_timepoints_into_msession, get_timepoint, get_target_force)


# signal name -> (TrialInfo CSV-path attribute, TrialInfo existence-check method).  Each
# CSV is a 'time' column plus one column per channel (tools.io.import_timed_csv).
SIGNAL_SPECS = {
    'joint_angles': ('post_kinematic_filename_csv', 'does_post_kin_file_exist'),
    'torques': ('torques_filename', 'does_torque_file_exist'),
    'digit_forces': ('digit_forces_filename', 'does_digit_force_file_exist'),
    'segment_forces': ('segment_forces_filename', 'does_segment_force_file_exist'),
}


def _resample_trial(filename, tp, bin_centers):
    """Resample every channel of a signal CSV onto the timepoint-aligned bin grid.

    The CSV times are seconds since the trial's TTL pulse (the same frame as ``tp``),
    so the channels are sampled at ``tp + bin_centers`` by linear interpolation.
    Returns (channel_names, values (n_channels, numbins)).
    """
    times, names, values = io.import_timed_csv(filename)
    times = np.asarray(times, dtype=float)
    sample_at = tp + bin_centers
    resampled = np.array([np.interp(sample_at, times, np.asarray(v, dtype=float))
                          for v in values])
    return names, resampled


def pool_behavior(server, processed_server, sessions, signal, group_column=GROUP_COLUMN,
                  align_key=ALIGN_TIMEPOINT, before=BEFORE, after=AFTER, bin_width=BIN_WIDTH):
    """Pool a behavioural signal's channels across sessions into per-condition averages.

    For each session: load the behavioural meta, keep the successful trials that have the
    signal file and a valid ``align_key`` timepoint, resample every channel onto a shared
    bin grid aligned to that timepoint, group the trials by ``group_column`` (the object
    condition, e.g. target force) and average (mean +/- SEM) per channel.  Sessions
    lacking the signal / meta / a usable trial are skipped with a warning.

    Arguments:
        signal {str} --- one of SIGNAL_SPECS ('joint_angles', 'torques', 'digit_forces',
            'segment_forces').

    Returns (entries, bin_centers, max_force) where each entry is
    {'label': '<session>: <channel>', 'frs_avg': (n_groups, numbins),
    'frs_sem': (n_groups, numbins), 'group_ids': [condition, ...]} -- the same shape
    pool_neurons returns -- and max_force is the largest condition value seen.
    """
    if signal not in SIGNAL_SPECS:
        raise ValueError('Unknown signal {!r} (expected {}).'.format(
            signal, sorted(SIGNAL_SPECS)))
    attr, exists = SIGNAL_SPECS[signal]

    found = ([s for s in sessions if os.path.isdir(os.path.join(processed_server, s))]
             if sessions else meta_session.find_session_dirs(processed_server))

    bins = np.arange(-before - bin_width / 2, after + bin_width / 2, bin_width)
    bin_centers = bins[:-1] + bin_width / 2

    entries = []
    max_force = 0.0
    for session in found:
        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                os.path.join(server, session), os.path.join(processed_server, session))
            load_timepoints_into_msession(msession, mstruct)
        except Exception as e:  # noqa: BLE001
            ws('Skipping session {}: {}'.format(session, e))
            continue

        trials = []
        for t in msession:
            if not t.success or not getattr(t, exists)():
                continue
            tp = get_timepoint(t, align_key)
            if tp is not None:
                trials.append((t, tp))
        if not trials:
            ws("Skipping session {}: no successful '{}' trials with a '{}' timepoint.".format(
                session, signal, align_key))
            continue

        group = [get_target_force(mobject, t.object_id, group_column) for t, _ in trials]
        group_ids = sorted(set(group))
        if group:
            max_force = max(max_force, max(group))

        # resample every trial onto the shared grid (aligned to its own timepoint)
        names = None
        per_trial = []
        for t, tp in trials:
            ch_names, resampled = _resample_trial(getattr(t, attr), tp, bin_centers)
            names = ch_names if names is None else names
            per_trial.append(resampled)
        arr = np.stack(per_trial, axis=0)   # (n_trials, n_channels, numbins)

        for i_c, ch in enumerate(names):
            frs = arr[:, i_c, :]            # (n_trials, numbins)
            frs_avg = np.zeros((len(group_ids), len(bin_centers)))
            frs_sem = np.zeros((len(group_ids), len(bin_centers)))
            for i_g, gid in enumerate(group_ids):
                ingroup = [i for i, g in enumerate(group) if g == gid]
                frs_avg[i_g, :] = np.mean(frs[ingroup, :], axis=0)
                frs_sem[i_g, :] = np.std(frs[ingroup, :], axis=0) / np.sqrt(len(ingroup))
            entries.append({'label': '{}: {}'.format(session, ch),
                            'frs_avg': frs_avg, 'frs_sem': frs_sem,
                            'group_ids': group_ids})
        rs('Pooled {} {} channel(s) from session {}.'.format(len(names), signal, session))

    return entries, bin_centers, max_force
