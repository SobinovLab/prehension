#!python3
# -*- coding: utf-8 -*-
"""
Spike reading and per-trial slicing (reusable across neural work).

Read per-unit spike times and TTL windows from the minimal neural NWB, slice each
unit's spikes into per-trial windows, and resolve a requested set of unit ids to
indices.  These operate only on arrays / an NWB path (no behavioural meta, no
NeuralConfig), so they are shared by both neural processing and plotting.  The
shared default alignment/windowing constants live here as the single source of
truth for every figure module.

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
import numpy as np

from ...tools.logs import rs, ws

# defaults; overridable through the calling functions / scripts
ALIGN_TIMEPOINT = 'first_grasp_start'
GROUP_COLUMN = 'targetForce(N)'
BEFORE = 1.0            # s before the alignment timepoint
AFTER = 1.0            # s after
BIN_WIDTH = 0.02       # s
FILTER_SIGMA = 0.05    # s, Gaussian smoothing of firing-rate traces


def read_nwb_spikes_and_ttl(nwb_path):
    """Read per-unit spike times, unit ids and per-pulse TTL windows from the NWB.

    Returns (spike_per_unit {list of arrays, s}, unit_ids {list}, events_time
    {list of [start, stop] arrays, s}).
    """
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_path), 'r') as fio:
        nwbfile = fio.read()

        udf = nwbfile.units.to_dataframe()
        spike_per_unit = [np.asarray(s, dtype=float) for s in udf['spike_times']]
        if 'unit_id' in udf.columns:
            unit_ids = [u for u in udf['unit_id']]
        else:
            unit_ids = list(udf.index)

        ttl = nwbfile.intervals['ttl_pulses'].to_dataframe()
        events_time = [np.array([float(a), float(b)])
                       for a, b in zip(ttl['start_time'], ttl['stop_time'])]
    rs('NWB: {} units, {} TTL pulses.'.format(len(unit_ids), len(events_time)))
    return spike_per_unit, unit_ids, events_time


def read_nwb_unit_depths(nwb_path):
    """Read per-unit depth (um along the probe) from the NWB, keyed by unit id.

    Returns {unit_id: depth} for the units in the NWB Units table (depth is the
    unit-location y-coordinate written by export_nwb; larger y is farther from the
    electrode tip).  Returns {} for products written before depth extraction (no
    'depth' column).  Keys are the same unit ids returned by
    read_nwb_spikes_and_ttl, so callers can align depths to those unit ids.
    """
    from pynwb import NWBHDF5IO

    with NWBHDF5IO(str(nwb_path), 'r') as fio:
        nwbfile = fio.read()
        udf = nwbfile.units.to_dataframe()
        if 'depth' not in udf.columns:
            rs('NWB has no unit depth column.')
            return {}
        if 'unit_id' in udf.columns:
            unit_ids = [u for u in udf['unit_id']]
        else:
            unit_ids = list(udf.index)
        depths = {uid: float(d) for uid, d in zip(unit_ids, udf['depth'])}
    rs('NWB: read depth for {} units.'.format(len(depths)))
    return depths


def get_trial_data_spike(spike_per_unit, events_time):
    """For each trial window [start, stop], extract per-unit spikes within it."""
    trial_spike = []
    for e in events_time:
        temp = []
        for spike in spike_per_unit:
            idx = np.where((spike > e[0]) & (spike < e[1]))
            temp.append(spike[idx])
        trial_spike.append(temp)
    return trial_spike


def resolve_neuron_selection(unit_ids, neuron_ids):
    """Map requested unit ids to indices into unit_ids.  None/[] -> all."""
    if not neuron_ids:
        return list(range(len(unit_ids))), list(unit_ids)
    id_to_index = {}
    for i, uid in enumerate(unit_ids):
        id_to_index[uid] = i
        id_to_index[str(uid)] = i
    selection, labels, missing = [], [], []
    for uid in neuron_ids:
        idx = id_to_index.get(uid, id_to_index.get(str(uid)))
        if idx is None:
            missing.append(uid)
        else:
            selection.append(idx)
            labels.append(unit_ids[idx])
    if missing:
        ws('Requested unit ids not found and skipped: {}'.format(missing))
    if not selection:
        raise ValueError('None of neuron_ids={} matched {}.'.format(neuron_ids, unit_ids))
    return selection, labels


def fit_session_drift(spike_per_unit, events_time):
    """Per-unit linear drift of firing rate across the session.

    Fits, for each unit, its per-trial firing rate (# spikes inside the trial's TTL
    window / window length) against the window's start time -- a line, i.e. drift
    assumed linear in time.  Returns (slopes, t_ref): ``slopes[i]`` is unit i's rate
    trend in Hz per second and ``t_ref`` is the mean window-start time, so the drift
    offset to SUBTRACT from unit i's rate for a trial whose window starts at ``t`` is
    ``slopes[i] * (t - t_ref)`` (Hz).  Because the offset is centred on t_ref, the
    subtraction removes the session-long trend while preserving each unit's overall
    mean rate.  ``slopes`` is 0 for a unit that cannot be fit (fewer than two trials,
    no time span, or too few finite per-trial rates).
    """
    starts = np.array([float(e[0]) for e in events_time], dtype=float)
    n_units = len(spike_per_unit)
    slopes = np.zeros(n_units, dtype=float)
    t_ref = float(starts.mean()) if starts.size else 0.0
    if starts.size < 2 or float(np.ptp(starts)) == 0.0:
        return slopes, t_ref

    durs = np.array([float(e[1] - e[0]) for e in events_time], dtype=float)
    durs[durs <= 0] = np.nan
    for i, s in enumerate(spike_per_unit):
        s = np.asarray(s, dtype=float)
        counts = np.array([np.count_nonzero((s > e[0]) & (s < e[1])) for e in events_time],
                          dtype=float)
        rates = counts / durs
        good = np.isfinite(rates)
        if np.count_nonzero(good) >= 2 and float(np.ptp(starts[good])) > 0.0:
            slopes[i] = float(np.polyfit(starts[good], rates[good], 1)[0])
    rs('Session drift fit over {} trials for {} units (linear in time).'.format(
        starts.size, n_units))
    return slopes, t_ref


def drift_offset(slopes, t_ref, unit_index, ttl_start):
    """Hz to SUBTRACT from unit ``unit_index``'s rate for a trial starting at ``ttl_start``.

    ``slopes``/``t_ref`` come from fit_session_drift; returns 0.0 when slopes is None
    (drift correction disabled), so callers can subtract unconditionally.
    """
    if slopes is None:
        return 0.0
    return float(slopes[unit_index]) * (float(ttl_start) - t_ref)
