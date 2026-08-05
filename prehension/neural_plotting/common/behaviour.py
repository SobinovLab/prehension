#!python3
# -*- coding: utf-8 -*-
"""
Behavioural-meta helpers shared across the neural figure modules.

These bridge the prehension behavioural model (msession trials, timepoints.csv,
object force definitions) to the neural figures: attach per-trial timepoints, look
up a trial's alignment time, and read an object property (e.g. targetForce).  They
are tied to the prehension meta model, so they live in neural_plotting rather than
the reusable neural_processing.common layer.

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

from ...tools import io
from ...tools.logs import rs, ws


def load_timepoints_into_msession(msession, mstruct):
    """Attach per-trial timepoints (from timepoints.csv) to each trial as trial.timepoints.

    The timepoints file is optional: when it is missing every trial gets an empty
    timepoints dict, and alignment can still fall back to the meta_session 'ttl_to_*'
    columns (already on trial.other_info); see get_timepoint.
    """
    tp_path = mstruct['timepoint_csv_filename']
    if not tp_path or not os.path.exists(tp_path):
        ws('No timepoints file at {}; only meta_session timepoints (ttl_to_*) will be '
           'available for alignment.'.format(tp_path))
        for trial in msession:
            trial.timepoints = {}
        return
    tp_dic = io.import_csv_as_dic(tp_path)
    trial_number_col = tp_dic['trial_number']
    del tp_dic['trial_number']
    # optional occurrence index to disambiguate duplicate recordings of the same trial_number
    dup_col = tp_dic.pop('trial_dup_index', None)
    for trial in msession:
        if dup_col is not None:
            trial_row = None
            for i, tn in enumerate(trial_number_col):
                if (int(float(tn)) == trial.trial_number and
                        int(float(dup_col[i])) == trial.dup_index):
                    trial_row = i
                    break
            if trial_row is None:
                raise ValueError('Trial {} (occurrence {}) not found in timepoints.'.format(
                    trial.trial_number, trial.dup_index))
        else:
            trial_row = trial_number_col.index(trial.trial_number)
        trial.timepoints = {k: v[trial_row] for k, v in tp_dic.items()}


def get_timepoint(trial, key):
    """Return the trial's alignment time (seconds since the TTL pulse) for `key`.

    `key` may name either a column from the timepoints CSV (e.g. 'first_grasp_start')
    or a column from meta_session.csv -- the 'ttl_to_*' offsets loaded onto
    trial.other_info by meta_session.load_meta_information (e.g. 'ttl_to_success_grasp',
    'ttl_to_reach', 'ttl_to_force_target_start'). The timepoints CSV is searched first,
    then meta_session. Both store times in the same reference frame (seconds since the
    trial's TTL pulse), so either can be used to align spikes.

    Returns None if the value is missing or not finite.
    """
    v = getattr(trial, 'timepoints', {}).get(key, None)
    if v is None:
        v = (trial.other_info or {}).get(key, None)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def get_target_force(mobject, object_id, group_column):
    key = object_id
    if key not in mobject and str(key) in mobject:
        key = str(key)
    return float(mobject[key]['def'][group_column])
