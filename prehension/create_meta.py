#!python3
# -*- coding: utf-8 -*-
"""
Creates meta information files for experimental sessions.

Copyright (C) 2019-2024 Anton Sobinov
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
import datetime
import glob
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .tools.constants import ORIGINAL_OPENSIM_MODEL, CALIBRATIONS_DIR
from .tools import io
from .tools.logs import rs, ws

from .tools.session_management import apply_to_sessions_helper

from . import meta_session


def get_object_sets(sy_column_names, sy_data, object_def_columns):
    odcs = [v for v in object_def_columns if v in sy_column_names]
    object_def_column_ids = [sy_column_names.index(v) for v in odcs]
    u_objects = np.unique(sy_data[:, object_def_column_ids], axis=0)

    object_ids = np.zeros((np.shape(sy_data)[0],), dtype=int)
    object_ids.fill(-1)

    rs('\tFound {} unique objects:'.format(np.shape(u_objects)[0]))
    rs('|'.join('{:>25}'.format(v) for v in ['id'] + list(odcs)))
    for i_object, u_object in enumerate(u_objects):
        rs('|'.join('{:25}'.format(v) for v in [i_object] + u_object.tolist()))
        object_ids[np.all(sy_data[:, object_def_column_ids] == u_object, axis=1)] = i_object

    return u_objects, object_ids, odcs


def collect_target_suffixes(base, ext, column_names):
    '''Find the per-target column suffixes present for a base behaviour-log column name.

    Multi-target trials append a target index to the relevant column names, mirroring the
    multi-target object-definition columns (e.g. ``targetForce(N)`` for the first target and
    ``targetForce_2(N)`` for the second). The first target uses the bare name and subsequent
    targets use ``_2``, ``_3``, ... suffixes.

    Args:
        base (str): the column name without the target suffix or extension, e.g.
            ``'started_touching_time'``.
        ext (str): the trailing unit/type portion of the column name, e.g. ``'(ms)'``.
        column_names (list): the behaviour-log column names to search.

    Returns:
        list of str: the ordered suffixes for which ``'{base}{suffix}{ext}'`` is a column, with
        ``''`` (the first target) first, followed by ``'_2'``, ``'_3'``, ... for as long as they
        are present.
    '''
    suffixes = []
    if base + ext in column_names:
        suffixes.append('')
    i_target = 2
    while '{}_{}{}'.format(base, i_target, ext) in column_names:
        suffixes.append('_{}'.format(i_target))
        i_target += 1
    return suffixes


def _duplicate_trial_counts(trial_nums):
    '''Given an array/list of trial numbers, return {trial_num: count} for every trial number that
    appears more than once (in the order the numbers first appear).'''
    trial_nums = np.asarray(trial_nums).astype(int)
    counts = {}
    for t in trial_nums:
        counts[int(t)] = counts.get(int(t), 0) + 1
    return {t: c for t, c in counts.items() if c > 1}


def find_duplicate_trials(mstruct):
    '''Read and concatenate the session's auto (behavioural) logs and report duplicate trial IDs.

    Mirrors the log loading done in ``import_logs`` so detection is consistent, but is read-only
    and creates nothing.

    Args:
        mstruct (dict): a meta structure with ``'auto_log'`` populated (a list of log paths).

    Returns:
        tuple(dict, int): ``(duplicates, n_total)`` where ``duplicates`` maps each ``trial_num``
        that appears more than once to its number of occurrences, and ``n_total`` is the total
        number of behavioural-log rows.
    '''
    if len(mstruct['auto_log']) == 0 or not os.path.exists(mstruct['auto_log'][0]):
        raise ValueError('Auto log for this session does not exist.')

    sy_column_names, sy_data = io.import_csv(mstruct['auto_log'][0])
    sy_data = np.array(sy_data).transpose()
    for al in mstruct['auto_log'][1:]:
        _, sd = io.import_csv(al)
        sd = np.array(sd).transpose()
        sy_data = np.concatenate((sy_data, sd), axis=0)

    trial_nums = sy_data[:, sy_column_names.index('trial_num')].astype(int)
    return _duplicate_trial_counts(trial_nums), len(trial_nums)


def import_logs(dirname, mstruct):
    # check file existence for meaningful error throw
    if not os.path.exists(mstruct['auto_log'][0]):
        raise ValueError('Auto log for this session does not exist.')

    sy_ja_file = os.path.join(mstruct['videos_dir'], 'trial_log*.csv')
    sy_ja_file = glob.glob(sy_ja_file)
    if len(sy_ja_file) == 0:  # if the images have not been converted to videos
        sy_ja_file = os.path.join(mstruct['images_dir'], 'trial_log.csv')
    else:
        sy_ja_file = sy_ja_file[0]

    no_cam_log = False
    if not os.path.exists(sy_ja_file):
        ws('Camera log for this session does not exist.')
        no_cam_log = True
        # raise ValueError('Camera log for this session does not exist.')

    sy_ps_file = os.path.join(mstruct['raw_ps_dir'], 'trial_log.csv')
    no_ps_log = False
    if not os.path.exists(sy_ps_file):
        no_ps_log = True
        ws('Pressure sensor log for this session does not exist.')
        # raise ValueError('Pressure sensor log for this session does not exist.')

    # If neither cam or ps log fail
    if no_cam_log and no_ps_log:
        raise ValueError("Must have camera log or ps log")

    # import general sync data
    sy_column_names, sy_data = io.import_csv(mstruct['auto_log'][0])
    sy_data = np.array(sy_data).transpose()

    for al in mstruct['auto_log'][1:]:
        _, sd = io.import_csv(al)
        sd = np.array(sd).transpose()
        sy_data = np.concatenate((sy_data, sd), axis=0)

    # check for duplication from logs
    # duplicate trial IDs are NOT fatal: they are preserved as distinct trials and routed to the
    # suffixed ('_1', ...) recordings later (see reorder_to_common_trials and TrialInfo).
    trial_nums = sy_data[:, sy_column_names.index('trial_num')].astype(int)
    duplicates = _duplicate_trial_counts(trial_nums)
    if duplicates:
        ws('Auto logs in {} have duplicating trials (preserved and suffixed): {}'.format(
            dirname,
            ', '.join('{}x{}'.format(t, c) for t, c in sorted(duplicates.items()))))

    # cameras data
    if os.path.exists(sy_ja_file):
        sy_ja_column_names, sy_ja_data = io.import_csv(sy_ja_file)
        sy_ja_data = np.array(sy_ja_data).transpose()
    else:
        sy_ja_column_names = []
        sy_ja_data = []

    # import pressure sensor sync data
    sy_ps_column_names, sy_ps_data = io.import_csv(sy_ps_file)
    sy_ps_data = np.array(sy_ps_data).transpose()

    return sy_column_names, sy_data, sy_ja_column_names, sy_ja_data, sy_ps_column_names, sy_ps_data


def reorder_to_common_trials(sy_column_names, sy_data, sy_ja_column_names, sy_ja_data,
                             sy_ps_column_names, sy_ps_data):
    '''
    Restrict the protocol, camera, and pressure-sensor logs to the trials they have in common and
    put every stream in the same, chronological (log appearance) order.

    Duplicate trial IDs (the same ``trial_num`` recorded more than once, e.g. after a session
    restart) are NOT dropped. Each row is tagged with an occurrence index ``dup_index`` (0 for the
    first recording of a trial number, 1 for the second, ...), and trials are matched across streams
    on the composite key ``(trial_num, dup_index)``. This keeps the k-th recording of a duplicated
    ID aligned across streams and routes later occurrences to their suffixed ('_1', ...) files.

    Ordering follows the protocol log's appearance order (i.e. recording order), NOT sorted by
    trial number, so the positional neural TTL-pulse<->trial correspondence stays valid across
    duplicates and multi-run sessions.

    If the camera log is empty, the common set is (protocol AND pressure); if the pressure log is
    empty, the common set is (protocol AND camera).

    Returns:
        tuple: ``(common_trials, common_dup_index, sy_data, sy_ja_data, sy_ps_data)`` where the
        first two are row-aligned arrays of trial numbers and occurrence indices, and the three
        data arrays are reordered/filtered to those same rows in the same order.
    '''

    # --- helpers --- #
    def get_logs_with_occurrences(data, data_columns, log_label):
        '''Keep every row (no dropping) and tag each with a 0-based occurrence index computed in
        appearance order per trial number. Returns (data, trials, occ).'''
        if len(data) > 0 and 'trial_num' in data_columns:
            trials = data[:, data_columns.index('trial_num')].astype(int)

            seen = {}
            occ = np.empty(len(trials), dtype=int)
            for index, trial in enumerate(trials):
                trial = int(trial)
                occ[index] = seen.get(trial, 0)
                seen[trial] = occ[index] + 1

            n_dup = sum(1 for c in seen.values() if c > 1)
            if n_dup > 0:
                ws('{} trial(s) in the {} log have duplicate recordings; '
                   'preserving all occurrences.'.format(n_dup, log_label))

            return data, trials, occ

        # Case where we don't have any incoming data (if no cam logs are found for example)
        return np.array([]), np.array([]), np.array([])

    def print_absent_trials(a_trials, b_trials, common, a_label, b_label):
        # In a but not b
        a_trials = np.array(a_trials)
        b_trials = np.array(b_trials)

        absent_from_a = np.isin(a_trials, common, invert=True)
        if sum(absent_from_a) > 0:
            rs('{} {} trials are absent from {}: {}'.format(
                sum(absent_from_a),
                b_label,
                a_label,
                ', '.join([str(v) for v in a_trials[absent_from_a]])))
        else:
            rs(f'No {b_label} trials are absent from {a_label}.')

        # In b but not a
        absent_from_b = np.isin(b_trials, common, invert=True)

        if sum(absent_from_b) > 0:
            rs('{} trials are absent from {}: {}'.format(
                sum(absent_from_b), b_label,
                ', '.join([str(v) for v in b_trials[absent_from_b]])))
        else:
            rs(f'No {a_label} trials are absent from {b_label}.')

    # --- Steps --- #
    # 1. get protocol, camera, and pressure sensor data WITH per-row occurrence indices
    #    (nothing is dropped here; duplicates are preserved)
    sy_data, protocol_trials, protocol_occ = get_logs_with_occurrences(
        sy_data, sy_column_names, 'protocol')
    sy_ja_data, ja_trials, ja_occ = get_logs_with_occurrences(
        sy_ja_data, sy_ja_column_names, 'camera')
    sy_ps_data, ps_trials, ps_occ = get_logs_with_occurrences(
        sy_ps_data, sy_ps_column_names, 'ps')

    # composite key (trial_num, occurrence) per row, keeping protocol appearance order
    protocol_keys = list(zip(protocol_trials.tolist(), protocol_occ.tolist()))
    ja_keys = list(zip(ja_trials.tolist(), ja_occ.tolist()))
    ps_keys = list(zip(ps_trials.tolist(), ps_occ.tolist()))

    # 2. intersect on the composite key, preserving protocol (recording) order.
    # if and only if ja is not empty -> get common between protocol and camera first,
    # else the common set is based on protocol alone.
    if len(ja_trials) > 0:
        ja_key_set = set(ja_keys)
        common_keys = [k for k in protocol_keys if k in ja_key_set]
        print_absent_trials(protocol_trials, ja_trials, [k[0] for k in common_keys],
                            'protocol', 'camera')
        assert len(common_keys) > 0, (
            f"Found {len(ja_trials)} camera logs but found zero common trials with protocol")
        matched_types = 'main protocol and camera recordings'
    else:
        rs('No camera trials found, searching for pressure sensor logs to match to protocol'
           ' logs...')
        common_keys = list(protocol_keys)
        matched_types = 'main protocol'

    # then intersect with pressure sensors when we have any
    if len(ps_trials) > 0:
        ps_key_set = set(ps_keys)
        common_keys = [k for k in common_keys if k in ps_key_set]
        matched_types = matched_types + ' and pressure sensors'

    print_absent_trials(protocol_trials, ps_trials, [k[0] for k in common_keys], 'protocol',
                        'pressure sensor')

    # NB: do NOT sort by trial number - preserve chronological (recording) order for the neural
    # TTL-pulse<->trial correspondence.
    common_trials = np.array([k[0] for k in common_keys], dtype=int)
    common_dup_index = np.array([k[1] for k in common_keys], dtype=int)

    rs('{} trials are common to {}: {}'.format(
        len(common_keys), matched_types,
        ', '.join('{}{}'.format(t, '' if d == 0 else '_%d' % d) for t, d in common_keys)))

    # recreate each stream's data based on the common composite keys, in the common (protocol)
    # order, so that all data structures refer to the same trials in the same order.
    def gather(data, keys):
        if len(data) == 0:
            return data
        key_to_row = {k: i for i, k in enumerate(keys)}
        idx = [key_to_row[k] for k in common_keys]
        return data[idx, :]

    sy_data = gather(sy_data, protocol_keys)
    sy_ja_data = gather(sy_ja_data, ja_keys)
    sy_ps_data = gather(sy_ps_data, ps_keys)

    return common_trials, common_dup_index, sy_data, sy_ja_data, sy_ps_data


def export_roms_from_osim(osim_filename, o_filename, verbose=False):
    tree = ET.parse(osim_filename)
    root = tree.getroot()

    dof_names = []
    dof_rmin = []
    dof_rmax = []
    for dof_e in root.findall('.//Coordinate'):
        dof_name = dof_e.attrib['name']

        rmin, rmax = [float(i) for i in dof_e.find('range').text.strip().split()]
        if '_tra' not in dof_name:
            rmin *= 180 / np.pi
            rmax *= 180 / np.pi

        if verbose:
            rs('DOF {}: [{}; {}]'.format(dof_name, rmin, rmax))
        dof_names.append(dof_name)
        dof_rmin.append(rmin)
        dof_rmax.append(rmax)

    column_names = ('dof_name', 'range_min', 'range_max', 'rotation')
    values = (dof_names, dof_rmin, dof_rmax, [0 if '_tra' in dn else 1 for dn in dof_names])

    io.export_csv(o_filename, column_names, values)


# only used here because does not support unresolving absolute paths into session-relative
def export_meta_structure(dirname, mstruct):
    filename = os.path.join(dirname, 'meta_structure.json')
    with open(filename, 'w') as f:
        json.dump(mstruct, f, sort_keys=True, indent=4)


# @https://stackoverflow.com/questions/4697006/python-split-string-by-list-of-separators
def split(txt, seps):
    default_sep = seps[0]

    # we skip seps[0] because that's the default separator
    for sep in seps[1:]:
        txt = txt.replace(sep, default_sep)
    return [i.strip() for i in txt.split(default_sep)]


def str2date(s):
    '''Accepts "yy[.-_]mm[.-_]dd"
    No checks.'''
    yy, mm, dd = [int(v) for v in split(s, ('.', '-', '_'))]
    # Jesus has been gone awhile
    if yy < 100:
        yy += 2000
    return datetime.date(yy, mm, dd)


def extract_date(s):
    match = re.search('[0-9]{2,4}[\._\-][0-9]{1,2}[\._\-][0-9]{1,2}', s)
    if match is not None:
        return match[0]
    return None


def find_calibration(session, calibrations_dir):
    candidates = glob.glob(os.path.join(calibrations_dir, '*.*.*_calibration'))

    # remove not matching the numerical format
    candidates = [c for c in candidates if extract_date(c) is not None]
    candidate_dates = [str2date(extract_date(c)) for c in candidates]

    # sort by date
    candidates, candidate_dates = (list(x) for x in zip(*sorted(zip(candidates, candidate_dates),
                                                                key=lambda p: p[1])))
    session_date = str2date(extract_date(session))
    if candidate_dates[0] > session_date:
        ValueError('Could not find calibration before the session date.')
        return None
    for i_c in range(len(candidates) - 1):
        if candidate_dates[i_c + 1] > session_date:
            return candidates[i_c]
    return candidates[-1]


def find_ncams_config(session, calibrations_dir):
    d = find_calibration(session, calibrations_dir)
    if d is None:
        return None
    f = os.path.join(d, 'ncams_config.yaml')
    if not os.path.exists(f):
        ValueError('Calibration folder exists, but the calibration has not been performed'
                   ' in {}.'.format(d))
    return f


def create_session_meta(raw_ss, processed_ss, preset, session, overwrite, export_roms):
    """Create meta information for the given raw session dir and write the meta information to the
    processed session dir.

    Creates the following files
    1. meta_dof.csv
    2. meta_object.csv
    3. meta_session.csv
    4. meta_structure.json

    Args:
        raw_ss (dir): the raw session dir
        processed_ss (dir): the processed session dir
        preset (dict): the preset to use
        session (str): the session name, should exist in the path of procesed_ss and raw_ss
        overwrite (bool): if True overwrite existing meta
        export_roms (bool): if True export range of motion data

    Raises:
        ValueError: raise if no log file is found for session
    """

    # handle meta structure
    # Create output directory if it doesn't exist already
    # processed_ss = os.path.join(preset['processed_server'], session)
    os.makedirs(processed_ss, exist_ok=True)

    # raw_ss = os.path.join(preset['default_server'], session)
    if not os.path.exists(raw_ss):
        raise ValueError('server session {} does not exist on the server.'.format(raw_ss))

    if overwrite or not os.path.exists(os.path.join(processed_ss, 'meta_structure.json')):
        mstruct_rel = meta_session.get_default_meta_structure()
        meta_session.fill_meta_structure(mstruct_rel, raw_ss, session)
        mstruct_rel['ncams_config'] = find_ncams_config(session, CALIBRATIONS_DIR)
        mstruct_rel['hand'] = preset['hand']
        mstruct_rel['ps_dic'] = preset['ps_dic']
        mstruct_rel['fps'] = preset['fps']
        mstruct_rel['object_def_columns'] = preset['object_def_columns']
        if preset['straight_to_video']:
            mstruct_rel['videos_dir'] = mstruct_rel['images_dir']

        export_meta_structure(processed_ss, mstruct_rel)
        rs('Exported meta structure.')
    else:
        rs('Meta structure already exists. Loading...')
    # this one will have resolved paths
    mstruct_path = os.path.join(processed_ss, 'meta_structure.json')
    mstruct = meta_session.import_meta_structure(mstruct_path, raw_ss, processed_ss)

    if len(mstruct['auto_log']) == 0:
        raise ValueError('Session {} does not have an auto log.'.format(session))

    # generate meta session
    if (overwrite or
            not os.path.exists(os.path.join(processed_ss, 'meta_session.csv')) or
            not os.path.exists(os.path.join(processed_ss, 'meta_object.csv'))):
        (sy_column_names, sy_data, sy_ja_column_names, sy_ja_data, sy_ps_column_names, sy_ps_data
         ) = import_logs(raw_ss, mstruct)

        # take subset of data that exists in all logs
        # now all data structure rows refer to the same trials in the same order
        # (nb) cannot merge them together because they have duplicate columns
        common_trials, common_dup_index, sy_data, sy_ja_data, sy_ps_data = reorder_to_common_trials(
            sy_column_names, sy_data, sy_ja_column_names, sy_ja_data, sy_ps_column_names,
            sy_ps_data)

        # find successful ones
        rewarded_trials = (sy_data[:, sy_column_names.index('reward')] > 0.0).astype(int)

        # synchronization period length
        sync_period_length = (
            sy_data[:, sy_column_names.index('log_sent_start_sync_messages(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000

        # NS Added:
        # NS: get the time offset to object in position time
        ttl_to_obj_end_pos = (
            sy_data[:, sy_column_names.index('object_in_position_time(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
        ttl_to_obj_end_pos[
            sy_data[:, sy_column_names.index('object_in_position_time(ms)')] == 0
        ] = math.nan

        # NS: get the time offset to beep (go cue)
        ttl_to_cue = (
            sy_data[:, sy_column_names.index('log_started_monitoring_ps(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
        ttl_to_cue[
            sy_data[:, sy_column_names.index('log_started_monitoring_ps(ms)')] == 0
        ] = math.nan

        # NS: get the time offset to reach
        ttl_to_reach = (
            sy_data[:, sy_column_names.index('arm_liftoff_time(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
        ttl_to_reach[sy_data[:, sy_column_names.index('arm_liftoff_time(ms)')] == 0] = math.nan

        # calculate time offsets (in seconds)
        # values on unsuccessful trials are undefined
        # get the time offset to grasp
        ttl_to_success_grasp = (
            sy_data[:, sy_column_names.index('started_touching_time(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
        ttl_to_success_grasp[np.logical_not(rewarded_trials.astype(bool))] = math.nan

        # Additional per-target touch and force-target-reached offsets (in seconds).
        # Multi-target trials repeat these columns per target (base target, then _2, _3, ...),
        # mirroring the multi-target object-definition columns (e.g. targetForce/targetForce_2).
        # Only the columns actually present in the behaviour log are exported, so single-target
        # sessions are unaffected.
        extra_column_names = []
        extra_values = []

        # touch onset for the 2nd, 3rd, ... targets, mirroring ttl_to_success_grasp above (the
        # first target is already exported as 'ttl_to_success_grasp'). Undefined on unsuccessful
        # trials, same as the first target.
        for suffix in collect_target_suffixes('started_touching_time', '(ms)', sy_column_names):
            if suffix == '':
                continue
            touch_col = 'started_touching_time{}(ms)'.format(suffix)
            ttl_to_touch = (
                sy_data[:, sy_column_names.index(touch_col)] -
                sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
            ttl_to_touch[np.logical_not(rewarded_trials.astype(bool))] = math.nan
            extra_column_names.append('ttl_to_success_grasp{}'.format(suffix))
            extra_values.append(ttl_to_touch)

        # time the force reached its target for each target present (base target, then _2, _3, ...).
        # A value of 0 means the target was never reached on that trial, so mask it as undefined.
        for suffix in collect_target_suffixes('force_target_start_time', '(ms)', sy_column_names):
            force_col = 'force_target_start_time{}(ms)'.format(suffix)
            ttl_to_force_target_start = (
                sy_data[:, sy_column_names.index(force_col)] -
                sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000
            ttl_to_force_target_start[sy_data[:, sy_column_names.index(force_col)] == 0] = math.nan
            extra_column_names.append('ttl_to_force_target_start{}'.format(suffix))
            extra_values.append(ttl_to_force_target_start)

        # NS: get the time offset to end trial/reward
        ttl_to_reward = (
            sy_data[:, sy_column_names.index('trial_end_time(ms)')] -
            sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]) / 1000

        # get camera time offset from TTL to start
        if len(sy_ja_data) > 0:
            ja_ttl_to_rec_start = (
                sy_ja_data[:, sy_ja_column_names.index('startedRecording(ms)')] -
                sy_ja_data[:, sy_ja_column_names.index('syncTrialStartTime(ms)')]) / 1000
        else:
            ja_ttl_to_rec_start = [math.nan] * len(ttl_to_reward)

        # make an object-based meta_object and assign object_id to each trial
        # Only keeps the columns present in the columns
        u_objects, object_ids, object_def_columns = get_object_sets(sy_column_names, sy_data,
                                                                    preset['object_def_columns'])

        # export the meta object information
        meta_object_filename = os.path.join(processed_ss, 'meta_object.csv')
        column_names = ['id'] + list(object_def_columns)
        u_objects_t = list(zip(*u_objects))
        values = [list(range(len(u_objects)))] + u_objects_t
        # always overwritten if meta_session does not exist
        io.export_csv(meta_object_filename, column_names, values)
        rs('Exported session meta object information to {}'.format(meta_object_filename))

        # export the meta session
        meta_session_filename = os.path.join(processed_ss, 'meta_session.csv')
        column_names = ['trial_number',
                        'trial_dup_index',
                        'success',
                        'object_id',
                        'sync_period_length',
                        'ttl_to_obj_end_pos',
                        'ttl_to_cue',
                        'ttl_to_reach',
                        'ttl_to_success_grasp',
                        'ttl_to_reward',
                        'ttl_to_ja_start']
        values = [common_trials,
                  common_dup_index,
                  rewarded_trials,
                  object_ids,
                  sync_period_length,
                  ttl_to_obj_end_pos,
                  ttl_to_cue,
                  ttl_to_reach,
                  ttl_to_success_grasp,
                  ttl_to_reward,
                  ja_ttl_to_rec_start]

        # append the per-target touch and force-target offsets discovered above
        column_names += extra_column_names
        values += extra_values

        io.export_csv(meta_session_filename, column_names, values)
        rs('Exported session meta information to {}'.format(meta_session_filename))

    meta_dof_filename = os.path.join(processed_ss, 'meta_dof.csv')
    if export_roms and (overwrite or not os.path.exists(meta_dof_filename)):
        export_roms_from_osim(ORIGINAL_OPENSIM_MODEL, meta_dof_filename)
        rs('Exported session meta DOF information from {} to {}'.format(
            ORIGINAL_OPENSIM_MODEL, meta_dof_filename))


def create_meta(current_preset, temp, overwrite, export_roms, sessions=[]):
    """Create meta information for
    1. all experimental sessions
    2. all training sessions if preset specifies a raw/processed training server

    Args:
        current_preset (dict): the preset to use
        temp (path): a temporary directory for logging usually C:\tmp
        overwrite (bool): whether to overwrite existing files
        export_roms (bool): whether to export range of motion data
        sessions (list, optional): the session names to process. Defaults to [] (i.e. all sessions)
    """

    # Step 1: process experiment sessions
    apply_to_sessions_helper(current_preset['default_server'],
                             current_preset['processed_server'],
                             current_preset,
                             temp,
                             create_session_meta,
                             args=(overwrite, export_roms),
                             sessions=sessions)

    # Step 2: process training sessions
    if not current_preset['default_training_server']:
        ws('No raw training server specified in preset. Skipping...')
        return

    if not current_preset['processed_training_server']:
        ws('No processed training server specified in preset. Skipping...')
        return

    # Step 2: Process training sessions
    apply_to_sessions_helper(current_preset['default_training_server'],
                             current_preset['processed_training_server'],
                             current_preset,
                             temp,
                             create_session_meta,
                             args=(overwrite, export_roms),
                             sessions=sessions)
