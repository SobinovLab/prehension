#!python3
# -*- coding: utf-8 -*-
"""
Creates meta information files for experimental sessions.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/BensmaiaLab/prehension

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
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import tqdm

from .tools.constants import ORIGINAL_OPENSIM_MODEL, CALIBRATIONS_DIR
from .tools import io
from .tools import logs
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
    trial_nums = sy_data[:, sy_column_names.index('trial_num')].astype(int)
    if len(set(trial_nums)) != len(trial_nums):
        ws(
            'Auto logs in {} have duplicating trials!'.format(dirname)
        )  # NOT FATAL BECAUSE WE HANDLE LATER
        # raise ValueError('Auto logs in {} have duplicating trials!'.format(dirname))

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


def reorder_to_common_trials(sy_column_names, sy_data, sy_ja_column_names, sy_ja_data,    sy_ps_column_names, sy_ps_data):
    '''
    This is equivalent to reorder_to_common_trials but instead of the common trials = (protocol U
    camera) U sensor, we exclude the camera overlap if camera logs are empty
    We also simplify checking for duplicate logs.
    '''

    # --- helpers --- #
    def get_logs_and_check_duplicates(data, data_columns, log_label):
        if 'trial_num' in data_columns:
            trials = data[:, data_columns.index('trial_num')].astype(int)

            last_occurrence = {}
            duplicate_indices = {}

            for index, trial in enumerate(trials):
                # If the trial number is already in the last_occurrence dictionary
                # it means it's a duplicate
                if trial in last_occurrence:
                    # Store the current index as a duplicate index for this trial number
                    duplicate_indices.setdefault(trial, []).append(index)
                # Add entry to dict where key is the trial in question and value is the last valid index
                last_occurrence[trial] = index

            # Include if we want more verbosity at some point
            # for trial, indices in duplicate_indices.items():
            #     ws(f"Excluding duplicate trials from {log_label}:
            # \n{trial}, Duplicate indices: {indices}")

            # Use list comprehension to filter rows based on the last occurrence of each trial number
            filtered_data_indices = [last_occurrence[trial] for trial in np.unique(trials)]

            if len(filtered_data_indices) != len(data):
                ws(
                    f'''Shortening log data (from {log_label})
                    by {len(data) - len(filtered_data_indices)} to remove duplicates'''
                )

            data_filtered = np.array([data[index] for index in filtered_data_indices])
            trials_filtered = [trials[index] for index in filtered_data_indices]

            return data_filtered, trials_filtered

        # Case where we don't have any incoming data (if no cam logs are found for example)
        return np.array([]), np.array([])

    def print_absent_trials(a_trials, b_trials, common, a_label, b_label):
        # In a but not b
        a_trials = np.array(a_trials)
        b_trials = np.array(b_trials)

        absent_from_a = np.isin(a_trials, common, invert=True)
        if sum(absent_from_a) > 0:
            rs(
                '{} {} trials are absent from {}: {}'.format(
                    sum(absent_from_a),
                    b_label,
                    a_label,
                    ', '.join([str(v) for v in a_trials[absent_from_a]]),
                )
            )
        else:
            rs(f'No {b_label} trials are absent from {a_label}.')

        # In b but not a
        absent_from_b = np.isin(b_trials, common, invert=True)

        if sum(absent_from_b) > 0:
            rs(
                '{} trials are absent from {}: {}'.format(
                    sum(absent_from_b),
                    b_label,
                    ', '.join([str(v) for v in b_trials[absent_from_b]]),
                )
            )
        else:
            rs(f'No {a_label} trials are absent from {b_label}.')

    # --- Steps --- #
    # 1. get protocol, camera, and pressure sensor data
    # AND filter out duplicates
    sy_data, protocol_trials = get_logs_and_check_duplicates(sy_data, sy_column_names, 'protocol')
    sy_ja_data, ja_trials = get_logs_and_check_duplicates(sy_ja_data, sy_ja_column_names, 'camera')
    sy_ps_data, ps_trials = get_logs_and_check_duplicates(sy_ps_data, sy_ps_column_names, 'ps')

    # 2. if and only if ps_trials is not empty:
    # -> get common between protocol and camera
    # else: common_between protocol and camera is just protocol
    if len(ja_trials) > 0:
        common_prot_cam = np.intersect1d(protocol_trials, ja_trials)
        print_absent_trials(protocol_trials, ja_trials, common_prot_cam, 'protocol', 'camera')
        assert (
            len(common_prot_cam) > 0
        ), f"Found {len(ja_trials)} camera logs but found zero common trials with protocol"

        # Common between all (protocol, camera, ps)
        common_trials_all = np.intersect1d(common_prot_cam, ps_trials)
        matched_types = 'main protocol, camera recordings, and pressure sensors'

    else:
        rs(
            'No camera trials found, searching for pressure sensor logs to match to protocol'
            ' logs...'
        )
        # Common between protocol and ps only since we lack any camera logs
        common_trials_all = np.intersect1d(protocol_trials, ps_trials)
        matched_types = 'main protocol and pressure sensors'

    print_absent_trials(
        protocol_trials, ps_trials, common_trials_all, 'protocol', 'pressure sensor'
    )

    common_trials_all.sort()

    rs(
        '{} trials are common to {}: {}'.format(
            len(common_trials_all), matched_types, ', '.join(str(v) for v in common_trials_all)
        )
    )

    # recreate data based on the common trials
    sy_data = sy_data[np.isin(protocol_trials, common_trials_all), :]
    # resort them by the trial_number just in case
    sy_data[sy_data[:, sy_column_names.index('trial_num')].argsort()]

    # Check that we actually have camera data to refine
    if len(sy_ja_data) > 0:
        sy_ja_data = sy_ja_data[np.isin(ja_trials, common_trials_all), :]
        sy_ja_data[sy_ja_data[:, sy_ja_column_names.index('trial_num')].argsort()]

    # Check that we actually have pressure data to refine
    if len(sy_ja_data) > 0:
        sy_ps_data = sy_ps_data[np.isin(ps_trials, common_trials_all), :]
        sy_ps_data[sy_ps_data[:, sy_ps_column_names.index('trial_num')].argsort()]

    return common_trials_all, sy_data, sy_ja_data, sy_ps_data


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
    candidates, candidate_dates = (
        list(x) for x in zip(*sorted(zip(candidates, candidate_dates), key=lambda p: p[1]))
    )
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
        ValueError(
            'Calibration folder exists, but the calibration has not been performed'
            ' in {}.'.format(d)
        )
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
    assert os.path.exists(raw_ss), 'server session {} does not exist on the server.'.format(raw_ss)

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
    if (
        overwrite
        or not os.path.exists(os.path.join(processed_ss, 'meta_session.csv'))
        or not os.path.exists(os.path.join(processed_ss, 'meta_object.csv'))
    ):
        (
            sy_column_names,
            sy_data,
            sy_ja_column_names,
            sy_ja_data,
            sy_ps_column_names,
            sy_ps_data,
        ) = import_logs(raw_ss, mstruct)

        # take subset of data that exists in all logs
        # now all data structure rows refer to the same trials in the same order
        # (nb) cannot merge them together because they have duplicate columns
        common_trials, sy_data, sy_ja_data, sy_ps_data = reorder_to_common_trials(
            sy_column_names, sy_data, sy_ja_column_names, sy_ja_data, sy_ps_column_names, sy_ps_data
        )

        # find successful ones
        rewarded_trials = (sy_data[:, sy_column_names.index('reward')] > 0.0).astype(int)

        # synchronization period length
        sync_period_length = (
            sy_data[:, sy_column_names.index('log_sent_start_sync_messages(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000

        # NS Added:
        # NS: get the time offset to object in position time
        ttl_to_obj_end_pos = (
            sy_data[:, sy_column_names.index('object_in_position_time(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000
        ttl_to_obj_end_pos[
            sy_data[:, sy_column_names.index('object_in_position_time(ms)')] == 0
        ] = math.nan

        # NS: get the time offset to beep (go cue)
        ttl_to_cue = (
            sy_data[:, sy_column_names.index('log_started_monitoring_ps(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000
        ttl_to_cue[
            sy_data[:, sy_column_names.index('log_started_monitoring_ps(ms)')] == 0
        ] = math.nan

        # NS: get the time offset to reach
        ttl_to_reach = (
            sy_data[:, sy_column_names.index('arm_liftoff_time(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000
        ttl_to_reach[sy_data[:, sy_column_names.index('arm_liftoff_time(ms)')] == 0] = math.nan

        # calculate time offsets (in seconds)
        # values on unsuccessful trials are undefined
        # get the time offset to grasp
        ttl_to_success_grasp = (
            sy_data[:, sy_column_names.index('started_touching_time(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000
        ttl_to_success_grasp[np.logical_not(rewarded_trials.astype(bool))] = math.nan

        # NS: get the time offset to end trial/reward
        ttl_to_reward = (
            sy_data[:, sy_column_names.index('trial_end_time(ms)')]
            - sy_data[:, sy_column_names.index('log_started_ephys_recording(ms)')]
        ) / 1000

        # get camera time offset from TTL to start
        if len(sy_ja_data) > 0:
            ja_ttl_to_rec_start = (
                sy_ja_data[:, sy_ja_column_names.index('startedRecording(ms)')]
                - sy_ja_data[:, sy_ja_column_names.index('syncTrialStartTime(ms)')]
            ) / 1000
        else:
            ja_ttl_to_rec_start = [math.nan] * len(ttl_to_reward)

        # make an object-based meta_object and assign object_id to each trial
        # Only keeps the columns present in the columns
        u_objects, object_ids, object_def_columns = get_object_sets(
            sy_column_names, sy_data, preset['object_def_columns']
        )

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
        column_names = [
            'trial_number',
            'success',
            'object_id',
            'sync_period_length',
            'ttl_to_obj_end_pos',
            'ttl_to_cue',
            'ttl_to_reach',
            'ttl_to_success_grasp',
            'ttl_to_reward',
            'ttl_to_ja_start',
        ]
        values = [
            common_trials,
            rewarded_trials,
            object_ids,
            sync_period_length,
            ttl_to_obj_end_pos,
            ttl_to_cue,
            ttl_to_reach,
            ttl_to_success_grasp,
            ttl_to_reward,
            ja_ttl_to_rec_start,
        ]
        io.export_csv(meta_session_filename, column_names, values)
        rs('Exported session meta information to {}'.format(meta_session_filename))

    meta_dof_filename = os.path.join(processed_ss, 'meta_dof.csv')
    if export_roms and (overwrite or not os.path.exists(meta_dof_filename)):
        export_roms_from_osim(ORIGINAL_OPENSIM_MODEL, meta_dof_filename)
        rs(
            'Exported session meta DOF information from {} to {}'.format(
                ORIGINAL_OPENSIM_MODEL, meta_dof_filename
            )
        )


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
    apply_to_sessions_helper(
        current_preset['default_server'],
        current_preset['processed_server'],
        current_preset,
        temp,
        create_session_meta,
        args=(overwrite, export_roms),
        sessions=sessions,
    )

    # Step 2: process training sessions
    if not current_preset['default_training_server']:
        ws('No raw training server specified in preset. Skipping...')
        return

    if not current_preset['processed_training_server']:
        ws('No processed training server specified in preset. Skipping...')
        return

    # Step 2: Process training sessions
    apply_to_sessions_helper(
        current_preset['default_training_server'],
        current_preset['processed_training_server'],
        current_preset,
        temp,
        create_session_meta,
        args=(overwrite, export_roms),
        sessions=sessions,
    )
