#!python3
# -*- coding: utf-8 -*-
"""
Functions related to sessions from a dataset.

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
import glob
import json
import os
import re
import warnings

from .tools.io import import_csv, export_csv
from .trial_info import TrialInfo


def find_session_dirs(dirname):
    sessions = []
    session_re = re.compile('[0-9]{4}_[0-9]{2}_[0-9]{2}.*')
    # process the directory
    for d in os.listdir(dirname):
        if session_re.fullmatch(d) is not None:
            sessions.append(d)
    return sessions


def get_default_meta_structure():
    '''Returns a default structure for a session.

    Recommend filling it with relevant:
        ps_dic
        ps_markers
    session-specific fields:
        opensim_model_locked_base
        mujoco_model_sensorized
        ncams_config

    CHECK:
        hand (right or left)

    The following can be filled in automatically using `fill_meta_structure()`:
        auto_log
        manual_log
        opensim_model_locked_base
        mujoco_model_sensorized
        cameras

    These are filled by preset in create_meta:
        ps_dic
        ps_markers
        hand
        fps
    '''
    return {
        'auto_log': '',
        'manual_log': '',
        'videos_dir': 'camera_videos',
        'images_dir': 'cameras',
        'cameras': {},
        'timepoint_plots_dir': 'timepoint_plots',
        'timepoint_csv_filename': 'timepoints.csv',
        'markers_2D_dir': 'markers_2D',
        'markers_2D_video_dir': 'markers_2D_videos',
        'markers_3D_dir': 'markers_3D',
        'markers_3D_jarvis_dir': 'markers_3D_jarvis',
        'jarvis_video_dir': 'jarvis_videos',
        'pre_ja_dir': 'joint_angles',
        'post_ja_dir': 'aligned_joint_angles',
        'raw_ps_dir': 'sensors',
        'transformed_ps_dir': 'transformed_sensors',
        'pre_ps_dir': 'filtered_sensors',
        'post_ps_dir': 'aligned_sensors',
        'matched_contacts_dir': 'matched_contacts',
        'manually_labelled_forces_dir': 'manually_labelled',
        'scaling_dir': 'scaling',
        'digit_forces_dir': 'digit_forces',
        'segment_forces_dir': 'segment_forces',
        'mujoco_videos_dir': 'mujoco_videos',
        'ps_dic': {},
        'ps_trialname_template': '{ps_serial}_{trial_number}',
        'kin_trialname_template': 'trial{trial_number}',
        'kin_adjustment_suffix': '_adjust',
        'ps_log_filename': 'sensors/trial_log.csv',
        'opensim_model': '../opensim_models/RightArmAndHand_NoMuscles_Scaled.osim',
        'opensim_model_locked_base': '',  # session-specific
        'mujoco_model': '../mujoco_models/RightArmAndHand_NoMuscles_Scaled.xml',
        'mujoco_model_sensorized': '',  # session-specific
        'ncams_config': '',  # session-specific
        'calibration': 'calibration',  # local calibration directory
        'hand': 'right',
        'fps': 50,  # fill in
        'ps_markers': {
            'medial_sensor': ('o_sensor_tb', 'o_sensor_tf', 'o_sensor_bb', 'o_sensor_bf'),
            'lateral_sensor': ('b_sensor_tb', 'b_sensor_tf', 'b_sensor_bb', 'b_sensor_bf')
        },
        'object_def_columns': (
            'pos_translation_z(mm)', 'pos_tilt(deg)', 'pos_aperture(mm)')
    }


# meta_session.fill_meta_structure(mstruct_rel, raw_dir, session)
def fill_meta_structure(mstruct, raw_dir, processed_dir, session, log_rel_dir='behavior'):
    '''If the meta structure dictionary has empty auto_log and manual_log, script searches for them.
    Replacement is done in place.

    Also fills mujoco and opensim models, and identifies cameras
    '''

    # Search auto log
    if len(mstruct['auto_log']) == 0:
        # search automatically
        auto_log = glob.glob(os.path.join(
            raw_dir, log_rel_dir, 'session_*.csv'))
        if len(auto_log) > 1:
            # sort them
            def order(v):
                if re.match('.*\([0-9]+\).csv$', v) is not None:
                    return int(re.findall('\([0-9]+\)', v)[-1][1:-1])
                return -1

            auto_log.sort(key=order)

            warnings.warn('Several session log filenames found: {}'.format(auto_log))
        elif len(auto_log) == 0:
            raise ValueError('Could not find auto log session filenames in {}.'.format(raw_dir))

        mstruct['auto_log'] = auto_log  ### SWITCH TO FULL PATH

    # Search manual log
    if len(mstruct['manual_log']) == 0:
        # search automatically
        manual_log = glob.glob(os.path.join(
            raw_dir, log_rel_dir, '*Daily experiment log - trials*.csv'))
        if len(manual_log) == 0:
            manual_log = glob.glob(os.path.join(
                raw_dir, log_rel_dir, 'Manual_*.csv'))
        if len(manual_log) > 1:
            warnings.warn(
                'Too many manual session log filenames found: {}. Using first one.'.format(
                    manual_log))
            mstruct['manual_log'] = os.path.join(log_rel_dir, os.path.basename(manual_log[0]))
        elif len(manual_log) == 0:
            warnings.warn('Could not find manual session log filenames in {}.'.format(log_rel_dir))
        else:
            mstruct['manual_log'] = os.path.join(log_rel_dir, os.path.basename(manual_log[0]))

    mstruct['opensim_model_locked_base'] = '{}_locked_{}.osim'.format(
        mstruct['opensim_model'][:-5], session)
    mstruct['mujoco_model_sensorized'] = '{}_Tessellated_{}.xml'.format(
        mstruct['mujoco_model'][:-4], session)

    # identify cameras
    if os.path.exists(os.path.join(raw_dir, mstruct['videos_dir'])):
        # TODO suboptimal
        _cameras = glob.glob(os.path.join(raw_dir, mstruct['videos_dir'], 'trial*', 'cam*.mp4'))
        _cameras_dict = {}
        for _c in _cameras:
            camera = os.path.split(_c)[1]
            try:
                serial = int(camera[3:-4])
            except Exception as e:
                continue
            _cameras_dict[serial] = camera[:-4]
    else:
        _cameras = glob.glob(os.path.join(raw_dir, mstruct['images_dir'], 'cam*'))
        _cameras_dict = {}
        for _c in _cameras:
            camera = os.path.split(_c)[1]
            try:
                serial = int(camera[3:])
            except Exception as e:
                continue
            _cameras_dict[serial] = camera
    mstruct['cameras'] = _cameras_dict


def normjoinpath(dirname, p):
    if len(p) == 0:
        return None
    return os.path.normpath(os.path.join(dirname, p))


def import_meta_structure(raw_dir, proc_dir):
    filename = os.path.join(proc_dir, 'meta_structure.json')
    with open(filename, 'r') as f:
        mstruct = json.load(f)

    # resolve relative paths
    # on processed server
    pth_2_resolve_proc = (
        'timepoint_plots_dir', 'timepoint_csv_filename',
        'markers_2D_dir', 'markers_2D_video_dir', 'markers_3D_dir', 'markers_3D_jarvis_dir',
        'jarvis_video_dir',
        'pre_ja_dir', 'post_ja_dir',
        'transformed_ps_dir', 'pre_ps_dir', 'post_ps_dir',
        'matched_contacts_dir', 'manually_labelled_forces_dir', 'scaling_dir',
        'digit_forces_dir', 'segment_forces_dir',
        'mujoco_videos_dir'
    )

    pth_2_resolve_raw = (
        'auto_log', 'manual_log', 'ps_log_filename',
        'videos_dir', 'images_dir',
        'opensim_model', 'opensim_model_locked_base',
        'mujoco_model', 'mujoco_model_sensorized',
        'calibration', 'raw_ps_dir',
    )

    def add_key(ptr, d):
        if ptr not in mstruct.keys():
            return
        # in case one has multiple elements
        if isinstance(mstruct[ptr], (list, tuple)):
            mstruct[ptr] = [normjoinpath(d, p) for p in mstruct[ptr]]
        else:
            mstruct[ptr] = normjoinpath(d, mstruct[ptr])

    for ptr in pth_2_resolve_proc:
        add_key(ptr, proc_dir)
    for ptr in pth_2_resolve_raw:
        if ptr in pth_2_resolve_proc:
            raise Exception(f'Attempting to resolve path for {ptr} twice.')
        add_key(ptr, raw_dir)

    ###############################################################################################
    # HACK FOR saving videos
    ###############################################################################################
    # should only work for *_camera_copy presets
    # mojito lhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Spring_2021',
        'Recording_sessions', 'Mojito')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'MojitoLeftHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    # pimms rhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Pimms',
        'RightHem_Recordings')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'PimmsRightHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    # mojito rhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Mojito',
        'RightHem_Recordings')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'MojitoRightHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    return mstruct


def import_meta_object(dirname):
    filename = os.path.join(dirname, 'meta_object.csv')
    column_names, values = import_csv(filename)
    object_ids = values[column_names.index('id')]
    object_def_columns = [v for v in column_names if v != 'id']

    # make a dictionary of all objects
    answ = {}
    for i_object, object_id in enumerate(object_ids):
        answ[object_id] = {'def': {}}
        for odc in object_def_columns:
            answ[object_id]['def'][odc] = values[column_names.index(odc)][i_object]
        answ[object_id]['sstr'] = ' '.join(
            str(v) for v in answ[object_id]['def'].values())
        answ[object_id]['str'] = ', '.join(
            '{}: {}'.format(k, v) for k, v in answ[object_id]['def'].items())
    return answ


def import_meta_dof(dirname):
    filename = os.path.join(dirname, 'meta_dof.csv')
    column_names, values = import_csv(filename)

    i_dofname = column_names.index('dof_name')
    i_rmin = column_names.index('range_min')
    i_rmax = column_names.index('range_max')
    i_rot = column_names.index('rotation')

    mdof = {name: {'range': [rmin, rmax],
                   'rot': rot
                   }
            for name, rmin, rmax, rot
            in zip(values[i_dofname], values[i_rmin], values[i_rmax], values[i_rot])}
    return mdof


def import_manual_log(filename):
    column_names, values = import_csv(filename, cast=str)
    mlog = {int(trial_number): code.split(',')
            for trial_number, code in zip(values[column_names.index('Trial')],
                                          values[column_names.index('Code')])}
    return mlog


def _column_pop(k, column_names, values):
    i_k = column_names.index(k)
    answ = values[i_k]
    del column_names[i_k]
    del values[i_k]
    return answ


def load_meta_information(raw_dir, proc_dir, only_successful_trials=False,
                          check_manual_log=False, session=None):
    # find the session name if it was None
    if session is None:
        session = os.path.basename(raw_dir)

    mstruct = import_meta_structure(raw_dir, proc_dir)
    mdof = import_meta_dof(proc_dir)
    mobject = import_meta_object(proc_dir)

    # meta session
    meta_session_filename = os.path.join(proc_dir, 'meta_session.csv')
    column_names, values = import_csv(meta_session_filename)

    # essential trial parameters
    trial_numbers = _column_pop('trial_number', column_names, values)
    successs = _column_pop('success', column_names, values)
    object_ids = _column_pop('object_id', column_names, values)

    # load manual log
    if check_manual_log:
        if mstruct['manual_log'] is None:
            warnings.warn('No manual log specified in session structure, cannot check it.')
            check_manual_log = False
        else:
            try:
                mlog = import_manual_log(mstruct['manual_log'])
                # total fail or multigrasp or multireach or ?
                mlog_failed_numbers = ['0', '2', '3', '4', '5', '?', '14']
            except Exception as e:
                warnings.warn('Could not load manual log: {}'.format(repr(e)))
                check_manual_log = False

    msession = []
    for i_trial, (trial_number, success, object_id) in enumerate(zip(
            trial_numbers, successs, object_ids)):
        if only_successful_trials and not success:
            continue
        if only_successful_trials and check_manual_log:
            if any([mfn in mlog[trial_number] for mfn in mlog_failed_numbers]):
                continue
        msession.append(TrialInfo(session, trial_number, object_id, success,
                                  other_info={k: v[i_trial] for k, v in zip(column_names, values)}))
        msession[-1].generate_filenames(mstruct)

    return mstruct, mdof, mobject, msession


def import_adjustment_trials(dirname):
    if not os.path.exists(os.path.join(dirname, 'adjustment_files.csv')):
        return {}

    column_names, values = import_csv(os.path.join(dirname, 'adjustment_files.csv'))

    trial_numbers = [int(v) for v in values[column_names.index('trial_number')]]
    adjustment_trials = [int(v) for v in values[column_names.index('adjustment_trial')]]

    return {k: v for k, v in zip(trial_numbers, adjustment_trials)}


def find_trial(msession, trial_number):
    trial = None
    for t in msession:
        if t.trial_number == trial_number:
            trial = t
            break
    return trial


def get_trial_log_info(mstruct, trial_number, column_names):
    if not isinstance(column_names, (list, tuple)):
        column_names = [column_names]

    sy_column_names, sy_data = import_csv(mstruct['auto_log'][0])
    # sy_data = np.array(sy_data).transpose()

    # TODO check if the trial not in the list
    row = sy_data[sy_column_names.index('trial_num')].index(trial_number)

    column_ids = [sy_column_names.index(cn) for cn in column_names]

    return [sy_data[ci][row] for ci in column_ids]


def export_optimal_frames(filename, trial_numbers, optimal_frames):
    column_names = ['trial_number', 'optimal_frame']
    values = [trial_numbers, optimal_frames]

    export_csv(filename, column_names, values)


def import_optimal_frames(filename):
    column_names, values = import_csv(filename)

    trial_numbers = [int(v) for v in values[column_names.index('trial_number')]]
    optimal_frames = [int(v) for v in values[column_names.index('optimal_frame')]]

    return {k: v for k, v in zip(trial_numbers, optimal_frames)}
