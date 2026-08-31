#!python3
# -*- coding: utf-8 -*-
"""
Uploading data to a shared server.

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
import os
import time

import tqdm

from . import meta_session
from .tools import logs
from .tools.logs import rs, ws
from .tools import filesystem

# We want to upload the following dirs and folders
GENERAL_RAW_DIRS = (
    'mujoco_models',
    'opensim_models',
)

GENERAL_RAW_FILES = ()

GENERAL_PROC_DIRS = (
    'mujoco_models',
    'opensim_models',
)

GENERAL_PROC_FILES = ()

# We want to upload the following dirs and folder PER session directory
SESSION_RAW_DIRS = (
    'behavior',
    # 'neural_processed_nwb',
)

SESSION_RAW_FILES = (
)


SESSION_PROC_DIRS = (
    'aligned_joint_angles',
    'digit_forces',
    'filtered_sensors',
    'markers_3D',
    'matched_contacts',
    'segment_forces',
)

SESSION_PROC_FILES = (
    'meta_dof.csv',
    'meta_object.csv',
    'meta_session.csv',
    'meta_structure.json',
    'meta_neural.json',
    'timepoints.csv',
    'neural_processed/neural.nwb'
)


def upload_data(current_preset, sessions, temp, target_dir, dry_run, overwrite):
    """Uploads the data from local server to server accessible to collaborators.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        temp {str} --- Folder for local temporary storage.
        target_dir {str} --- Where to upload the data.
        dry_run {bool} --- Do not copy the data, only print out the files to be copied.
        overwrite {bool} --- Overwrites the created files if they exist.
    """
    rserv = current_preset['default_server']
    pserv = current_preset['processed_server']
    logs.setup_logging(temp, sessions_dir=pserv)

    if not os.path.exists(rserv):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(rserv))

    monkey_dir = current_preset['share_upload_name']
    rs('Identified monkey directory {}.'.format(monkey_dir))
    target_dir = os.path.join(target_dir, monkey_dir)

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(rserv)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    start_time = time.time()
    copy_function = filesystem.PrintCopyAccumulateSize(dry_run, 1)

    ## upload general monkey stuff like models
    if len(GENERAL_RAW_DIRS) > 0 or len(GENERAL_RAW_FILES) > 0:
        filesystem.copy_folder_contents(
            rserv, target_dir,
            dir_names=GENERAL_RAW_DIRS, file_names=GENERAL_RAW_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)
    if len(GENERAL_PROC_DIRS) > 0 or len(GENERAL_PROC_FILES) > 0:
        filesystem.copy_folder_contents(
            pserv, target_dir,
            dir_names=GENERAL_PROC_DIRS, file_names=GENERAL_PROC_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        raw_ss = os.path.join(rserv, session)
        proc_ss = os.path.join(pserv, session)

        if not os.path.exists(raw_ss):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        filesystem.copy_folder_contents(
            raw_ss, os.path.join(target_dir, session),
            dir_names=SESSION_RAW_DIRS, file_names=SESSION_RAW_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)
        filesystem.copy_folder_contents(
            proc_ss, os.path.join(target_dir, session),
            dir_names=SESSION_PROC_DIRS, file_names=SESSION_PROC_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)

    timedelta = str(datetime.timedelta(seconds=time.time() - start_time))
    if dry_run:
        rs('In total, found {} files for copying. Search took {}.'.format(
            copy_function, timedelta))
    else:
        rs('In total, copied {} files over {}.'.format(copy_function, timedelta))
