#!python3
# -*- coding: utf-8 -*-
"""
Uploading data to a shared server.

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
import os
import time

import tqdm

from . import meta_session
from .tools import logs
from .tools.logs import rs, ws
from .tools import filesystem

# We want to upload the following dirs and folders
GENERAL_DIRS = (
    'mujoco_models',
    'opensim_models',
)

GENERAL_FILES = ()

# We want to upload the following dirs and folder PER session directory
SESSION_DIRS = (
    'aligned_joint_angles',
    'behavior',
    'digit_forces',
    'filtered_sensors',
    'markers_3D',
    'matched_contacts',
    'neural_processed_nwb',
    'segment_forces',
)

SESSION_FILES = (
    'meta_dof.csv',
    'meta_object.csv',
    'meta_session.csv',
    'meta_structure.json',
    'timepoints.csv'
)


def upload_data(server, sessions, temp, target_dir, dry_run, overwrite):
    """Uploads the data from local server to server accessible to collaborators.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        temp {str} --- Folder for local temporary storage.
        target_dir {str} --- Where to upload the data.
        dry_run {bool} --- Do not copy the data, only print out the files to be copied.
        overwrite {bool} --- Overwrites the created files if they exist.
    """
    logs.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    monkey_dir = os.path.basename(os.path.dirname(server))
    rs('Identified monkey directory {}.'.format(monkey_dir))
    target_dir = os.path.join(target_dir, monkey_dir)

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(
        len(sessions), ', '.join(sessions)))

    start_time = time.time()
    copy_function = filesystem.PrintCopyAccumulateSize(dry_run, 1)

    ## upload general monkey stuff like models
    filesystem.copy_folder_contents(
        server, target_dir,
        dir_names=GENERAL_DIRS, file_names=GENERAL_FILES, copy_function=copy_function,
        overwrite=overwrite, box=True)

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        filesystem.copy_folder_contents(
            server_session, os.path.join(target_dir, session),
            dir_names=SESSION_DIRS, file_names=SESSION_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)

    timedelta = str(datetime.timedelta(
        seconds=time.time() - start_time))
    if dry_run:
        rs('In total, found {} files for copying. Search took {}.'.format(
            copy_function, timedelta))
    else:
        rs('In total, copied {} files over {}.'.format(
            copy_function, timedelta))
