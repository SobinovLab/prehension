#!python3.7
import os
import sys
import re
import inspect
import argparse
import uuid
import warnings
import copy
import time
import datetime
import random
import tqdm

# include local library functions - TB included in NCams
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common import preset
from common import tools
from common.tools import rs, ws
from common import io_tools
from common import meta_session

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


def main(server, sessions, temp, target_dir, dry_run, overwrite):
    tools.setup_logging(temp, sessions_dir=server)

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
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    start_time = time.time()
    copy_function = tools.PrintCopyAccumulateSize(dry_run, 1)

    ## upload general monkey stuff like models
    tools.copy_folder_contents(
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

        tools.copy_folder_contents(
            server_session, os.path.join(target_dir, session),
            dir_names=SESSION_DIRS, file_names=SESSION_FILES, copy_function=copy_function,
            overwrite=overwrite, box=True)

    timedelta = str(datetime.timedelta(seconds=time.time() - start_time))
    if dry_run:
        rs('In total, found {} files for copying. Search took {}.'.format(
            copy_function, timedelta))
    else:
        rs('In total, copied {} files over {}.'.format(copy_function, timedelta))


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uploads the data from local server to server accessible to collaborators.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    default_target_dir = os.path.join(os.environ['USERPROFILE'], 'Box', 'PrehensionProject')
    parser.add_argument(
        '--target_dir',
        type=str, default=default_target_dir,
        help='Where to upload the data. Default: {}'.format(default_target_dir))

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Do not copy the data, only print out the files to be copied.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.temp, args.target_dir, args.dry_run, args.overwrite)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
