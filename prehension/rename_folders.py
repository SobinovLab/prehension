#!python3.7
import os

import tqdm

from . import meta_session
from . import tools
from .tools import rs, ws

RENAMES = {
    'processed_sensors': 'transformed_sensors',
    'processed_joint_angles_aligned': 'aligned_joint_angles',
    'processed_sensors_aligned': 'aligned_sensors_old_csv'
}


def rename_folders(server, sessions, temp):
    """Changes the names of some folders in each session.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        temp {str} --- Folder for local temporary storage.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        for src, dst in RENAMES.items():
            src_f = os.path.join(server_session, src)
            if os.path.exists(src_f):
                dst_f = os.path.join(server_session, dst)
                rs('\t{} -> {}'.format(src_f, dst_f))
                if os.path.exists(dst_f):
                    ws('Destination folder already exists. Consider deleting source: {}'.format(
                        src_f))
                else:
                    os.rename(src_f, dst_f)
