#!python3.7
import os
import argparse
import time
import datetime
import tqdm

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import meta_session

RENAMES = {
    'processed_sensors': 'transformed_sensors',
    'processed_joint_angles_aligned': 'aligned_joint_angles',
    'processed_sensors_aligned': 'aligned_sensors_old_csv'
}

def main(server, sessions, temp):
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


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Changes the names of some folders in each session.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.temp)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
