#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.kinematics.compress_session_cameras import compress_session_cameras
from prehension.tools import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Transforms images from a session into video.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    parser.add_argument(
        '--clean',
        action='store_true',
        help='DANGER! Remove directories from the server that were converted into videos.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    compress_session_cameras(
        args.server, args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.clean)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
