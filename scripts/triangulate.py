#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.kinematics.triangulate import run_triangulate


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Triangulates marker positions from 2D to 3D and creates inverse kinematics '
                     'files.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    parser.add_argument(
        '--threshold',
        type=float, default=0.4,
        help='Threshold for likelihood of 3D points to be used for triangulation.')
    parser.add_argument(
        '--dont_triangulate',
        action='store_false', dest='do_triangulate',
        help='If specified, triangulation itself will be skipped, but the supporting files will'
        ' be generated.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    run_triangulate(
        args.server, args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.threshold, args.do_triangulate)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
