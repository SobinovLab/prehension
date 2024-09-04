#!python3.8
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.inverse_kinematics import inverse_kinematics


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Runs the inverse kinematics OpenSim tool.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    parser.add_argument(
        '--base',
        action='store_true',
        help='Runs inverse kinematics on the most proximal markers that can be used to estimate '
        'the default static thorax position.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    inverse_kinematics(
        args.server, args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.base)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
