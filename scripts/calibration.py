#!python3.7
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension import tools
from prehension.calibration import calibration


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Copies session extrinsic calibration images into their own directory and '
                     'runs extrinsic calibration for each session.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    parser.add_argument(
        '--relocate',
        action='store_true',
        help='Copies extrinsic calibration from "cameras" folder into "calibration/extrinsic"'
        ' directory.')
    parser.add_argument(
        '--run_extrinsic_calibration',
        action='store_true',
        help='Runs local extrinsic calibration in the session directory.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    calibration(args.server, args.sessions, args.temp, args.overwrite, args.relocate,
                args.run_extrinsic_calibration)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    # show the resulting calibrations
    if args.run_extrinsic_calibration:
        plt.show()
