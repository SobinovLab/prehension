#!python3
# -*- coding: utf-8 -*-
"""
Performs camera calibrations for each session.

TODO Currently uses NCams calibrations, need to use Jarvis approach in the future.

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
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension.tools import cmd_args
from prehension.kinematics.ncams_3d import calibration
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Copies session extrinsic calibration images into their own directory and '
                     'runs extrinsic calibration for each session.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
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

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    # show the resulting calibrations
    if args.run_extrinsic_calibration:
        plt.show()
