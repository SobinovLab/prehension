#!python3
# -*- coding: utf-8 -*-
"""
Uses NCams to triangulate 2D points into 3D marker positions.

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
import argparse
import datetime
import time

from prehension import preset
from prehension.tools import cmd_args
from prehension.kinematics.triangulate import run_triangulate
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Triangulates marker positions from 2D to 3D and creates inverse kinematics '
                     'files.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
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

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
