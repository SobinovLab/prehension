#!python3
# -*- coding: utf-8 -*-
"""
Runs OpenSim inverse kinematics on the specified sessions and trials.

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

from prehension import preset
from prehension.tools import cmd_args
from prehension.kinematics.inverse_kinematics import inverse_kinematics
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Runs the inverse kinematics OpenSim tool.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
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

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
