#!python3
# -*- coding: utf-8 -*-
"""
Animate a trial.

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
from prehension.kinematics.mark_base import mark_base
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Manually label points on macaque torso to find its location once per'
                     ' calibration.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    parser.add_argument(
        '--skip_gui',
        action='store_true',
        help='Do not launch the GUI for labeling.')

    args = parser.parse_args(args=argv)

    # get default (scaling) session if asked
    if len(args.sessions) > 0 and args.sessions[0] == 'scaling':
        args.sessions = [current_preset['scaling']['session']]

    start_time = time.time()
    mark_base(args.server, args.sessions, args.temp, args.overwrite, args.skip_gui)
    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
