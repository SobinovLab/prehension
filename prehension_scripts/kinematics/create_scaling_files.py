#!python3
# -*- coding: utf-8 -*-
"""
Creates scaling files for an animal based on a specific trial. In the second mode of operation,
transfers the position from IK into a default animal OpenSim model posture.

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
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension.tools import cmd_args
from prehension.kinematics.create_scaling_files import create_scaling_files
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Create IK and SC files for scaling an OpenSim model.'))
    cmd_args.add_default_arguments(
        parser, ('session', 'trial', 'temp', 'overwrite'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    # other
    parser.add_argument(
        '--period',
        type=float, default=[], nargs=2,
        help='Time period in seconds to use for scaling. If empty, use best estimation.'
        ' Empty by default.')
    parser.add_argument(
        '--transfer_position',
        action='store_true',
        help='Transfer the joint angles that have resulted from IK into the model that is being'
        ' scaled. Different mode of operation, does not generate scaling files.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    create_scaling_files(
        current_preset, args.session, args.trial, args.temp,
        args.overwrite, args.period, args.transfer_position)

    rs('Program took {} s.'.format(time.time() - start_time))

    plt.show()
