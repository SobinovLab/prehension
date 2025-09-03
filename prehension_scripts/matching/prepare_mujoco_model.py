#!python3
# -*- coding: utf-8 -*-
"""
Makes a mask of active sensels and a session's MuJoCo model based on the general MuJoCo model and
sensel maps.

Copyright (C) 2019-2025 Anton Sobinov
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
from prehension.matching.prepare_mujoco_model import prepare_mujoco_model


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Generates a mask of pressure sensors matrix that highlights activated '
                     'sensels and tessellates model sensors based on it.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite'))

    parser.add_argument(
        '--dont_make_mask',
        action='store_false', dest='make_mask',
        help='Converts.')
    parser.add_argument(
        '--dont_tessellate',
        action='store_false', dest='tessellate',
        help='Tessellates the pressure sensors into sensels.')
    parser.add_argument(
        '--sense_distance',
        type=float, default=0.025,
        help='Distance between geom centers for "contact" calculation. Larger values slow down the'
        ' execution, but low values are too short for relatively large bending bones like'
        ' metacarpals and large muscle areas like thenar eminence. In meters.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    prepare_mujoco_model(
        current_preset,
        args.sessions, args.trials, args.temp, args.overwrite,
        args.make_mask, args.tessellate, args.sense_distance)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
