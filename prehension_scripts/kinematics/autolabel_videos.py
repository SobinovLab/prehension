#!python3
# -*- coding: utf-8 -*-
"""
Labels videos uses a pretrained machine vision network, from DLC.

TODO used to work on 3.8, check if works on 3.11.

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
from prehension.kinematics.ncams_3d import analyze_videos
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uses pretrained machine vision network to label videos.')
    )
    cmd_args.add_default_kwarguments(parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(parser, ('sessions', 'trials', 'temp', 'overwrite'))

    parser.add_argument(
        '--dlc_config_path',
        type=str,
        default=current_preset['dlc_config_path'],
        help='Location of the DLC config to use. Be sure to use the correct monkey config!',
    )
    parser.add_argument(
        '--dont_analyze',
        action='store_false',
        dest='analyze',
        help='Do not analyze videos using a DLC network.',
    )
    parser.add_argument(
        '--make_videos', action='store_true', help='Make videos with the labelled markers. '
    )

    args = parser.parse_args(args=argv)

    start_time = time.time()

    analyze_videos(
        args.server,
        args.sessions,
        args.trials,
        args.temp,
        args.overwrite,
        args.dlc_config_path,
        args.analyze,
        args.make_videos,
    )
    rs('Program took {}.'.format(datetime.timedelta(seconds=time.time() - start_time)))
