#!python3
# -*- coding: utf-8 -*-
"""
Transforms recorded images into similar quality videos.

Decreases the size requirements by >100 times.

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
from prehension.tools.video import compress_session_cameras
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Transforms images from a session into video.'))
    cmd_args.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    parser.add_argument(
        '--clean',
        action='store_true',
        help='DANGER! Remove directories from the server that were converted into videos.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    compress_session_cameras(
        current_preset, args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.clean)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
