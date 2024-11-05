#!python3
# -*- coding: utf-8 -*-
"""
Uploading data for sharing, defaults to Box.

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
import os
import time

from prehension import preset
from prehension.tools import cmd_args
from prehension.upload_data import upload_data
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uploads the data from local server to server accessible to collaborators.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    default_target_dir = os.path.join(os.environ['USERPROFILE'], 'Box', 'PrehensionProject')
    parser.add_argument(
        '--target_dir',
        type=str, default=default_target_dir,
        help='Where to upload the data. Default: {}'.format(default_target_dir))

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Do not copy the data, only print out the files to be copied.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    upload_data(args.server, args.sessions, args.temp, args.target_dir, args.dry_run,
                args.overwrite)
    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
