#!python3
# -*- coding: utf-8 -*-
"""
Finds a median thorax position in a session.

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
from prehension.kinematics.find_static_thorax_position import find_static_thorax_position
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Calculates and saves median base body position (thorax) in the'
                     ' OpenSim model.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    find_static_thorax_position(
        current_preset['default_server'],
        current_preset['processed_server'],
        args.sessions, args.trials, args.temp, args.processes,
        args.overwrite)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
