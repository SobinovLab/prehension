#!python3
# -*- coding: utf-8 -*-
"""
Transforms an OpenSim model into a MuJoCo one. Once per monkey.

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
from prehension.matching.transform_osim_model import transform_osim_model


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Generates a MuJoCo model from an OpenSim model.'))
    cmd_args.add_default_arguments(
        parser, ('session', 'overwrite'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    args = parser.parse_args(args=argv)

    start_time = time.time()
    transform_osim_model(
        current_preset['default_server'],
        current_preset['processed_server'],
        args.session, args.overwrite)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
