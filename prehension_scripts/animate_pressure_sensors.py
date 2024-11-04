#!python3
# -*- coding: utf-8 -*-
"""
Animate pressure sensors.

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
from prehension.visualization.animate_pressure_sensors import animate_pressure_sensors


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=(
            'Shows the forces exerted by each finger as measured manually and matched'
            ' automatically.'
        )
    )
    cmd_args.add_default_kwarguments(parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(parser, ('session', 'trial'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    animate_pressure_sensors(args.server, args.session, args.trial)
    print('Program took {}.'.format(datetime.timedelta(seconds=time.time() - start_time)))
