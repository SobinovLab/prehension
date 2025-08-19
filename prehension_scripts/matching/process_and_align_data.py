#!python3
# -*- coding: utf-8 -*-
"""
Filters, resamples, and aligns pressure sensor and kinematic data to grasp onset.

Copyright (C) 2019 Anton Sobinov
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
from prehension.matching.process_and_align_data import process_and_align_data


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Filters, resamples, and aligns pressure sensor and kinematic data to grasp '
                     'onset.'))
    cmd_args.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite', 'make_plots'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    process_and_align_data(
        current_preset['default_server'],
        current_preset['processed_server'],
        args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.make_plots)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
