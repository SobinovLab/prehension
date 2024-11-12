#!python3
# -*- coding: utf-8 -*-
"""
Exporting digit forces from the matched contacts and the pressure sensor recordings.

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
from prehension import tools
from prehension.matching.export_digit_forces import export_digit_forces

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=("Compare manually-labeled to the automatically-labeled forces"
                     " using sensor masks."))
    tools.add_default_kwarguments(parser, {"server": current_preset["default_server"]})
    tools.add_default_arguments(parser, ("sessions", "trials", "temp", "overwrite", "processes"))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    export_digit_forces(args.server, args.sessions, args.trials, args.temp, args.overwrite,
                        args.processes)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
