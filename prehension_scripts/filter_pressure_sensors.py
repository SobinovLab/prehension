#!python3
# -*- coding: utf-8 -*-
"""
Creates meta-information for sessions.

Copyright (C) 2019-2024 Anton Sobinov, Caleb Raman
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
from prehension.pressure_sensors import filter_pressure_sensors as fps


if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(description="Creates meta information for a session.")
    cmd_args.add_default_kwarguments(parser, {"server": current_preset["default_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "temp", "overwrite", "trials", "processes"))

    # custom
    parser.add_argument("--make_plots", action="store_true")
    parser.add_argument("--dont_export_roms",
                        dest="export_roms",
                        action="store_false",
                        help=("Exports range of motion data from OpenSim model into a convenient "
                            " CSV meta file. If this flag is provided, meta_dof is not created."))

    args = parser.parse_args(args=argv)
    start_time = time.time()

    fps.filter_pressure_sensors(
        current_preset,
        args.trials,
        args.temp,
        args.overwrite,
        args.processes,
        sessions_sel=args.sessions,
    )

    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
