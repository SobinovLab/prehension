#!python3
# -*- coding: utf-8 -*-
"""
Run the necessary neural processing chain to produce neural.nwb for a session:
    load_recording -> preprocess_recording -> run_spike_sorting -> export_nwb.

Copyright (C) 2026 Anton Sobinov
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
from prehension.neural_processing import config
from prehension.neural_processing.pipeline import run_necessary

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Produce the minimal neural NWB (load, preprocess, sort, export).")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session", "temp", "processes"))
    parser.add_argument(
        "--nwb_units", type=str, default="all", choices=["all", "curated"],
        help="Units written to the NWB. Default: all.")
    parser.add_argument(
        "--sorter", type=str, default="kilosort4",
        help="SpikeInterface sorter name. Default: kilosort4.")

    args = parser.parse_args(args=argv)

    # probe type is read from the session meta_structure ('neural' field), not the command line
    probe_type = config.probe_type_from_meta(args.server, args.processed_server, args.session)

    start_time = time.time()
    run_necessary(args.server, args.processed_server, args.session, probe_type,
                  args.temp, nwb_units=args.nwb_units, sorter=args.sorter,
                  processes=args.processes)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
