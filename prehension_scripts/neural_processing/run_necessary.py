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
from prehension.neural_processing.pipeline import run_necessary

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Produce the minimal neural NWB (load, preprocess, sort, export) for one or "
                    "more sessions. The probe type is read per session from its meta_structure. "
                    "Sessions already processed are skipped unless --overwrite is given.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "temp", "processes", "overwrite"))
    parser.add_argument(
        "--nwb_units", type=str, default="noise_excluded",
        choices=["noise_excluded", "curated", "all"],
        help="Units written to the NWB. Default: noise_excluded (drops Phy 'noise' "
             "clusters when a phy export exists, else all units).")
    parser.add_argument(
        "--sorter", type=str, default="kilosort4",
        help="SpikeInterface sorter name. Default: kilosort4.")
    parser.add_argument(
        "--recording", type=str, default=None, metavar="N",
        help="Open Ephys recording within experiment1 to process, 1-based "
             "(Recording1, Recording2, ...); accepts 2 or 'Recording2'. "
             "Default: probe default (vprobe Recording1, neuropixels Recording2).")

    args = parser.parse_args(args=argv)

    start_time = time.time()
    run_necessary(args.server, args.processed_server, args.sessions,
                  args.temp, nwb_units=args.nwb_units, sorter=args.sorter,
                  processes=args.processes, overwrite=args.overwrite,
                  recording=args.recording)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
