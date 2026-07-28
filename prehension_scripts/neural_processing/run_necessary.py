#!python3
# -*- coding: utf-8 -*-
"""
Run the necessary neural processing chain to produce neural.nwb for a session:
    load_recording -> preprocess_recording -> run_spike_sorting -> export_nwb.

By default the whole chain runs. Pass one or more of --preprocessing,
--spike_sorting, --export_nwb to run only those steps (they still execute in
pipeline order). Steps run in isolation rely on earlier steps' outputs already
existing on disk.

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
        "--nwb_units", type=str, default=None,
        choices=["noise_excluded", "curated", "all"],
        help="Units written to the NWB. Default: from meta_neural.json (then "
             "noise_excluded).")
    parser.add_argument(
        "--sorter", type=str, default=None,
        help="SpikeInterface sorter name. Default: from meta_neural.json (then kilosort4).")
    parser.add_argument(
        "--recording", type=str, default=None, metavar="N",
        help="Open Ephys recording within experiment1 to process, 1-based "
             "(Recording1, Recording2, ...); accepts 2 or 'Recording2'. "
             "Default: probe default (vprobe Recording1, neuropixels Recording2).")
    parser.add_argument(
        "--preprocessing", dest="steps", action="append_const", const="preprocessing",
        help="Run only the preprocessing step. Combine with other step flags to run "
             "several; give none to run the whole chain.")
    parser.add_argument(
        "--spike_sorting", dest="steps", action="append_const", const="spike_sorting",
        help="Run only the spike sorting step (see --preprocessing).")
    parser.add_argument(
        "--export_nwb", dest="steps", action="append_const", const="export_nwb",
        help="Run only the NWB export step (see --preprocessing).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    run_necessary(args.server, args.processed_server, sessions,
                  args.temp, nwb_units=args.nwb_units, sorter=args.sorter,
                  processes=args.processes, overwrite=args.overwrite,
                  recording=args.recording, steps=args.steps)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
