#!python3
# -*- coding: utf-8 -*-
"""
Create per-session meta_neural.json (run before run_necessary).

Writes processed_server/<session>/meta_neural.json with defaults for the session's
probe type (resolved from meta_structure): the probe type, anatomical annotations
(region/burr_hole/depth_um/notes), and the default processing/plotting parameters
(recording, skip_ttl, sorter, nwb_units, block_index, common reference, curation
query, the probe defaults, and -- for V-probes -- the geometry/wiring).  The neural
steps and plotting scripts read their defaults from this file when the matching CLI
kwarg is omitted.

Created once per session; an existing file is left unchanged (there is no
--overwrite).  Edit the file (region, burr_hole, recording, contact wiring, ...)
before running run_necessary.  Requires create_meta to have run first.

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
from prehension.neural_processing.pipeline import create_meta_neural

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Create per-session meta_neural.json (run before run_necessary). "
                    "Existing files are left unchanged; there is no --overwrite.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "temp"))

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    create_meta_neural(args.server, args.processed_server, sessions, args.temp)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
