#!python3
# -*- coding: utf-8 -*-
"""
Re-export the minimal neural.nwb product from an externally-produced kilosorted.nwb.

For each session this reads
    processed_server/<session>/neural_processed_nwb/kilosorted.nwb
and writes
    processed_server/<session>/neural_processed/neural.nwb
-- the same minimal NWB the prehension neural module consumes (a Units table with
spike_times + unit_id and a ttl_pulses TimeIntervals), so it is a drop-in substitute
for running the local load -> preprocess -> sort -> export_nwb chain when a session
was already sorted and curated by the earlier pipeline.

Only the selected units are written: by default the Phy 'good' units (the
kilosorted.nwb per-unit 'unit_label').  Pass --include_mua to also keep 'mua' units,
--labels to give an explicit label set, or --all_units to keep every unit.

The kept units are written in ascending probe-depth order (matching
figure_neural.get_phy_data order_by_depth=1), using the depth in
server/<session>/neural/final_phy/cluster_info.tsv; pass --no_depth_order to keep the
kilosorted.nwb source order instead.

Sessions already having a neural.nwb are skipped unless --overwrite is given.

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
from prehension.neural_processing.import_kilosorted import (
    reexport_kilosorted, DEFAULT_SELECTED_LABELS)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Re-export the minimal neural.nwb from each session's "
                    "neural_processed_nwb/kilosorted.nwb, keeping only the selected "
                    "units. Sessions already having a neural.nwb are skipped unless "
                    "--overwrite is given.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "temp", "overwrite"))
    parser.add_argument(
        "--no_depth_order", dest="order_by_depth", action="store_false",
        help="Keep the kilosorted.nwb source order instead of ordering the written "
             "units by ascending probe depth (final_phy/cluster_info.tsv).")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--labels", type=str, nargs="+", metavar="LABEL", default=None,
        help="Phy unit labels to keep from kilosorted.nwb (e.g. good mua). "
             "Default: {}.".format(list(DEFAULT_SELECTED_LABELS)))
    group.add_argument(
        "--include_mua", action="store_true",
        help="Keep both 'good' and 'mua' units (shorthand for --labels good mua).")
    group.add_argument(
        "--all_units", action="store_true",
        help="Keep every unit regardless of its label.")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    if args.all_units:
        selected_labels = None
    elif args.include_mua:
        selected_labels = ("good", "mua")
    elif args.labels:
        selected_labels = tuple(args.labels)
    else:
        selected_labels = DEFAULT_SELECTED_LABELS

    start_time = time.time()
    reexport_kilosorted(args.server, args.processed_server, sessions, args.temp,
                        selected_labels=selected_labels, overwrite=args.overwrite,
                        order_by_depth=args.order_by_depth)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
