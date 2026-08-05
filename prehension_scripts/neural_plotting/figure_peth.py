#!python3
# -*- coding: utf-8 -*-
"""
Plot peri-event time histograms (PETH) for a session from its neural NWB and the
prehension behavioural meta, aligned to a trial timepoint and colour-coded by an
object property.

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

import matplotlib.pyplot as plt

from prehension import preset
from prehension.tools import cmd_args
from prehension.neural_processing.common import probe
from prehension.neural_plotting.figure_peth import plot_perievent_histograms, BEFORE, AFTER

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="PETH traces from the neural NWB, coloured by an object property.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session",))
    parser.add_argument(
        "--units", type=str, default=None, nargs="*", metavar="UNIT_ID",
        help="Unit ids to plot. If empty, plot all units.")
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to: a timepoints.csv column (e.g. "
             "first_grasp_start) or a meta_session 'ttl_to_*' column (e.g. "
             "ttl_to_success_grasp, ttl_to_reach, ttl_to_force_target_start). "
             "Default: first_grasp_start.")
    parser.add_argument(
        "--group_column", type=str, default="targetForce(N)",
        help="Object property to colour-code by. Default: targetForce(N).")
    parser.add_argument(
        "--before", type=float, default=BEFORE, metavar="SECONDS",
        help="Seconds before the alignment event to plot. Default: {}.".format(BEFORE))
    parser.add_argument(
        "--after", type=float, default=AFTER, metavar="SECONDS",
        help="Seconds after the alignment event to plot. Default: {}.".format(AFTER))
    parser.add_argument(
        "--skip_ttl", type=int, default=None, metavar="N",
        help="Positional pulse<->trial alignment offset. Positive N drops the "
             "first N TTL pulses; negative N drops the first |N| behavioural "
             "trials. Default: from meta_neural.json (then 0).")
    parser.add_argument(
        "--skip_ttl_last", type=int, default=None, metavar="N",
        help="Like --skip_ttl but trimming the end: positive N drops the last N "
             "TTL pulses; negative N drops the last |N| trials. Default: from "
             "meta_neural.json (then 0).")
    parser.add_argument(
        "--recording", type=str, default=None, metavar="N",
        help="Open Ephys recording within experiment1, 1-based (Recording1, "
             "Recording2, ...); accepts 2 or 'Recording2'. The NWB is per-session "
             "so this does not change what is read. Default: probe default.")
    parser.add_argument(
        "--only_good", action="store_true",
        help="Plot only the unit ids listed in meta_neural.json 'good_neurons' "
             "(ignored if --units is given).")
    parser.add_argument(
        "--min_rate", type=float, default=None, metavar="HZ",
        help="Optional threshold: drop units with mean rate at or below this (Hz). "
             "Default: from meta_neural.json 'min_rate' if present, else no filter.")

    args = parser.parse_args(args=argv)

    # probe type is read from the session meta_structure ('neural' field), not the command line
    probe_type = probe.probe_type_from_meta(args.server, args.processed_server, args.session)

    start_time = time.time()
    plot_perievent_histograms(
        args.server, args.processed_server, args.session, probe_type,
        neuron_ids=args.units, align_timepoint=args.align, group_column=args.group_column,
        before=args.before, after=args.after,
        skip_ttl=args.skip_ttl, skip_ttl_last=args.skip_ttl_last,
        recording=args.recording, only_good=args.only_good, min_rate=args.min_rate)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
