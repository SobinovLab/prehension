#!python3
# -*- coding: utf-8 -*-
"""
Pooled peri-event time histogram (PETH) figure across multiple sessions.

Pools the selected neurons from a list of sessions onto one figure (one subplot
per neuron, titled "<session>: <unit id>").  Per session, plots all units, or --
with --only_good -- the unit ids in that session's meta_neural.json 'good_neurons'.
Recording / skip_ttl come from each session's meta_neural.json; there is no
--units / --recording / --skip_ttl.

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
from prehension.neural_plotting.figure_peth import BEFORE, AFTER
from prehension.neural_plotting.figure_peth_pooled import plot_perievent_histograms_pooled

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Pooled PETH across multiple sessions, one subplot per neuron "
                    "('<session>: <unit id>'), coloured by an object property.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions",))
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to. Default: first_grasp_start.")
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
        "--only_good", action="store_true",
        help="Plot only each session's meta_neural.json 'good_neurons' (else all units).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    plot_perievent_histograms_pooled(
        args.server, args.processed_server, sessions,
        align_timepoint=args.align, group_column=args.group_column,
        before=args.before, after=args.after, only_good=args.only_good)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
