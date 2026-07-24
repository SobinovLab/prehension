#!python3
# -*- coding: utf-8 -*-
"""
Pooled neural state-space (demixed PCA) figures across multiple sessions.

Pools the selected neurons from a list of sessions (reusing figure_peth_pooled),
keeps the ones with mean rate above --min_rate Hz, runs demixed PCA on their
per-condition average activity, prints the variance per marginalization
(conditions vs condition-independent), and outputs the matlab_dpca summary figure
plus the top 3 condition-dependent demixed components through time.
Recording / skip_ttl / good_neurons come from each session's meta_neural.json;
there is no --units.

demixed PCA needs the local 'matlab_dpca' package (install with `pip install -e .`
from its source tree).

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
from prehension.neural_plotting.figure_spaces_pooled import MIN_RATE_HZ
from prehension.neural_plotting.figure_spaces_dpca_pooled import (
    figure_spaces_dpca_pooled, N_COMPONENTS)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Pooled neural demixed PCA across multiple sessions: the matlab_dpca "
                    "summary figure plus variance per marginalization (conditions vs "
                    "condition-independent) and the top condition-dependent demixed "
                    "components through time.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions",))
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to. Default: first_grasp_start.")
    parser.add_argument(
        "--group_column", type=str, default="targetForce(N)",
        help="Object property that defines the conditions. Default: targetForce(N).")
    parser.add_argument(
        "--before", type=float, default=BEFORE, metavar="SECONDS",
        help="Seconds before the alignment event. Default: {}.".format(BEFORE))
    parser.add_argument(
        "--after", type=float, default=AFTER, metavar="SECONDS",
        help="Seconds after the alignment event. Default: {}.".format(AFTER))
    parser.add_argument(
        "--only_good", action="store_true",
        help="Pool only each session's meta_neural.json 'good_neurons' (else all units).")
    parser.add_argument(
        "--min_rate", type=float, default=MIN_RATE_HZ, metavar="HZ",
        help="Keep only neurons with mean rate above this (Hz). Default: {}.".format(MIN_RATE_HZ))
    parser.add_argument(
        "--n_components", type=int, default=N_COMPONENTS, metavar="N",
        help="Demixed components kept overall (top 3 condition-dependent are traced). "
             "Default: {}.".format(N_COMPONENTS))

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    figure_spaces_dpca_pooled(
        args.server, args.processed_server, sessions,
        align_timepoint=args.align, group_column=args.group_column,
        before=args.before, after=args.after, only_good=args.only_good,
        min_rate=args.min_rate, n_components=args.n_components)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
