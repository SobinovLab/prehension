#!python3
# -*- coding: utf-8 -*-
"""
Demixed PCA (or PCA) of a behavioural signal across sessions.

Pools the channels of a behavioural signal (joint angles, torques, digit or segment
forces) into per-condition averages and runs demixed PCA (default) or PCA on them,
reusing the neural spaces machinery and drawers.  Conditions come from an object
property (--group_column, e.g. targetForce(N)); trials are aligned to --align.  Figures
are written under <processed_server>/pooled_figures/spaces_behavior_<signal>_<method>.

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
from prehension.neural_processing.common.spikes import BEFORE, AFTER
from prehension.analysis.behavior_pooling import SIGNAL_SPECS
from prehension.analysis.spaces_behavior import spaces_behavior, N_COMPONENTS

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Demixed PCA (or PCA) of a behavioural signal (joint angles, torques, "
                    "digit or segment forces) across sessions, conditioned on an object "
                    "property, aligned to a trial timepoint.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions",))
    parser.add_argument(
        "--signal", type=str, default="joint_angles", choices=sorted(SIGNAL_SPECS),
        help="Behavioural signal to decompose. Default: joint_angles.")
    parser.add_argument(
        "--method", type=str, default="dpca", choices=("dpca", "pca"),
        help="Decomposition: demixed PCA (dpca) or plain PCA (pca). Default: dpca.")
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to: a timepoints.csv column (e.g. "
             "first_grasp_start) or a meta_session 'ttl_to_*' column. "
             "Default: first_grasp_start.")
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
        "--n_components", type=int, default=N_COMPONENTS, metavar="N",
        help="Demixed components kept (dpca) or PCs computed (pca). "
             "Default: {}.".format(N_COMPONENTS))
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into "
             "<processed_server>/pooled_figures/spaces_behavior_<signal>_<method>).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)
    name = cmd_args.sessions_name_stub(args.sessions)

    start_time = time.time()
    spaces_behavior(
        args.server, args.processed_server, sessions, args.signal, method=args.method,
        group_column=args.group_column, align_timepoint=args.align,
        before=args.before, after=args.after, n_components=args.n_components,
        name=None if name is None else '{}_{}'.format(name, args.signal), save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
