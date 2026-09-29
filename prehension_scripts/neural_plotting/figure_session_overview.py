#!python3
# -*- coding: utf-8 -*-
"""
Plot a per-session diagnostic overview through continuous recording time: a depth-sorted
spike raster, the population mean firing rate, the per-sensor total grasp force and the
joint angles, stacked on a shared time axis with TTL pulses (red) and reward times (green)
marked as dashed vertical lines.

Only the panels whose data a session has are drawn.  Behaviour is aligned to recording time
by each trial's TTL pulse (positional pulse<->trial pairing, meta_neural skip_ttl /
skip_ttl_last) and stitched, with gaps between trials, into a continuous trace.

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
from prehension.tools.filters import JOINT_GROUPS, DEFAULT_JOINT_GROUP
from prehension.neural_plotting.figure_session_overview import (
    plot_session_overview, BIN_WIDTH, FILTER_SIGMA, REWARD_KEY, DEFAULT_MINUTES_PER_FIGURE)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Per-session diagnostic overview (spike raster, population firing rate, "
                    "grasp force, joint angles) through continuous recording time, aligned by "
                    "each trial's TTL pulse.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session", "trials"))
    parser.add_argument(
        "--joint_group", "--joints", dest="joint_group", choices=sorted(JOINT_GROUPS),
        default=DEFAULT_JOINT_GROUP, metavar="GROUP",
        help="Right-arm DOF group for the joint-angle panel (selected as in the encoding "
             "models): 'hand' (distal DOFs), 'proximal' (shoulder + elbow), or 'all' (every "
             "independent right-arm DOF). Default: {}.".format(DEFAULT_JOINT_GROUP))
    parser.add_argument(
        "--bin_width", type=float, default=BIN_WIDTH, metavar="SECONDS",
        help="Bin width (s) of the population firing-rate trace. Default: {}.".format(BIN_WIDTH))
    parser.add_argument(
        "--filter_sigma", type=float, default=FILTER_SIGMA, metavar="SECONDS",
        help="SD (s) of the Gaussian smoothing of the firing-rate trace. Default: {}.".format(
            FILTER_SIGMA))
    parser.add_argument(
        "--min_rate", type=float, default=None, metavar="HZ",
        help="Only display units whose mean firing rate over the whole session window exceeds "
             "this (Hz); units at or below are dropped from the raster and the population rate. "
             "The same units appear on every page. Default: from meta_neural.json 'min_rate' if "
             "present, else no filter.")
    parser.add_argument(
        "--minutes_per_figure", type=float, default=DEFAULT_MINUTES_PER_FIGURE, metavar="MIN",
        help="Split the recording into separate figures covering this many minutes each (one "
             "figure per page, saved with a _pN suffix). Default: {:g}.".format(
                 DEFAULT_MINUTES_PER_FIGURE))
    parser.add_argument(
        "--reward_key", type=str, default=REWARD_KEY, metavar="COLUMN",
        help="meta_session offset column (seconds since the pulse) for the reward lines. "
             "Default: {}.".format(REWARD_KEY))
    parser.add_argument(
        "--skip_ttl", type=int, default=None, metavar="N",
        help="Positional pulse<->trial offset. Positive N drops the first N TTL pulses; "
             "negative N drops the first |N| trials. Default: from meta_neural.json (then 0).")
    parser.add_argument(
        "--skip_ttl_last", type=int, default=None, metavar="N",
        help="Like --skip_ttl but trimming the end: positive N drops the last N TTL pulses; "
             "negative N drops the last |N| trials. Default: from meta_neural.json (then 0).")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Read the threshold-crossing product (neural_threshold_crossings.nwb) instead of "
             "the sorted neural.nwb (the sorted file is used by default, or the threshold "
             "crossings automatically, with a warning, when it is missing).")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into "
             "<processed_server>/<session>/prehension_plots/figure_session_overview.png, or "
             "figure_session_overview_pN.png per page when there is more than one).")

    args = parser.parse_args(args=argv)

    start_time = time.time()
    plot_session_overview(
        args.server, args.processed_server, args.session, trials=args.trials,
        joint_group=args.joint_group, bin_width=args.bin_width, filter_sigma=args.filter_sigma,
        min_rate=args.min_rate, minutes_per_figure=args.minutes_per_figure,
        reward_key=args.reward_key, skip_ttl=args.skip_ttl, skip_ttl_last=args.skip_ttl_last,
        use_threshold_crossings=args.threshold_crossings, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
