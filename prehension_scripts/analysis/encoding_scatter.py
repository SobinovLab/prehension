#!python3
# -*- coding: utf-8 -*-
"""
Scatter the encoding-model performance (adjusted pseudo-R2) of predictors against each
other, for whichever models a session has saved.

Reads the encoding_<predictor>.json files under each session's encoding/ folder, aligns
each unit's adjusted pseudo-R2 across predictors, and draws a pairwise scatter (one panel
per predictor pair, with the identity line).  With no --predictors, every available
predictor is included.

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
from prehension.analysis.encoding import (
    ALL_PREDICTORS, PERIODS, MIN_ADJ_R2, JOINT_GROUPS, DEFAULT_JOINT_GROUP)
from prehension.analysis.encoding_scatter import encoding_scatter, encoding_difference_hist

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Compare the adjusted pseudo-R2 of the available encoding models against "
                    "each other, one panel per predictor pair: a pairwise scatter, and a "
                    "histogram of the paired per-unit differences annotated with a Wilcoxon "
                    "signed-rank test and rank-biserial correlation.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions",))
    parser.add_argument(
        "--predictors", "--modality", dest="predictors", nargs="+", default=None,
        choices=ALL_PREDICTORS, metavar="MODALITY",
        help="Restrict to these modalities (base signals or combined groups), e.g. --modality "
             "joint_angles position_torques. Default: every available predictor.")
    parser.add_argument(
        "--min_adj_r2", type=float, default=MIN_ADJ_R2, metavar="R2",
        help="Drop units whose adjusted pseudo-R2 is below this before scattering. "
             "Default: {}.".format(MIN_ADJ_R2))
    parser.add_argument(
        "--period", choices=PERIODS, default=None, metavar="PERIOD",
        help="Which trial sub-period's encoding files to scatter: 'all', 'active_movement' or "
             "'active_grasp'. Default (per modality): kinematics/torques -> active_movement, "
             "forces -> active_grasp (so paired predictors may differ in period; pass one "
             "period for a like-for-like comparison).")
    parser.add_argument(
        "--joint_group", "--joints", dest="joint_group", choices=sorted(JOINT_GROUPS),
        default=DEFAULT_JOINT_GROUP, metavar="GROUP",
        help="Which joint-group encoding files to scatter: 'hand', 'proximal' or 'all'. Must "
             "match the group encoding.py fit. Default: {}.".format(DEFAULT_JOINT_GROUP))
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Scatter the threshold-crossing encoding files instead of the sorted-unit ones.")
    parser.add_argument(
        "--bins", type=int, default=30, metavar="N",
        help="Number of bins in the difference histograms. Default: 30.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into "
             "<processed_server>/pooled_figures/encoding_scatter and .../encoding_difference_hist).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    encoding_scatter(
        args.processed_server, sessions, predictors=args.predictors,
        min_adj_r2=args.min_adj_r2, use_threshold_crossings=args.threshold_crossings,
        save=args.save, period=args.period, joint_group=args.joint_group)
    encoding_difference_hist(
        args.processed_server, sessions, predictors=args.predictors,
        min_adj_r2=args.min_adj_r2, use_threshold_crossings=args.threshold_crossings,
        save=args.save, period=args.period, bins=args.bins, joint_group=args.joint_group)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
