#!python3
# -*- coding: utf-8 -*-
"""
Kalman filter decoding of behaviour from neural activity through time, per session.

For each session and target (joint angles, joint torques, or the summed grasp force), fits a
cross-validated Kalman filter decoder (Wu et al. 2006) that estimates the target through time
from the population firing rates, holding out whole trials.  Reports one figure per session with
a panel per target: a decoded-vs-actual scatter of the held-out samples (z-scored per dimension)
with the identity line and the median per-dimension R2 / Pearson correlation.  All targets are
decoded by default; pass --target to run only some.

The figures are saved per session to <session>/prehension_plots/.  Joint-angle targets use the
right-arm independent joint DOFs (as in the encoding models); the neural source is the sorted
neural.nwb, or the threshold crossings with --threshold_crossings.

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
from prehension.analysis.encoding import PERIODS
from prehension.analysis.decoding import decode_targets, DECODE_TARGETS

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Per-session Kalman filter decoding of behaviour (joint angles, joint "
                    "torques, grasp force) from neural activity through time; a decoded-vs-actual "
                    "performance scatter per target.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "drift_correct"))
    parser.add_argument(
        "--target", "--targets", dest="targets", nargs="+", default=list(DECODE_TARGETS),
        choices=DECODE_TARGETS, metavar="TARGET",
        help="Targets to decode; pass one to decode only that, e.g. --target grasp_force. "
             "Default (all): {}.".format(' '.join(DECODE_TARGETS)))
    parser.add_argument(
        "--n_folds", type=int, default=5, metavar="K",
        help="Trial-wise cross-validation folds (whole trials held out). Default: 5.")
    parser.add_argument(
        "--period", choices=PERIODS, default=None, metavar="PERIOD",
        help="Restrict every trial to a sub-period before decoding: 'all', 'active_movement' or "
             "'active_grasp'. Default: per target, matching the encoding models "
             "(joint angles / torques -> active_movement, grasp force -> active_grasp).")
    parser.add_argument(
        "--bin_width", type=float, default=None, metavar="SECONDS",
        help="Firing-rate / target bin width. Default: the session kinematic/video period 1/fps.")
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Random seed for the cross-validation fold split. Default: None (random each run).")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Decode from the threshold-crossing channels (neural_threshold_crossings.nwb) "
             "instead of the sorted units.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into each "
             "<session>/prehension_plots/).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    decode_targets(
        args.server, args.processed_server, sessions, targets=args.targets, n_folds=args.n_folds,
        bin_width=args.bin_width, use_threshold_crossings=args.threshold_crossings,
        period=args.period, drift_correct=args.drift_correct, seed=args.seed, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
