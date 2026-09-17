#!python3
# -*- coding: utf-8 -*-
"""
Fit Poisson GLM encoding models of individual neurons / channels for a session.

For each session and predictor (joint angles, joint velocity, digit or segment forces,
torques), assembles the continuous design across the session's trials and fits a
cross-validated Poisson GLM per unit, saving the per-unit pseudo-R2 / adjusted pseudo-R2
and the full fit specification to the session's encoding/ folder (one file per predictor,
naming the neural source).  If encoding/<predictor>_lag.json exists (see the encoding_lag
script), the session optimal lag for that predictor is applied.  With --plot_units, the
actual vs predicted firing-rate traces of those units are also plotted.

Bins default to the session kinematic/video rate (1/fps); the neural source is the sorted
neural.nwb, or the threshold crossings with --threshold_crossings (or automatically when
the sorted product is missing).

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
    encoding_models, ALL_PREDICTORS, PERIODS, JOINT_GROUPS, DEFAULT_JOINT_GROUP)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Fit Poisson GLM encoding models of individual neurons / channels per "
                    "session and save their pseudo-R2 / adjusted pseudo-R2 to the session's "
                    "encoding/ folder, one file per predictor.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "processes", "overwrite"))
    parser.add_argument(
        "--predictors", "--modality", dest="predictors", nargs="+",
        default=list(ALL_PREDICTORS), choices=ALL_PREDICTORS, metavar="MODALITY",
        help="Modalities to fit -- base signals or combined groups (position_velocity, "
             "position_torques, position_segment_forces); pass one to encode only that, e.g. "
             "--modality joint_angles. Default (all): {}.".format(' '.join(ALL_PREDICTORS)))
    parser.add_argument(
        "--n_folds", type=int, default=5, metavar="K",
        help="Cross-validation folds. Default: 5.")
    parser.add_argument(
        "--alpha", type=float, default=1e-4, metavar="A",
        help="L2 penalty of the Poisson GLM (small ~ maximum likelihood). Default: 1e-4.")
    parser.add_argument(
        "--max_predictors", type=int, default=None, metavar="N",
        help="Constrain the model to at most N predictors, chosen by a lasso (L1) selector. "
             "Default: use all predictor channels.")
    parser.add_argument(
        "--n_pcs", type=int, default=None, metavar="N",
        help="Replace the predictors with their top N principal components before fitting. "
             "Default: off.")
    parser.add_argument(
        "--bin_width", type=float, default=None, metavar="SECONDS",
        help="Firing-rate bin width. Default: the session kinematic/video period 1/fps.")
    parser.add_argument(
        "--period", choices=PERIODS, default=None, metavar="PERIOD",
        help="Restrict every trial to a sub-period before fitting: 'all' (whole trial, saved "
             "without a suffix), 'active_movement' (earliest movement onset -> hand retreat), or "
             "'active_grasp' (first grasp -> release). The lag applied is the one saved for the "
             "same period. Default (per modality): kinematics/torques -> active_movement, "
             "forces -> active_grasp.")
    parser.add_argument(
        "--joint_group", "--joints", dest="joint_group", choices=sorted(JOINT_GROUPS),
        default=DEFAULT_JOINT_GROUP, metavar="GROUP",
        help="Restrict the per-DOF predictors (positions, velocities, torques) to a joint group: "
             "'hand' (distal DOFs), 'proximal' (shoulder + elbow), or 'all' (every independent "
             "right-arm DOF). Applied identically to all three; forces are unaffected. Non-'all' "
             "groups are appended to the output file names. Default: {}.".format(
                 DEFAULT_JOINT_GROUP))
    parser.add_argument(
        "--units", nargs="+", default=None, metavar="UNIT_ID",
        help="Compute encoding for only these unit ids and report them to the log WITHOUT "
             "reading or writing the saved encoding JSON (no existence check, no overwrite). "
             "Combine with --plot_units to plot them. Default: all units, saved to encoding/.")
    parser.add_argument(
        "--plot_units", nargs="+", default=None, metavar="UNIT_ID",
        help="Plot the actual vs predicted rate traces for these unit ids. Like --units, "
             "passing --plot_units fits ONLY these units and does not read or overwrite the "
             "saved encoding JSON. Default: none (fit and save all units).")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Encode the threshold-crossing channels (neural_threshold_crossings.nwb) instead "
             "of the sorted units (used automatically, with a warning, when sorted is missing).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    encoding_models(
        args.server, args.processed_server, sessions, predictors=args.predictors,
        n_folds=args.n_folds, alpha=args.alpha, max_predictors=args.max_predictors,
        n_pcs=args.n_pcs, bin_width=args.bin_width, units=args.units,
        plot_units=args.plot_units, use_threshold_crossings=args.threshold_crossings,
        processes=args.processes, overwrite=args.overwrite, period=args.period,
        joint_group=args.joint_group)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    if args.plot_units:
        plt.show()
