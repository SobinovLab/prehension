#!python3
# -*- coding: utf-8 -*-
"""
Identify the optimal input->output lag for the GLM encoding models, per session.

For each session and predictor, sweeps the lag between the covariates and each unit's
firing rate over [--lag_min, --lag_max] (step defaults to the bin width 1/fps), picks the
pseudo-R2-maximizing lag per unit, and saves the per-unit lags and their session mode to
the session's encoding/ folder (one file per predictor).  By default it then also fits and
writes the encoding pseudo-R2 file at the session mode lag (the same output the encoding
script produces), so a separate encoding run is not needed; pass --no_encoding to sweep the
lag only.

Positive lags mean the covariate follows the firing rate (the encoding direction is
opposite to causality).

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
from prehension.analysis.encoding import (
    optimal_lag, ALL_PREDICTORS, PERIODS, LAG_MIN, LAG_MAX, LAG_MIN_RATE_HZ, MIN_ADJ_R2,
    JOINT_GROUPS, DEFAULT_JOINT_GROUP)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Find and save the pseudo-R2-maximizing input->output lag per unit "
                    "(and its session mode) for the GLM encoding models, one file per "
                    "predictor in the session's encoding/ folder.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "processes", "overwrite"))
    parser.add_argument(
        "--predictors", "--modality", dest="predictors", nargs="+",
        default=list(ALL_PREDICTORS), choices=ALL_PREDICTORS, metavar="MODALITY",
        help="Modalities to sweep -- base signals or combined groups (position_velocity, "
             "position_torques, position_segment_forces); pass one to sweep only that, e.g. "
             "--modality joint_angles. Default (all): {}.".format(' '.join(ALL_PREDICTORS)))
    parser.add_argument(
        "--lag_min", type=float, default=LAG_MIN, metavar="SECONDS",
        help="Most negative lag (covariate leads firing). Default: {}.".format(LAG_MIN))
    parser.add_argument(
        "--lag_max", type=float, default=LAG_MAX, metavar="SECONDS",
        help="Most positive lag (covariate follows firing). Default: {}.".format(LAG_MAX))
    parser.add_argument(
        "--lag_step", type=float, default=None, metavar="SECONDS",
        help="Lag grid step. Default: the bin width 1/fps.")
    parser.add_argument(
        "--n_folds", type=int, default=5, metavar="K",
        help="Cross-validation folds. Default: 5.")
    parser.add_argument(
        "--alpha", type=float, default=1e-4, metavar="A",
        help="L2 penalty of the Poisson GLM. Default: 1e-4.")
    parser.add_argument(
        "--max_predictors", type=int, default=None, metavar="N",
        help="Constrain the model to at most N predictors (lasso selector). Default: all.")
    parser.add_argument(
        "--n_pcs", type=int, default=None, metavar="N",
        help="Replace the predictors with their top N principal components. Default: off.")
    parser.add_argument(
        "--min_rate", type=float, default=LAG_MIN_RATE_HZ, metavar="HZ",
        help="Only search lags for units with mean firing rate above this (Hz); low-rate "
             "units are skipped and left out of the mode. Default: {}.".format(LAG_MIN_RATE_HZ))
    parser.add_argument(
        "--min_adj_r2", type=float, default=MIN_ADJ_R2, metavar="R2",
        help="Drop units whose confirming adjusted pseudo-R2 is below this from the lag mode. "
             "Default: {}.".format(MIN_ADJ_R2))
    parser.add_argument(
        "--bin_width", type=float, default=None, metavar="SECONDS",
        help="Firing-rate bin width. Default: the session kinematic/video period 1/fps.")
    parser.add_argument(
        "--period", choices=PERIODS, default=None, metavar="PERIOD",
        help="Restrict every trial to a sub-period before the lag search: 'all' (whole trial, "
             "saved without a suffix), 'active_movement' (earliest movement onset -> hand "
             "retreat), or 'active_grasp' (first grasp -> release). Default (per modality): "
             "kinematics/torques -> active_movement, forces -> active_grasp.")
    parser.add_argument(
        "--joint_group", "--joints", dest="joint_group", choices=sorted(JOINT_GROUPS),
        default=DEFAULT_JOINT_GROUP, metavar="GROUP",
        help="Restrict the per-DOF predictors (positions, velocities, torques) to a joint group: "
             "'hand' (distal DOFs), 'proximal' (shoulder + elbow) or 'all' (every independent "
             "right-arm DOF). Must match the group encoding.py will fit. Default: {}.".format(
                 DEFAULT_JOINT_GROUP))
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Use the threshold-crossing channels instead of the sorted units.")
    parser.add_argument(
        "--no_encoding", dest="write_encoding", action="store_false",
        help="Only sweep the lag; do NOT also write the encoding r2 file at the session mode "
             "lag. By default the lag run also fits and saves encoding_<predictor>[...] .json "
             "so a separate encoding.py run is not needed.")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    optimal_lag(
        args.server, args.processed_server, sessions, predictors=args.predictors,
        lag_min=args.lag_min, lag_max=args.lag_max, lag_step=args.lag_step,
        n_folds=args.n_folds, alpha=args.alpha, max_predictors=args.max_predictors,
        n_pcs=args.n_pcs, min_rate=args.min_rate, min_adj_r2=args.min_adj_r2,
        bin_width=args.bin_width, use_threshold_crossings=args.threshold_crossings,
        processes=args.processes, overwrite=args.overwrite, period=args.period,
        write_encoding=args.write_encoding, joint_group=args.joint_group)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
