#!python3
# -*- coding: utf-8 -*-
"""
Plot a two-panel encoding summary across sessions: per-unit optimal lags (top swarm) and
the per-unit adjusted pseudo-R2 distribution (bottom swarm), grouped by predictor modality.

Reads the encoding/<predictor>_lag[_tx].json and encoding/encoding_<predictor>[_tx].json
files the encoding_lag / encoding scripts saved for the requested sessions.

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
from prehension.analysis.encoding import ALL_PREDICTORS, PERIODS, MIN_ADJ_R2
from prehension.analysis.encoding_summary import encoding_summary

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Two-panel encoding summary across sessions: per-unit optimal lags "
                    "(top swarm) and adjusted pseudo-R2 (bottom swarm, with a cutoff line) "
                    "per predictor modality.")
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
        help="Draw a dashed adjusted-pseudo-R2 cutoff line at this value on the bottom panel. "
             "Default: {}.".format(MIN_ADJ_R2))
    parser.add_argument(
        "--period", choices=PERIODS, default=None, metavar="PERIOD",
        help="Which trial sub-period's encoding files to summarize: 'all', 'active_movement' or "
             "'active_grasp'. Default (per modality): kinematics/torques -> active_movement, "
             "forces -> active_grasp (each column's period is shown in its x tick label).")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Summarize the threshold-crossing encoding files instead of the sorted-unit ones.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/pooled_figures/encoding_summary).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    encoding_summary(
        args.processed_server, sessions, predictors=args.predictors,
        min_adj_r2=args.min_adj_r2, use_threshold_crossings=args.threshold_crossings,
        save=args.save, period=args.period)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
