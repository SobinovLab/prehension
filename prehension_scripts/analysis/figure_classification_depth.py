#!python3
# -*- coding: utf-8 -*-
"""
Cross-validated classification of an object condition through time, by depth group.

Like figure_classification_time, but each session's neurons (above --min_rate Hz) are split
by depth along the probe: the top --top_fraction of them (farthest from the electrode tip,
default 0.25) and the bottom --bottom_fraction (nearest the tip, default 0.25).  Each group
is classified per session (thin lines) and as a pseudo-population pooled across sessions
(thick line) with k-fold cross-validated LDA at every time bin, each with its own
shuffle-based chance line; both groups are overlaid in two colours, plus the theoretical
(1 / n_conditions) chance line.  Recording / skip_ttl / good_neurons come from each
session's meta_neural.json; sessions whose NWB has no unit depths are left out.

The per-time-bin classification is parallelized with a multiprocessing pool whose size
is set by --processes.  scikit-learn provides the LDA / cross-validation.

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
from prehension.analysis.figure_classification_depth import (
    figure_classification_depth, TOP_FRACTION, BOTTOM_FRACTION, ALIGN_TIMEPOINT,
    GROUP_COLUMN, BEFORE, AFTER, CAUSAL_SIGMA, N_FOLDS, MIN_RATE_HZ, SUSTAINED_DURATION_S)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Classification of an object condition through time for the top and "
                    "bottom depth groups of each session's neurons: per-session and pooled "
                    "pseudo-population k-fold cross-validated LDA accuracy vs time, with "
                    "shuffle-based and theoretical chance lines.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "processes", "drift_correct"))
    parser.add_argument(
        "--top_fraction", type=float, default=TOP_FRACTION, metavar="FRACTION",
        help="Fraction of each session's neurons farthest from the electrode tip that form "
             "the top group. Default: {}.".format(TOP_FRACTION))
    parser.add_argument(
        "--bottom_fraction", type=float, default=BOTTOM_FRACTION, metavar="FRACTION",
        help="Fraction of each session's neurons nearest the electrode tip that form the "
             "bottom group. Default: {}.".format(BOTTOM_FRACTION))
    parser.add_argument(
        "--align", type=str, default=ALIGN_TIMEPOINT,
        help="Trial timepoint to align to: a timepoints.csv column (e.g. "
             "first_grasp_start) or a meta_session 'ttl_to_*' column (e.g. "
             "ttl_to_success_grasp, ttl_to_reach, ttl_to_force_target_start). "
             "Default: {}.".format(ALIGN_TIMEPOINT))
    parser.add_argument(
        "--group_column", type=str, default=GROUP_COLUMN,
        help="Object property that defines the classes. Default: {}.".format(GROUP_COLUMN))
    parser.add_argument(
        "--before", type=float, default=BEFORE, metavar="SECONDS",
        help="Seconds before the alignment event. Default: {}.".format(BEFORE))
    parser.add_argument(
        "--after", type=float, default=AFTER, metavar="SECONDS",
        help="Seconds after the alignment event. Default: {}.".format(AFTER))
    parser.add_argument(
        "--causal_sigma", type=float, default=CAUSAL_SIGMA, metavar="SECONDS",
        help="SD of the causal half-gaussian rate filter applied before classification. "
             "Default: {}.".format(CAUSAL_SIGMA))
    parser.add_argument(
        "--avg_window", type=float, default=None, metavar="SECONDS",
        help="If given, additionally average the activity over a centred moving window "
             "of this width (s) at each time bin. Off by default.")
    parser.add_argument(
        "--n_folds", type=int, default=N_FOLDS, metavar="K",
        help="Cross-validation folds (trials held out). Default: {}.".format(N_FOLDS))
    parser.add_argument(
        "--only_good", action="store_true",
        help="Use only each session's meta_neural.json 'good_neurons' (else all units).")
    parser.add_argument(
        "--min_rate", type=float, default=MIN_RATE_HZ, metavar="HZ",
        help="Keep only neurons with mean rate above this (Hz), before the depth split. "
             "Default: {}.".format(MIN_RATE_HZ))
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Read the threshold-crossing product (neural_threshold_crossings.nwb) instead "
             "of the sorted neural.nwb (the sorted file is used by default, or the threshold "
             "crossings automatically, with a warning, when it is missing).")
    parser.add_argument(
        "--sustained_ms", type=float, default=SUSTAINED_DURATION_S * 1e3, metavar="MS",
        help="Report the sessions whose accuracy stays above their own shuffle chance "
             "threshold for a contiguous stretch longer than this (ms). "
             "Default: {:.0f}.".format(SUSTAINED_DURATION_S * 1e3))
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/pooled_figures/figure_classification_depth, named after "
             "the --sessions string).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)
    name = cmd_args.sessions_name_stub(args.sessions)

    start_time = time.time()
    figure_classification_depth(
        args.server, args.processed_server, sessions,
        top_fraction=args.top_fraction, bottom_fraction=args.bottom_fraction,
        align_timepoint=args.align, group_column=args.group_column,
        before=args.before, after=args.after, causal_sigma=args.causal_sigma,
        avg_window=args.avg_window, n_folds=args.n_folds,
        only_good=args.only_good, min_rate=args.min_rate, processes=args.processes,
        drift_correct=args.drift_correct, use_threshold_crossings=args.threshold_crossings,
        name=name, sustained_duration=args.sustained_ms / 1e3, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
