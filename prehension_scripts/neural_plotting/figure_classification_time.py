#!python3
# -*- coding: utf-8 -*-
"""
Cross-validated classification of an object condition through time, pooled across sessions.

Pools the selected neurons from a list of sessions (reusing the figure_peth session
machinery), causally smooths and square-roots each trial's firing rate, keeps the
neurons with mean rate above --min_rate Hz, and trains a k-fold cross-validated LDA at
every time bin to predict the trial's condition (default targetForce(N)).  Chance is
the 90th percentile of the label-shuffled accuracies (one shuffle per bin, pooled over
time).  Plots the classification through time for each session (thin grey) and for the
pseudo-population pooled across sessions (thick black), each with a green dashed chance
line, plus the theoretical (1 / n_conditions) chance line as a green solid line.  A
second set of sessions (--sessions2) is pooled separately and overlaid on the same axes
in another colour for comparison.  Recording / skip_ttl / good_neurons come from each
session's meta_neural.json; there is no --units.

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
from prehension.neural_plotting.figure_peth import BEFORE, AFTER
from prehension.neural_plotting.figure_spaces_pooled import MIN_RATE_HZ
from prehension.neural_plotting.figure_classification_time import (
    figure_classification_time, CAUSAL_SIGMA, N_FOLDS, SUSTAINED_DURATION_S)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Classification of an object condition through time, pooled across "
                    "sessions: per-session (grey) and pooled pseudo-population (black) "
                    "k-fold cross-validated LDA accuracy vs time, with shuffle-based and "
                    "theoretical chance lines. An optional second session set (--sessions2) "
                    "is pooled separately and overlaid for comparison.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "sessions2", "processes", "drift_correct"))
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to: a timepoints.csv column (e.g. "
             "first_grasp_start) or a meta_session 'ttl_to_*' column (e.g. "
             "ttl_to_success_grasp, ttl_to_reach, ttl_to_force_target_start). "
             "Default: first_grasp_start.")
    parser.add_argument(
        "--group_column", type=str, default="targetForce(N)",
        help="Object property that defines the classes. Default: targetForce(N).")
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
        help="Keep only neurons with mean rate above this (Hz). Default: {}.".format(MIN_RATE_HZ))
    parser.add_argument(
        "--sustained_ms", type=float, default=SUSTAINED_DURATION_S * 1e3, metavar="MS",
        help="Report the sessions whose accuracy stays above their own shuffle chance "
             "threshold for a contiguous stretch longer than this (ms). "
             "Default: {:.0f}.".format(SUSTAINED_DURATION_S * 1e3))
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/pooled_figures/figure_classification_time, named after "
             "the --sessions/--sessions2 strings).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)
    sessions2 = cmd_args.resolve_sessions(args.sessions2, args.processed_server)

    # legend labels: the raw token string used to select each set (before resolution)
    sessions_label = ' '.join(args.sessions) if args.sessions else 'all sessions'
    sessions2_label = ' '.join(args.sessions2) if args.sessions2 else None
    # filename stub from the raw --sessions / --sessions2 tokens
    name = cmd_args.sessions_name_stub(args.sessions, args.sessions2)

    start_time = time.time()
    figure_classification_time(
        args.server, args.processed_server, sessions, sessions2=sessions2,
        align_timepoint=args.align, group_column=args.group_column,
        before=args.before, after=args.after, causal_sigma=args.causal_sigma,
        avg_window=args.avg_window, n_folds=args.n_folds,
        only_good=args.only_good, min_rate=args.min_rate, processes=args.processes,
        drift_correct=args.drift_correct,
        sessions_label=sessions_label, sessions2_label=sessions2_label,
        name=name, sustained_duration=args.sustained_ms / 1e3, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
