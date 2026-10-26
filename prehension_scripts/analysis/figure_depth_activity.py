#!python3
# -*- coding: utf-8 -*-
"""
Aligned neural activity by depth along the probe, for one session.

Aligns every unit's firing rate to a trial timepoint, averages it over the successful
trials, groups the units into --n_depth_bins equal-width bins over the span of unit
depths, averages the units in each bin and normalizes each depth bin to its own peak
(0..1).  The figure is a heatmap of time vs depth (electrode tip at the bottom); depth
bins without units are grey.  With --stack_by_depth each unit gets its own row instead,
stacked in depth order.  Recording / skip_ttl / good_neurons come from the
session's meta_neural.json.

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
from prehension.analysis.figure_depth_activity import (
    figure_depth_activity, ALIGN_TIMEPOINT, BEFORE, AFTER, FILTER_SIGMA, N_DEPTH_BINS)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="One session's aligned, trial-averaged activity binned by unit depth along "
                    "the probe, each depth bin normalized to its own peak (0..1).")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session", "drift_correct"))
    parser.add_argument(
        "--align", type=str, default=ALIGN_TIMEPOINT,
        help="Trial timepoint to align to: a timepoints.csv column (e.g. "
             "first_grasp_start) or a meta_session 'ttl_to_*' column (e.g. "
             "ttl_to_success_grasp, ttl_to_reach, ttl_to_force_target_start). "
             "Default: {}.".format(ALIGN_TIMEPOINT))
    parser.add_argument(
        "--before", type=float, default=BEFORE, metavar="SECONDS",
        help="Seconds before the alignment event. Default: {}.".format(BEFORE))
    parser.add_argument(
        "--after", type=float, default=AFTER, metavar="SECONDS",
        help="Seconds after the alignment event. Default: {}.".format(AFTER))
    parser.add_argument(
        "--filter_sigma", type=float, default=FILTER_SIGMA, metavar="SECONDS",
        help="SD (s) of the Gaussian smoothing of the firing-rate traces. Default: {}.".format(
            FILTER_SIGMA))
    parser.add_argument(
        "--n_depth_bins", type=int, default=N_DEPTH_BINS, metavar="N",
        help="Number of equal-width bins over the span of unit depths. Default: {}.".format(
            N_DEPTH_BINS))
    parser.add_argument(
        "--stack_by_depth", action="store_true",
        help="Ignore the precise depth: draw one row per unit, stacked in depth order (tip at "
             "the bottom) and each normalized to its own peak; --n_depth_bins is unused. "
             "Saved with a _stacked suffix.")
    parser.add_argument(
        "--only_good", action="store_true",
        help="Use only the unit ids listed in meta_neural.json 'good_neurons' (else all units).")
    parser.add_argument(
        "--min_rate", type=float, default=None, metavar="HZ",
        help="Optional threshold: drop units with mean rate at or below this (Hz). "
             "Default: from meta_neural.json 'min_rate' if present, else no filter.")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Read the threshold-crossing product (neural_threshold_crossings.nwb) instead "
             "of the sorted neural.nwb (the sorted file is used by default, or the threshold "
             "crossings automatically, with a warning, when it is missing).")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/<session>/prehension_plots/figure_depth_activity_<align>.png).")

    args = parser.parse_args(args=argv)

    start_time = time.time()
    figure_depth_activity(
        args.server, args.processed_server, args.session, align_timepoint=args.align,
        before=args.before, after=args.after, filter_sigma=args.filter_sigma,
        n_depth_bins=args.n_depth_bins, stack_by_depth=args.stack_by_depth,
        only_good=args.only_good, min_rate=args.min_rate,
        drift_correct=args.drift_correct, use_threshold_crossings=args.threshold_crossings,
        save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
