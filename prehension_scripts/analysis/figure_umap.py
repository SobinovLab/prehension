#!python3
# -*- coding: utf-8 -*-
"""
Plot per-session UMAP embeddings of single-trial state at three moments of the reach-to-grasp.

For each session and modality (kinematics = joint-angle position, segment forces, torques, or
neural activity), embeds the trials in 2D with UMAP (umap-learn) using each trial's features at
three timepoints -- the beginning of the movement, the initial grasp, and the middle of the
grasp -- and draws a grid of modalities (rows) x timepoints (columns).  The base figure colours
points by target force (the repo's yellow->red colormap) and marker-shapes them by kinematic
condition (object geometry); a purely force-coloured figure (one marker) and one figure per
object parameter that varies (aperture, tilt, ...) recolour the same embeddings.  All modalities
are embedded by default; pass --modality to run only some.

The figures are saved per session to <session>/prehension_plots/.  Timepoints come from
timepoints.csv (create_timepoints); the neural source is the sorted neural.nwb, or the
threshold crossings with --threshold_crossings.

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
from prehension.neural_processing.common.spikes import GROUP_COLUMN
from prehension.analysis.figure_umap import (
    figure_umap, MODALITIES, NEURAL_WINDOW_S, UMAP_N_NEIGHBORS, UMAP_MIN_DIST)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Per-session UMAP embeddings of single-trial state (kinematics, segment "
                    "forces, torques, neural) at the movement onset, initial grasp and mid "
                    "grasp; a force-coloured figure (marker = kinematic condition), a purely "
                    "force-coloured figure, plus one figure per varying object parameter "
                    "(aperture, tilt, ...).")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "drift_correct"))
    parser.add_argument(
        "--modality", dest="modalities", nargs="+", default=list(MODALITIES), choices=MODALITIES,
        metavar="MODALITY",
        help="Modalities to embed; pass one to run only that modality, e.g. --modality neural. "
             "Default (all): {}.".format(' '.join(MODALITIES)))
    parser.add_argument(
        "--group_column", type=str, default=GROUP_COLUMN,
        help="Object property used for the point colour (force). Default: {}.".format(GROUP_COLUMN))
    parser.add_argument(
        "--window", type=float, default=NEURAL_WINDOW_S, metavar="SECONDS",
        help="Window for the single-timepoint neural spike-count rate. "
             "Default: {}.".format(NEURAL_WINDOW_S))
    parser.add_argument(
        "--n_neighbors", type=int, default=UMAP_N_NEIGHBORS, metavar="N",
        help="UMAP n_neighbors (capped at n_trials - 1). Default: {}.".format(UMAP_N_NEIGHBORS))
    parser.add_argument(
        "--min_dist", type=float, default=UMAP_MIN_DIST, metavar="D",
        help="UMAP min_dist. Default: {}.".format(UMAP_MIN_DIST))
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Use the threshold-crossing channels (neural_threshold_crossings.nwb) instead of "
             "the sorted units for the neural modality.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into each "
             "<session>/prehension_plots/).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    figure_umap(
        args.server, args.processed_server, sessions, modalities=args.modalities,
        group_column=args.group_column, window=args.window, n_neighbors=args.n_neighbors,
        min_dist=args.min_dist, use_threshold_crossings=args.threshold_crossings,
        drift_correct=args.drift_correct, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
