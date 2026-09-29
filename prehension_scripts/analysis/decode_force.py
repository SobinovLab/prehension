#!python3
# -*- coding: utf-8 -*-
"""
Wiener / Kalman filter decoding of the total grasp force from neural activity, per session.

For each session, cross-validates a linear force decoder that estimates the total grasp force
(summed over all sensels of both filtered pressure sensors) through time from the population
firing rates, holding out whole trials (5-fold by default).  The decoder is a Wiener / linear-FIR
filter by default, or a Kalman filter with --method kalman.  Every held-out trial is scored with a
per-trial pseudo-R2 (coefficient of determination of the decoded vs actual force).  With --n_pcs
the neural population is first reduced to its top principal components (fit per fold on the
training trials only).

Two figures are saved per session to <session>/prehension_plots/: the held-out actual-vs-decoded
force traces (one panel per trial) and the per-trial pseudo-R2 distribution overlaid with a
neural<->force time-alignment shuffle (a chance null).  Decoding runs on the active-touch period
(active_grasp: first grasp -> release) by default; --period exposes the other sub-periods, as in
the encoding models.  The neural source is the sorted neural.nwb, or the threshold crossings with
--threshold_crossings.

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
from prehension.tools.decoding import DECODE_METHODS
from prehension.analysis.encoding import PERIODS
from prehension.analysis.decode_force import (
    decode_force, DEFAULT_METHOD, DEFAULT_PERIOD, DEFAULT_N_PCS, DEFAULT_WIENER_HISTORY_S)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Per-session Wiener / Kalman filter decoding of the total grasp force (summed "
                    "filtered pressure-sensor force) from neural activity through time; 5-fold "
                    "cross-validation holding out whole trials. Saves a held-out actual-vs-decoded "
                    "trace figure and a per-trial pseudo-R2 distribution figure (with a "
                    "time-shuffled null) per session.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "drift_correct"))
    parser.add_argument(
        "--method", choices=DECODE_METHODS, default=DEFAULT_METHOD, metavar="METHOD",
        help="Decoder: 'wiener' (linear FIR filter, default) or 'kalman' (Wu et al. 2006). "
             "Default: {}.".format(DEFAULT_METHOD))
    parser.add_argument(
        "--n_folds", type=int, default=5, metavar="K",
        help="Trial-wise cross-validation folds (whole trials held out). Default: 5.")
    parser.add_argument(
        "--n_pcs", type=int, nargs="?", const=DEFAULT_N_PCS, default=None, metavar="N",
        help="Decode from the top N principal components of the neural population instead of the "
             "raw units (a scaler + PCA fit per fold on the training trials only). Pass --n_pcs "
             "for the default top {} PCs, or --n_pcs N for a different count. Default: off.".format(
                 DEFAULT_N_PCS))
    parser.add_argument(
        "--wiener_history", type=float, default=DEFAULT_WIENER_HISTORY_S, metavar="SECONDS",
        help="Wiener filter only: neural history length in seconds, converted to preceding bins "
             "per session (round(history / bin_width)); an FIR filter over recent activity, 0 is "
             "instantaneous. Default: {:.3g} ({:.0f} ms).".format(
                 DEFAULT_WIENER_HISTORY_S, DEFAULT_WIENER_HISTORY_S * 1e3))
    parser.add_argument(
        "--period", choices=PERIODS, default=DEFAULT_PERIOD, metavar="PERIOD",
        help="Trial sub-period to decode over: 'active_grasp' (the active-touch period, first "
             "grasp -> release; default), 'active_movement' (earliest movement onset -> hand "
             "retreat) or 'all' (whole trial). Default: {}.".format(DEFAULT_PERIOD))
    parser.add_argument(
        "--bin_width", type=float, default=None, metavar="SECONDS",
        help="Firing-rate / force bin width. Default: the session kinematic/video period 1/fps.")
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Random seed for the cross-validation fold split and the time-shuffle null. "
             "Default: None (random each run).")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Decode from the threshold-crossing channels (neural_threshold_crossings.nwb) "
             "instead of the sorted units.")
    parser.add_argument(
        "--include_unsuccessful", dest="successful_only", action="store_false",
        help="Also decode unsuccessful trials. By default only successful trials are used.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into each "
             "<session>/prehension_plots/).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    start_time = time.time()
    decode_force(
        args.server, args.processed_server, sessions, method=args.method, n_folds=args.n_folds,
        n_pcs=args.n_pcs, wiener_history=args.wiener_history, bin_width=args.bin_width,
        use_threshold_crossings=args.threshold_crossings, period=args.period,
        drift_correct=args.drift_correct, successful_only=args.successful_only, seed=args.seed,
        save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
