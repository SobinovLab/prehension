#!python3
# -*- coding: utf-8 -*-
"""
Neuron rate vs summed grasp force cross-correlation figure across sessions.

For each requested session, cross-correlates every neuron's firing rate with the
summed pressure-sensor force in the same trial, restricted to the continuous
active-grasp period (the stable-grasp threshold of matching.process_and_align_data).
Rate and force share the seconds-since-TTL frame, so no alignment timepoint is used;
the lag is explored from --pre_lag to --post_lag seconds, and the neuron window is
widened by the lag range so every lag is evaluated over the full active span.
Correlations are shown as r^2, so a neuron's peak lag is the lag of its strongest
correlation regardless of sign.  Two figures are produced, each with one subplot per
session (titled with region and burr hole from meta_neural.json): the per-neuron r^2
curves with the median peak lag marked, and a density histogram of the per-neuron peak
lag.  Per session, all units are used, or -- with --only_good -- the unit ids in that
session's meta_neural.json 'good_neurons'.  Recording / skip_ttl come from each
session's meta_neural.json; there is no --units / --recording / --skip_ttl.

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
from prehension.neural_plotting.figure_cross_correlation import (
    figure_cross_correlation, PRE_LAG, POST_LAG, FORCE_ACTIVE_FRACTION, MIN_RATE_HZ)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Neuron rate vs summed grasp-force cross-correlation across sessions, "
                    "one subplot per session (titled with region and burr hole), restricted "
                    "to the continuous active-grasp period, with the median peak lag marked.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "processes"))
    parser.add_argument(
        "--pre_lag", type=float, default=PRE_LAG, metavar="SECONDS",
        help="Cross-correlation lag explored before zero. Default: {}.".format(PRE_LAG))
    parser.add_argument(
        "--post_lag", type=float, default=POST_LAG, metavar="SECONDS",
        help="Cross-correlation lag explored after zero. Default: {}.".format(POST_LAG))
    parser.add_argument(
        "--force_fraction", type=float, default=FORCE_ACTIVE_FRACTION, metavar="FRACTION",
        help="Active-grasp threshold as a fraction of each trial's peak summed force. "
             "Default: {}.".format(FORCE_ACTIVE_FRACTION))
    parser.add_argument(
        "--only_good", action="store_true",
        help="Use only each session's meta_neural.json 'good_neurons' (else all units).")
    parser.add_argument(
        "--min_rate", type=float, default=MIN_RATE_HZ, metavar="HZ",
        help="Keep only neurons with mean rate above this (Hz) over the active periods. "
             "Default: {} (the same activity floor as the dPCA figure).".format(MIN_RATE_HZ))
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/pooled_figures/figure_cross_correlation, named after "
             "the --sessions string).")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)
    name = cmd_args.sessions_name_stub(args.sessions)

    start_time = time.time()
    figure_cross_correlation(
        args.server, args.processed_server, sessions,
        pre_lag=args.pre_lag, post_lag=args.post_lag, force_fraction=args.force_fraction,
        only_good=args.only_good, min_rate=args.min_rate, processes=args.processes,
        name=name, save=args.save)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
