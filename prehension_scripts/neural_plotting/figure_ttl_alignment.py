#!python3
# -*- coding: utf-8 -*-
"""
Plot the neural TTL sync pulses over the behavioural trial windows for a session,
aligned by the first pulse-trial, to find/set the pulse<->trial alignment.

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
from prehension.neural_processing.common import probe
from prehension.neural_plotting.figure_ttl_alignment import plot_ttl_trial_alignment

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="Plot TTL pulses over trial windows, aligned by the first pulse-trial.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session",))
    parser.add_argument(
        "--skip", type=int, default=None,
        help="Pulses to skip for alignment. If negative, that many trials are "
             "skipped instead. Default: from meta_neural.json 'skip_ttl' (then 0).")
    parser.add_argument(
        "--ignore", type=int, default=None, metavar="N",
        help="Drop the first N TTL pulses and first N trial starts before aligning "
             "(applied before --skip). Default: from meta_neural.json 'ignore' (then 0).")
    parser.add_argument(
        "--recording", type=str, default=None, metavar="N",
        help="Open Ephys recording within experiment1 to read, 1-based "
             "(Recording1, Recording2, ...); accepts 2 or 'Recording2'. "
             "Default: probe default (vprobe Recording1, neuropixels Recording2).")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figure (saving is on by default, into "
             "<processed_server>/<session>/prehension_plots/figure_ttl_alignment.png).")

    args = parser.parse_args(args=argv)

    # Utah arrays are externally sorted and have no Open Ephys TTL stream to extract; the
    # reexported ttl_pulses come from the source, so this Open Ephys alignment diagnostic
    # does not apply.
    if probe.is_utah_session(args.server, args.processed_server, args.session):
        print("Session {} is a Utah array (externally sorted); TTL alignment is an Open "
              "Ephys diagnostic and does not apply. Skipping.".format(args.session))
    else:
        # probe type is read from the session meta_structure ('neural' field), not the CLI
        probe_type = probe.probe_type_from_meta(args.server, args.processed_server, args.session)
        start_time = time.time()
        plot_ttl_trial_alignment(args.server, args.processed_server, args.session,
                                 probe_type, skip=args.skip, ignore=args.ignore,
                                 recording=args.recording, save=args.save)
        print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
        plt.show()
