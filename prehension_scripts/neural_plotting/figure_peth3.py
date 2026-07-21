#!python3
# -*- coding: utf-8 -*-
"""
Plot peri-event time histograms (PETH) for a session from its neural NWB.

Same as neural_plotting.figure_peth, but paired with the updated export_nwb whose
Units + ttl_pulses are produced exactly like figure_peth2 (sorter output + direct
Open Ephys TTL windows).  Reading the NWB here therefore reproduces the
figure_peth2 figure from a single portable file.

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
from prehension.neural_processing import config
from prehension.neural_plotting.figure_peth3 import plot_perievent_histograms

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description="PETH traces from the neural NWB (figure_peth2 data path), "
                    "coloured by an object property.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("session",))
    parser.add_argument(
        "--units", type=str, default=None, nargs="*", metavar="UNIT_ID",
        help="Unit ids to plot. If empty, plot all units.")
    parser.add_argument(
        "--align", type=str, default="first_grasp_start",
        help="Trial timepoint to align to. Default: first_grasp_start.")
    parser.add_argument(
        "--group_column", type=str, default="targetForce(N)",
        help="Object property to colour-code by. Default: targetForce(N).")

    args = parser.parse_args(args=argv)

    # probe type is read from the session meta_structure ('neural' field), not the command line
    probe_type = config.probe_type_from_meta(args.server, args.processed_server, args.session)

    start_time = time.time()
    plot_perievent_histograms(
        args.server, args.processed_server, args.session, probe_type,
        neuron_ids=args.units, align_timepoint=args.align, group_column=args.group_column)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
