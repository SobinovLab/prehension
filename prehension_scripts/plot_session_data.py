#!python3
# -*- coding: utf-8 -*-
"""
Create plots for many sessions.

Copyright (C) 2019-2024 Anton Sobinov, Caleb Raman
https://github.com/BensmaiaLab/prehension

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
import time
import os
import sys
import matplotlib as mpl

from tqdm import tqdm
from datetime import timedelta
from prehension_presets.prehension_presets import PRESETS
from prehension.tools.logs import rs, setup_logging
from prehension.tools import cmd_args
from prehension.visualization.session_data_visualization import SessionWrapper, SessionGroup
from prehension.tools.session_management import (does_training_servers_exist,
                                                 fetch_server_session_dirs)

# --- future features ---
# Plot performance should consider all sessions with red dot for experimental sessions
# Add slight x offset for training on same data
# Add transfer
# Add overwrite


def main(preset, sessions, temp, overwrite, transfer=True):
    """Does the following steps:
    1. Moves sessions from experiment to training if they don't have both cam and sensor data
    2. Groups these sessions into experimental sessions and training sessions
    3. For each group make the following plots
        3.1. Force traces
        3.2. Condition success x2(this session and all previous sessions in same group)
        3.3 Performance x2 (all time and last 10 days)

    Args:
        preset (dict): the preset in question
        sessions (list[str]): the sessions to process, if empty, process all
        temp (dir): the temporary directory for logging, usually C:\tmp
        overwrite (bool): if true, overwrite existing files
        transfer (bool, optional): if true, transfer sessions from experiment to training.
            This can be time consuming.Defaults to True.

    Raises:
        ValueError: If either the raw or processed servers do not exist
    """

    # Check both raw and processed servers exist
    if not os.path.exists(preset["default_server"]):
        raise ValueError(f"Cannot find default server: {preset['default_server']}")

    if not os.path.exists(preset["processed_server"]):
        raise ValueError(f"Cannot find processed server: {preset['processed_server']}")

    # Setup logging
    setup_logging(temp, sessions_dir=preset["processed_server"])

    # Check if preset defines training servers, if so we try transferring sessions to training,
    has_training_server = does_trianing_servers_exist(preset)

    # Move sessions from experiment to training
    if transfer:
        rs("\n" * 3 + "=" * 200)
        if has_training_server:
            rs("Moving training sessions")
            experimental_ss_pairs_pre_move, _ = fetch_server_session_dirs(
                preset, sessions, filter=False)
            for sw in tqdm([SessionWrapper(*exp_pair) for exp_pair in
                            experimental_ss_pairs_pre_move]):
                sw.ensure_transfer_to_training_server(
                    preset["default_training_server"],
                    preset["processed_training_server"],
                    overwrite)
            rs("\n" * 3 + "=" * 200)
        else:
            rs(f"No training servers defined for preset {preset['names'][0]}, skipping transfer"
                " step.")

    # Get raw/processed session dirs for preset
    # Note: we want to fetch all sessions here for performance analysis
    experimental_ss_pairs, training_ss_pairs = fetch_server_session_dirs(preset, filter=True)

    exp_session_wrappers = [SessionWrapper(*exp_pair) for exp_pair in experimental_ss_pairs]
    train_session_wrappers = [SessionWrapper(*train_pair) for train_pair in training_ss_pairs]

    exp_grp = SessionGroup(exp_session_wrappers, group_label="Experiment")
    train_grp = SessionGroup(train_session_wrappers, group_label="Training")

    exp_grp.do_all_analysis(overwrite, sessions=sessions)
    rs("\n" * 3 + "=" * 200)
    train_grp.do_all_analysis(overwrite, sessions=sessions)


# Entry
if __name__ == "__main__":
    preset_name = sys.argv[1]
    if preset_name not in PRESETS.keys():
        raise ValueError(f"preset_name {preset_name} not found in presets {list(PRESETS.keys())}")

    current_preset = PRESETS[preset_name]

    # Remove axis spines from plot
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False

    # Add arguments
    parser = argparse.ArgumentParser(description="Create plots for a given monkey")

    cmd_args.add_default_arguments(parser, ("sessions", "temp", "overwrite"))

    args = parser.parse_args(sys.argv[2:])
    start_time = time.time()

    main(current_preset, args.sessions, args.temp, args.overwrite)

    rs("Program took {}.".format(timedelta(seconds=time.time() - start_time)))
