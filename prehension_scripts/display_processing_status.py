#!python3
# -*- coding: utf-8 -*-
"""
Script for displaying status of prehension processing steps across presets and sessions

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


import sys
import argparse
import tqdm

from colorama import init

from prehension.visualization import session_processing_status
from prehension_presets.prehension_presets import PRESETS
from prehension.tools import cmd_args
from prehension.tools.session_management import fetch_exp_session_dirs
from prehension.visualization.session_data_visualization import SessionWrapper


PRESET_NAMES = ['mrhem']
# PRESET_NAMES = ['daiquiri_right_hemisphere']


def main(args):
    """
    Helper script for displaying sessions (raw & processed) in experiment and training groups
    across the presets in preset names. Option to transfer sessions to training server if
    --clean is passed.
    """
    if args.preset:
        preset_names = args.preset
    else:
        preset_names = PRESET_NAMES

    if args.sessions and len(preset_names) > 1:
        raise ValueError('Sessions argument can only be provided if a single preset was selected.')

    for preset_name in preset_names:
        print(preset_name)
        init(autoreset=True)  # for colorama

        if preset_name not in PRESETS.keys():
            raise ValueError(f'preset_name {preset_name} not found in presets {PRESETS.keys()}')

        # Get raw/processed session dirs for preset
        experimental_ss_pairs = fetch_exp_session_dirs(PRESETS[preset_name], sessions=args.sessions)

        exp_session_wrappers = []
        for exp_pair in tqdm.tqdm(experimental_ss_pairs, ncols=100, desc='Pooling sessions'):
            exp_session_wrappers.append(SessionWrapper(*exp_pair))

        session_processing_status.report_sessions_processing_status(
            exp_session_wrappers, verbose=0)

        print()


# Entry
if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser(description="Display session info about a given monkey")

    parser.add_argument(
        '--preset',
        type=str, nargs='*',
        help='List presets to use. Defaults to [{}].'.format(', '.join(PRESET_NAMES)))
    cmd_args.add_default_arguments(parser, ("sessions"))

    args = parser.parse_args(sys.argv[1:])
    main(args)
