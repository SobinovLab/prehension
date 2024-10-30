#!python3.11

# CLI for displaying status of prehension analyses
# from prehension import tools, meta_session


import sys
import argparse

from prehension.tools import cmd_args
from prehension.visualization import session_visualization
from prehension_presets.prehension_presets import PRESETS
from prehension.tools.utils import fetch_server_session_dirs
from prehension_scripts.plot_session_data import SessionWrapper
from colorama import init


preset_names = ['daiquiri_right_hemisphere',
                'pimms_left_hemisphere_training_k1', 'pappy_left_hemisphere_training_v1',]


def main(args):
    """
    Helper script for displaying sessions (raw & processed) in experiment and training groups
    across the presets in preset names. Option to transfer sessions to training server if
    --clean is passed.
    """
    for preset_name in preset_names:
        print(preset_name.upper())
        init(autoreset=True)  # for colorama

        if preset_name not in PRESETS.keys():
            raise ValueError(
                f'preset_name {preset_name} not found in presets {list(PRESETS.keys())}')

        current_preset = PRESETS[preset_name]
        # Get raw/processed session dirs for preset
        experimental_ss_pairs, training_ss_pairs = fetch_server_session_dirs(current_preset,
                                                                             sessions=[],
                                                                             filter=False)

        exp_session_wrappers = [SessionWrapper(
            *exp_pair) for exp_pair in experimental_ss_pairs]
        train_session_wrappers = [SessionWrapper(
            *train_pair) for train_pair in training_ss_pairs]

        session_visualization.display_session_info(exp_session_wrappers,
                                                   train_session_wrappers, args.clean, args.last)
        print()
        print()


# Entry
if __name__ == "__main__":

    # Add arguments
    parser = argparse.ArgumentParser(
        description=("Display session info about a given monkey"))

    parser.add_argument(
        '--clean', action='store_true',
        help=('Remove session folders if the log file is found somewhere else and there is'
              ' no sensor data in raw'),
        default=False
    )

    parser.add_argument(
        '--last', type=int, default=-1,
        help=('Number of sessions to display. Default: -1 (all)')
    )

    args = parser.parse_args(sys.argv[1:])
    main(args)
