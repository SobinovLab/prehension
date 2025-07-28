#!python3
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension.tools import cmd_args
from prehension.validation.compare_masked_forces import compare_masked_forces


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Compare manually-labeled to the automatically-labeled forces using sensor'
                     ' masks.'))
    cmd_args.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'make_plots'))

    parser.add_argument(
        '--find_good',
        action='store_true',
        help='Find good trials - candidates for labeling.')
    parser.add_argument(
        '--find_good_n',
        type=int, default=20,
        help='Default number of random good trials to select from a session.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    compare_masked_forces(
        current_preset['default_server'], current_preset['processed_server'],
        args.sessions, args.trials, args.temp, args.find_good,
        args.make_plots, args.find_good_n)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
