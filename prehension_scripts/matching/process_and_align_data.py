#!python3.7
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension import tools
from prehension.matching.process_and_align_data import process_and_align_data


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Filters, resamples, and aligns pressure sensor and kinematic data to grasp '
                     'onset.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite', 'make_plots'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    process_and_align_data(
        args.server, args.sessions, args.trials, args.temp, args.processes,
        args.overwrite, args.make_plots)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
