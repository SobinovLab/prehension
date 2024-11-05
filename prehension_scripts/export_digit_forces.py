#!python3.7
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension import tools
from prehension.matching.export_digit_forces import export_digit_forces


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Compare manually-labeled to the automatically-labeled forces using sensor'
                     ' masks.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite', 'processes'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    export_digit_forces(args.server, args.sessions, args.trials, args.temp, args.overwrite, args.processes)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
