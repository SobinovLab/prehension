#!python3.7
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension import tools
from prehension.mark_base import mark_base


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Manually label points on macaque torso to find its location once per'
                     ' calibration.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    parser.add_argument(
        '--skip_gui',
        action='store_true',
        help='Do not launch the GUI for labeling.')

    args = parser.parse_args(args=argv)

    # get default (scaling) session if asked
    if len(args.sessions) > 0 and args.sessions[0] == 'scaling':
        args.sessions = [current_preset['scaling']['session']]

    start_time = time.time()
    mark_base(args.server, args.sessions, args.temp, args.overwrite, args.skip_gui)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
