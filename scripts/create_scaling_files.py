#!python3.7
import argparse
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension import tools
from prehension.matching.create_scaling_files import create_scaling_files
from prehension.tools import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Create IK and SC files for scaling an OpenSim model.'))
    tools.add_default_arguments(
        parser, ('session', 'trial', 'temp', 'overwrite'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    # other
    parser.add_argument(
        '--period',
        type=float, default=[], nargs=2,
        help='Time period in seconds to use for scaling. If empty, use best estimation.'
        ' Empty by default.')
    parser.add_argument(
        '--transfer_position',
        action='store_true',
        help='Transfer the joint angles that have resulted from IK into the model that is being'
        ' scaled. Different mode of operation, does not generate scaling files.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    create_scaling_files(
        args.server, args.session, args.trial, args.temp,
        args.overwrite, args.period, args.transfer_position)

    rs('Program took {} s.'.format(time.time() - start_time))

    plt.show()
