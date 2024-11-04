#!python3.7
import argparse
import os
import time

from prehension import preset
from prehension import tools
from prehension.matching.make_adjustment import make_adjustment
from prehension.tools import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    executable_filename = os.path.join(
        '../../stereo_inverse_kinematics',
        'mjc_vs_code',
        'MuJoCoInverseDynamics',
        'x64',
        'Debug',
        'MuJoCoInverseDynamicsProject.exe',
    )

    parser = argparse.ArgumentParser(
        description=('Create an adjustment to the position of the pressure sensor in MuJoCo.')
    )
    tools.add_default_kwarguments(parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(parser, ('session', 'trial', 'temp', 'overwrite'))

    # other
    parser.add_argument(
        '--executable_filename',
        type=str,
        default=executable_filename,
        help='Filename of the executable MuJoCo file. Default: {}.'.format(executable_filename),
    )

    args = parser.parse_args(args=argv)

    start_time = time.time()
    make_adjustment(
        args.server, args.session, args.trial, args.temp, args.overwrite, args.executable_filename
    )

    rs('Program took {} s.'.format(time.time() - start_time))
