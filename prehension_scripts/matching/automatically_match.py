#!python3.7
import argparse
import os
import time

from prehension import preset
from prehension import tools
from prehension.matching.automatically_match import automatically_match
from prehension.tools import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    executable_filename = os.path.join(
        '../../stereo_inverse_kinematics', 'mjc_vs_code',
        'MuJoCoInverseDynamics', 'x64', 'Debug', 'MuJoCoInverseDynamicsProject.exe')

    parser = argparse.ArgumentParser(
        description=('Automatically matches sensels with hand segments using MuJoCo program.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    # other
    parser.add_argument(
        '--executable_filename',
        type=str, default=executable_filename,
        help='Filename of the executable MuJoCo file. Default: {}.'.format(executable_filename))
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the grasping motion. Disables parallel execution.')
    parser.add_argument(
        '--skip_export',
        action='store_true',
        help='Does not export the results. Useful when just trying to visualize the trial,'
        ' instead of specifying --overwrite.')
    parser.add_argument(
        '--write_video',
        action='store_true',
        help='Write video during force matching simulation, when running.'
    )
    parser.add_argument(
        '--quality_threshold',
        type=float, default=0.1,
        help='If the unmatched force exceeds this portion of total force, the trial will throw'
        ' an error. Useful to detect when the model is actually breaking bad.')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose prints about program running.')

    args = parser.parse_args(args=argv)

    if (not args.visualize and args.write_video):
        raise Exception("Cannot specify --write_video without --visualize")

    start_time = time.time()
    automatically_match(
         args.server, args.sessions, args.trials, args.temp, args.processes,
         args.overwrite,
         args.executable_filename,
         args.visualize,
         args.skip_export,
         args.write_video,
         args.quality_threshold,
         args.verbose)

    rs('Program took {} s.'.format(time.time() - start_time))
