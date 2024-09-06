#!python3.11
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.predict_points_jarvis import predict_points_jarvis


if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Runs a trained Jarvis model on videos generating 3D points and IK files.'))
    tools.add_default_kwarguments(
        parser, {
            'server': current_preset['default_server'],
            'processed_server': current_preset['processed_server']})
    tools.add_default_arguments(parser, ('sessions', 'temp', 'overwrite', 'trials', 'processes'))

    # # custom
    parser.add_argument(
        '--make_videos',
        action='store_true',
        help='Renders videos with prediction.')
    parser.add_argument(
        '--jarvis_proj',
        type=str, default=current_preset['jarvis_config_path'],
        help='Jarvis project to use.')
    parser.add_argument(
        '--threshold',
        type=float, default=0.4,
        help='Threshold for likelihood of 3D points to be used.')
    parser.add_argument(
        '--dont_predict',
        action='store_true',
        help='Do not run JARVIS on the videos.')
    parser.add_argument(
        '--dont_transform',
        action='store_true',
        help='Do not transform JARVIS files into our format and create IK files.')

    args = parser.parse_args(args=argv)
    start_time = time.time()

    predict_points_jarvis(
        args.server, args.processed_server, args.sessions, args.temp, args.trials,
        args.jarvis_proj, args.threshold, args.overwrite, args.processes,
        not args.dont_predict, not args.dont_transform, args.make_videos)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
