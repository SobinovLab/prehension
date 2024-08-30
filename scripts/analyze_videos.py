#!python3.8
'''On AS computer runs in py 3.8 environ
'''
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.analyze_videos import analyze_videos

if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uses pretrained machine vision network to label videos.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite'))

    parser.add_argument(
        '--dlc_config_path',
        type=str, default=current_preset['dlc_config_path'],
        help='Location of the DLC config to use. Be sure to use the correct monkey config!')
    parser.add_argument(
        '--dont_analyze',
        action='store_false', dest='analyze',
        help='Do not analyze videos using a DLC network.')
    parser.add_argument(
        '--make_videos',
        action='store_true',
        help='Make videos with the labelled markers. ')

    args = parser.parse_args(args=argv)

    start_time = time.time()

    analyze_videos(args.server, args.sessions, args.trials, args.temp, args.overwrite,
                   args.dlc_config_path, args.analyze, args.make_videos)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
