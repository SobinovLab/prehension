#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.matching.prepare_mujoco_model import prepare_mujoco_model


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Generates a mask of pressure sensors matrix that highlights activated '
                     'sensels and tessellates model sensors based on it.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite'))

    parser.add_argument(
        '--dont_make_mask',
        action='store_false', dest='make_mask',
        help='Converts.')
    parser.add_argument(
        '--dont_tessellate',
        action='store_false', dest='tessellate',
        help='Tessellates the pressure sensors into sensels.')
    parser.add_argument(
        '--sense_distance',
        type=float, default=0.025,
        help='Distance between geom centers for "contact" calculation. Larger values slow down the'
        ' execution, but low values are too short for relatively large bending bones like'
        ' metacarpals and large muscle areas like thenar eminence. In meters.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    prepare_mujoco_model(
        args.server, args.sessions, args.trials, args.temp, args.overwrite,
        args.make_mask, args.tessellate, args.sense_distance)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
