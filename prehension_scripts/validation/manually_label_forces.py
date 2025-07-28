#!python3
import argparse
import time

from prehension import preset
from prehension import tools
from prehension.validation.manually_label_forces import manually_label_forces
from prehension.tools.logs import rs


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    if current_preset['hand'] == 'right':
        lps_ref_camera_name = 'cam19340298'
        rps_ref_camera_name = 'cam19194005'
    else:
        lps_ref_camera_name = 'cam19194005'
        rps_ref_camera_name = 'cam19340396'

    parser = argparse.ArgumentParser(
        description=('Opens a GUI to manually assign sensels to digits.'))
    tools.cmd_args.add_default_arguments(
        parser, ('session', 'trial', 'temp'))

    parser.add_argument(
        '--show_automatic',
        action='store_true')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    manually_label_forces(
        current_preset['default_server'],
        current_preset['processed_server'],
        args.session, args.trial, args.temp,
        lps_ref_camera_name, rps_ref_camera_name, args.show_automatic)

    rs('Program took {} s.'.format(time.time() - start_time))
