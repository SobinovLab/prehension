#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.create_meta import create_meta


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Creates meta information for a session.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(parser, ('sessions', 'temp', 'overwrite'))

    # custom
    parser.add_argument(
        '--dont_export_roms',
        dest='export_roms',
        action='store_false',
        help='Exports range of motion data from OpenSim model into a convenient CSV meta file.'
        ' If this flag is provided, meta_dof is not created.')

    args = parser.parse_args(args=argv)
    start_time = time.time()

    create_meta(
        args.server, args.sessions, args.temp, args.overwrite,
        args.export_roms, current_preset)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
