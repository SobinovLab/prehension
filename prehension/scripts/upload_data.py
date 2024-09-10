#!python3.7
import argparse
import datetime
import os
import time

from prehension import preset
from prehension import tools
from prehension.upload_data import upload_data


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uploads the data from local server to server accessible to collaborators.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    default_target_dir = os.path.join(os.environ['USERPROFILE'], 'Box', 'PrehensionProject')
    parser.add_argument(
        '--target_dir',
        type=str, default=default_target_dir,
        help='Where to upload the data. Default: {}'.format(default_target_dir))

    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Do not copy the data, only print out the files to be copied.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    upload_data(args.server, args.sessions, args.temp, args.target_dir, args.dry_run, args.overwrite)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
