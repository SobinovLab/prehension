#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.pressure_sensors.rename_folders import rename_folders


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Changes the names of some folders in each session.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    rename_folders(args.server, args.sessions, args.temp)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
