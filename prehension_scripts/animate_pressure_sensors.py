#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.animate_pressure_sensors import animate_pressure_sensors


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Shows the forces exerted by each finger as measured manually and matched'
                     ' automatically.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('session', 'trial'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    animate_pressure_sensors(args.server, args.session, args.trial)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
