#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.kinematics.find_static_thorax_position import find_static_thorax_position


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Calculates and saves median base body position (thorax) in the'
                     ' OpenSim model.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    find_static_thorax_position(args.server, args.sessions, args.trials, args.temp, args.processes, args.overwrite)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
