#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.matching.transform_osim_model import transform_osim_model


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Generates a MuJoCo model from an OpenSim model.'))
    tools.add_default_arguments(
        parser, ('session', 'overwrite'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    args = parser.parse_args(args=argv)

    start_time = time.time()
    transform_osim_model(args.server, args.session, args.overwrite)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
