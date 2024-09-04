#!python3.7
# Example call:
# py preprocess_pressure_sensors.py cr_local --sessions 2022_04_14_Set1
# py preprocess_pressure_sensors.py cr_test --server "C:\PrehensionDataLocal\MojitoRightHemisphere"
# --sessions 2022_04_19_Set1 --overwrite
# Linting: py -3.7 -m pycodestyle
# --first .\transform_pressure_sensor_data\preprocess_pressure_sensors.py --max-line-length=100
# include local library functions - TB included in NCams
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.preprocess_pressure_sensors import preprocess_pressure_sensors
from prehension.tools import rs


if __name__ == '__main__':
    # Add arguments
    preset_name, current_preset, argv = preset.process_args_for_preset()
    parser = argparse.ArgumentParser(
        description=('Creates meta information for a session.'))

    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    tools.add_default_arguments(parser, ('sessions', 'trials', 'temp', 'overwrite', 'processes'))
    args = parser.parse_args(args=argv)

    start_time = time.time()
    preprocess_pressure_sensors(
        args.server, args.sessions, args.trials, args.temp,
        args.overwrite, args.processes, current_preset)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
