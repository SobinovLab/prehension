#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.matching.find_optimal_frames import find_optimal_frames


if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=("Find optimal frames that represent a static grasping posture.")
    )
    tools.add_default_kwarguments(parser, {"server": current_preset["default_server"]})
    tools.add_default_arguments(parser, ("sessions", "trials", "temp", "processes", "overwrite"))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    find_optimal_frames(
        args.server, args.sessions, args.trials, args.temp, args.processes, args.overwrite
    )

    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
