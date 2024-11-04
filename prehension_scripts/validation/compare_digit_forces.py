#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.validation import compare_digit_forces


if __name__ == "__main__":
    raise DeprecationWarning()
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=("Compare manually-labeled to the automatically-labeled digit forces.")
    )
    tools.add_default_kwarguments(parser, {"server": current_preset["default_server"]})
    tools.add_default_arguments(parser, ("sessions", "trials", "temp", "make_plots"))

    parser.add_argument(
        "--find_good", action="store_true", help="Find good trials - candidates for labeling."
    )
    parser.add_argument(
        "--find_good_n",
        type=int,
        default=20,
        help="Default number of random good trials to select from a session.",
    )

    args = parser.parse_args(args=argv)

    start_time = time.time()
    compare_digit_forces(
        args.server,
        args.sessions,
        args.trials,
        args.temp,
        args.find_good,
        args.make_plots,
        args.find_good_n,
    )
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
