#!python3.7
import argparse
import datetime
import time

from prehension import preset
from prehension import tools
from prehension.find_event_onsets import find_event_onsets
from prehension.tools import rs

# ============================================ Notes ============================================= #
# Lint with: py -3.7 -m pycodestyle find_event_onsets.py --max-line-length 100 --ignore E402
# - trial_id - same as meta_session
# - shoulder_movement_onset - onset of movement of shoulder joints
# - elbow_movement_onset
# - wrist_movement_onset
# - fingers_movement_onset
# - maximum_aperture - we will have to try a few, but probably start
#     with the sum of index and thumb joints as approximation of aperture
# - grasp_start - use the same threshold as is written out in `align_`
#     script, it was something like 1% of 95th percentile
# - fingers_static - when fingers stop moving after grasp starts
# - release_start - when fingers leave the static posture. These two
#     might be hard to get reliably. If it is too hard, we will not have them.
#     Do not focus on them in the start.
# - release - same as grasp_start approach, just the last point
# - hand_retreated - when the arm returns to the resting position
#     (shoulder and elbow kinematics close to beginning)
# - (boolean) regrasp - should be 1 if the force drops to 0 at any
#     point between grasp and release.

# =========================================== Examples =========================================== #
# Example call:
# py find_event_onsets.py cr_test --server "C:\PrehensionDataLocal\MojitoRightHemisphere"
# --sessions 2022_04_19_Set1 --overwrite --processes 1

# Server:
# py find_event_onsets.py cr_test --server
# "S:\ProjectFolders\Prehension\Data\MojitoRightHemisphere\sessions"
#  --sessions 2022_04_27_Set1 --trials 34 --overwrite

# py find_event_onsets.py cr_test --server
#  "S:\ProjectFolders\Prehension\Data\MojitoRightHemisphere\sessions"
#  --sessions 2022_04_27_Set1 --overwrite --processes 1 --make_plots


if __name__ == "__main__":
    # Add arguments
    preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=("Outputs a csv of movement onset times for each session")
    )

    tools.add_default_kwarguments(
        parser, {"server": current_preset["default_server"]})
    tools.add_default_arguments(
        parser, ("sessions", "trials", "temp", "overwrite", "processes", "make_plots")
    )

    parser.add_argument(
        "--store_plots",
        action='store_true',
        help="Save plots to disk.")
    parser.add_argument(
        "--dont_show_plots",
        action='store_true',
        help="Does not show the generated plots.")
    parser.add_argument(
        '--make_trial_plots',
        action='store_true',
        help='Makes more inspection figures.')

    # Plot logic note:
    # When running the first time if make_plots is not set
    # We only create the timepoints csv
    # If a future run specifies make_plots (timepoints.csv existing)
    # Then we want to create the timepoints folder and plots per row of csv

    args = parser.parse_args(args=argv)

    start_time = time.time()
    find_event_onsets(
        args.server,
        args.sessions,
        args.trials,
        args.temp,
        args.overwrite,
        args.processes,
        args.make_plots,
        args.store_plots,
        args.make_trial_plots,
        not args.dont_show_plots,
        current_preset
    )
    rs("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))
