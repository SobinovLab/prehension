#!python3.7
import os
import argparse
import time

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs
from prehension import io_tools
from prehension import meta_session


def main(server, session, trial_number, temp, overwrite, executable_filename):
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(session) == 0:
        session = meta_session.find_session_dirs(server)[0]

    rs('Processing session {}.'.format(session))
    server_session = os.path.join(server, session)

    if not os.path.exists(server_session):
        ValueError('Session {} does not exist on the server.'.format(session))

    # load session meta
    mstruct, _, _, msession = meta_session.load_meta_information(server_session)

    # find trial
    trial = meta_session.find_trial(msession, trial_number)
    if trial is None:
        ValueError('Could not find trial #{}.'.format(trial_number))

    # find the frame
    optimal_frames = io_tools.import_optimal_frames(
        os.path.join(server_session, 'optimal_frames.csv'))
    frame = optimal_frames[trial_number]

    command = ('{executable_filename} --manual --verbose -m "{model_filename}" '
               '--ja_in "{ja_filename}" '
               '--frame {frame} '
               '--leps_in "{leps_in}" --rips_in "{reps_in}" '
               '--adj "{adjustment_filename}"'.format(
                   executable_filename=executable_filename,
                   model_filename=mstruct['mujoco_model_sensorized'],
                   # model_filename=mstruct['mujoco_model'],  # can be used instead
                   ja_filename=trial.post_kinematic_filename_csv,
                   frame=frame,
                   leps_in=trial.get_post_ps_filenames()['medial_sensor'],
                   reps_in=trial.get_post_ps_filenames()['lateral_sensor'],
                   adjustment_filename=trial.adjustment_kinematic_filename))
    rs('Executing command:')
    rs(command)
    os.system(command)


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    executable_filename = os.path.join(
        '../../stereo_inverse_kinematics', 'mjc_vs_code',
        'MuJoCoInverseDynamics', 'x64', 'Debug', 'MuJoCoInverseDynamicsProject.exe')

    parser = argparse.ArgumentParser(
        description=('Create an adjustment to the position of the pressure sensor in MuJoCo.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('session', 'trial', 'temp', 'overwrite'))

    # other
    parser.add_argument(
        '--executable_filename',
        type=str, default=executable_filename,
        help='Filename of the executable MuJoCo file. Default: {}.'.format(executable_filename))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.session, args.trial, args.temp, args.overwrite,
         args.executable_filename)

    rs('Program took {} s.'.format(time.time() - start_time))
