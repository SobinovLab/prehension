#!python3.7
import os

from .. import meta_session
from .. import tools
from ..tools import rs


def make_adjustment(server, session, trial_number, temp, overwrite, executable_filename):
    """Create an adjustment to the position of the pressure sensor in MuJoCo.

    Arguments:
        server {str} --- Folder where the sessions are located.
        session {str} --- Session directory to use.
        trial_number {int} --- Trial to do adjustment on.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        executable_filename {str} --- Filename of the executable MuJoCo file.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(server))

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
    optimal_frames = meta_session.import_optimal_frames(
        os.path.join(server_session, 'optimal_frames.csv')
    )
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
