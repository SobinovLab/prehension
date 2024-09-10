#!python3.7
import glob
import os
import shutil

import tqdm

import ncams
from .. import meta_session
from .. import tools
from ..tools import rs, ws


def calibration(server, sessions, temp, overwrite, relocate, run_extrinsic_calibration):
    """Copies session extrinsic calibration images into their own directory and
    runs extrinsic calibration for each session.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} -- Overwrites the created files if they exist.
        relocate {bool} --- Copies extrinsic calibration from "cameras" folder into "calibration/extrinsic" directory.
        run_extrinsic_calibration {bool} --- Runs local extrinsic calibration in the session directory.
    """

    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct = meta_session.import_meta_structure(server_session)
        except Exception as e:
            ws('Could not load meta structure from session {}, skipping.'.format(session))
            continue

        images_calibration_dir = os.path.join(mstruct['images_dir'], 'calibration')
        dest_extrinsic_calibration_dir = os.path.join(mstruct['calibration'], 'extrinsic')

        if relocate:
            if os.path.exists(images_calibration_dir):
                os.makedirs(dest_extrinsic_calibration_dir, exist_ok=True)

                # get the list of files
                files = glob.glob(os.path.join(images_calibration_dir, '*'))
                dest_files = [os.path.join(dest_extrinsic_calibration_dir, os.path.split(v)[1])
                              for v in files]
                for file, dest_file in zip(files, dest_files):
                    if overwrite or not os.path.exists(dest_file):
                        shutil.copy(file, dest_file)
            else:
                ws('Could not find calibration for session {}, skipping relocation.'.format(
                    session))

        if run_extrinsic_calibration and os.path.exists(dest_extrinsic_calibration_dir):
            extrinsic_calibration_filename = os.path.join(
                mstruct['calibration'], 'extrinsic', 'extrinsic_calib.pickle')
            # check all files existing
            # TODO check if not just jpeg
            if not all([os.path.exists(os.path.join(dest_extrinsic_calibration_dir, cn + '.jpeg'))
                        for cn in mstruct['cameras'].values()]):
                ws('Calibration images missing for session {}.'.format(session))
            elif not overwrite and os.path.exists(extrinsic_calibration_filename):
                pass
            else:
                ncams_config = tools.yaml_to_config(
                    mstruct['ncams_config'], overwrite_setup_path=True)

                # load intrinsics config
                intrinsics_config = tools.import_intrinsics(ncams_config)

                # hack to export extrinsics into different place
                ncams_config['setup_path'] = mstruct['calibration']

                # run the calibration
                extrinsics_config, extrinsics_info = ncams.camera_pose.one_shot_multi_PnP(
                    ncams_config, intrinsics_config, export_full=True, show_extrinsics=True,
                    inspect=True)

                # # specify in the mstruct the local extrinsic calibration
                # mstruct_filename = os.path.join(server_session, 'meta_structure.json')
                # with open(mstruct_filename, 'r') as f:
                #     l_mstruct = json.load(f)
                # l_mstruct['extrinsic_calibration'] = os.path.join(
                #     'calibration', ncams_config['extrinsic_path'],
                #     ncams_config['extrinsic_filename'])
                # with open(mstruct_filename, 'w') as f:
                #     json.dump(l_mstruct, f, sort_keys=True, indent=4)
