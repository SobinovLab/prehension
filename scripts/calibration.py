#!python3.7
import os
import argparse
import shutil
import glob
import time
import datetime
import tqdm
import matplotlib.pyplot as plt
import ncams

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import meta_session



def main(server, sessions, temp, overwrite, relocate, run_extrinsic_calibration):

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
                ncams_config = ncams.camera_io.yaml_to_config(
                    mstruct['ncams_config'], overwrite_setup_path=True)

                # load intrinsics config
                intrinsics_config = ncams.camera_io.import_intrinsics(ncams_config)

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



if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Copies session extrinsic calibration images into their own directory and '
                     'runs extrinsic calibration for each session.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'temp', 'overwrite'))

    parser.add_argument(
        '--relocate',
        action='store_true',
        help='Copies extrinsic calibration from "cameras" folder into "calibration/extrinsic"'
        ' directory.')
    parser.add_argument(
        '--run_extrinsic_calibration',
        action='store_true',
        help='Runs local extrinsic calibration in the session directory.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.temp, args.overwrite, args.relocate,
         args.run_extrinsic_calibration)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    # show the resulting calibrations
    if args.run_extrinsic_calibration:
        plt.show()
