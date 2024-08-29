#!python3.7
import os
import sys
import inspect
import argparse
import shutil
import time
import datetime
import copy
import tqdm
import ncams
from reporting_pool import ReportingPool

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import io_tools
from prehension import meta_session

def new_make_video(frame_filenames, filename_ou, rate):
    # otherwise overwrites logging
    import ffmpegio
    # if the frame is empty, replace is with the next one
    # if there are no more non-empty frames, replace it with the previous one
    for i_ff in range(len(frame_filenames)):
        if os.path.getsize(frame_filenames[i_ff]) == 0:
            replace_i_ffs = [i_ff]
            i_ff2 = None
            # go forward until non-empty file
            for i_ff2 in range(i_ff+1, len(frame_filenames)):
                if os.path.getsize(frame_filenames[i_ff2]) != 0:
                    break
                replace_i_ffs.append(i_ff2)
            # check if found non-empty one
            if i_ff2 is not None and os.path.getsize(frame_filenames[i_ff2]) != 0:
                non_empty_i_ff = i_ff2
            else:
                non_empty_i_ff = i_ff - 1
            # if valid place, replace
            if non_empty_i_ff >= 0:
                for r_i_ff in replace_i_ffs:
                    frame_filenames[r_i_ff] = frame_filenames[non_empty_i_ff]

    ffconcat = ffmpegio.FFConcat()
    ffconcat.add_files(frame_filenames)
    with ffconcat:  # generates temporary ffconcat file
        ffmpegio.transcode(
            ffconcat, filename_ou,
            f_in='concat', safe_in=0, r_in=rate,
            overwrite=True,  # y=None,  # overwrite
            hide_banner=None, loglevel='error',  # show_log=False,  # printing to stdout
            an=None, an_in=None,  # no sound for output or input
            **{
                'r': rate,
                'codec:v': 'h264_nvenc',
                'pix_fmt': 'rgb0',
                'preset:v': 'hq',
                'profile:v': 'high',
                'rc:v': 'vbr', 'b:v': 0,
                'cq:v': 23,
                'coder': 'cabac',
                'maxrate': '80000k',
            }
        )



def make_video(trial, mstruct, clean):
    camera_serials = list(mstruct['cameras'].keys())
    # HACK because of how images_to_video function was written
    dirname_ou = os.path.split(trial.videos[camera_serials[0]])[0]

    os.makedirs(dirname_ou, exist_ok=True)

    for camera_serial in camera_serials:
        # make a video
        frame_filenames = ncams.utils.get_image_list(path=trial.images_dirnames[camera_serial])
        if len(frame_filenames) == 0:
            ws(f'Folder {trial.images_dirnames[camera_serial]} has no images.')
            continue
        # filename_ou = os.path.split(trial.videos[camera_serial])[1]

        # ncams.image_tools.images_to_video(frame_filenames, filename_ou,
        #                                   fps=mstruct['fps'], output_folder=dirname_ou, logger=None)
        new_make_video(frame_filenames, trial.videos[camera_serial], mstruct['fps'])

        # copy log
        shutil.copy2(trial.images_logs[camera_serial], trial.videos_logs[camera_serial])

        if False:  # clean:
            shutil.rmtree(trial.images_dirnames[camera_serial])


def main(server, sessions, trials_sel, temp, processes, overwrite, clean):
    tools.setup_logging(temp, sessions_dir=server)

    # To enable multiple videos encoding at the same time, use
    # https://github.com/keylase/nvidia-patch/tree/master
    # ws('Forcing processes to 1 since moved to GPU.')
    # processes = 1

    if clean:
        ws('Coding lock of "clean" function. If you do not know how to remove it, you should try'
           ' to use it.')
        clean = False

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    failed_trial_reports = []
    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(
                server_session, only_successful_trials=False)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # accumulate data
        trials = []
        for trial in tqdm.tqdm(msession, ncols=100, desc='Finding trials'):
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_images_dirs_files_exist():
                continue
            if not overwrite and trial.do_videos_files_exist():
                continue
            trials.append(trial)

        print()
        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        # results will go here
        os.makedirs(mstruct['videos_dir'], exist_ok=True)

        p_args = list(zip(*[
            trials,
            [copy.deepcopy(mstruct) for _ in trials],
            [clean for _ in trials]
        ]))

        if len(p_args) > 0:
            pool = ReportingPool(make_video, p_args, processes=processes,
                                 report_on_change=True, track_failures=True)
            pool.start()

            if len(pool.failed_i_jobs) > 0:
                print()
                ws('Failed trials:')
                for v in pool.failed_i_jobs:
                    ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append('session {} trial {} error: {}'.format(
                        session, trials[v].trial_number, pool.error_reports[v]))

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Transforms images from a session into video.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    parser.add_argument(
        '--clean',
        action='store_true',
        help='DANGER! Remove directories from the server that were converted into videos.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp, args.processes,
         args.overwrite, args.clean)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
