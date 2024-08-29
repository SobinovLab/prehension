#!python3.8
'''On AS computer runs in py 3.8 environ
'''
import os
import sys
import inspect
import re
import argparse
import uuid
import glob
import shutil
import warnings
import logging
import time
import datetime
import tqdm
import deeplabcut

from reporting_pool import ReportingPool

# include local library functions - TB included in NCams
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0, parentdir)
# from common import preset
# from common import tools
# from common.tools import rs, ws
# from common import io_tools
# from common import meta_session

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import io_tools
from prehension import meta_session


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def parallel_analyze_videos(dlc_config_path, video, markers_2D_dirname):
    # blockPrint()
    deeplabcut.analyze_videos(
        dlc_config_path, video,
        gputouse=0, save_as_csv=True, destfolder=markers_2D_dirname)
    # enablePrint()


def main(server, sessions, trials_sel, temp, overwrite,
         dlc_config_path, analyze, make_videos):

    tools.setup_logging(temp, sessions_dir=server)

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

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # accumulate data
        trials = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_videos_files_exist():
                continue
            trials.append(trial)

        # into these the result will go
        os.makedirs(mstruct['markers_2D_dir'], exist_ok=True)

        for trial in tqdm.tqdm(trials, ncols=100, desc='Trials'):
            if analyze:
                # to prevent duplication bc DLC creates its own filenames
                if trial.do_dlc_files_exist() and overwrite:
                    # remove existing files
                    trial.remove_dlc_files()
                if not trial.do_dlc_files_exist():
                    # p_args = list(zip(*[
                    #     [dlc_config_path for _ in trial.videos],
                    #     [v for v in trial.videos.values()],
                    #     [trial.markers_2D_dirname for _ in trial.videos],
                    # ]))
                    # pool = ReportingPool(parallel_analyze_videos, p_args,
                    #                      processes=len(trial.videos),
                    #                      report_on_change=True, track_failures=True)
                    # pool.start()
                    parallel_analyze_videos(
                        dlc_config_path, list(trial.videos.values()), trial.markers_2D_dirname)

            if make_videos:
                if trial.do_2d_marker_video_files_exist() and overwrite:
                    trial.remove_2d_marker_video_files()
                # os.makedirs(trial.markers_2D_video_dirname, exist_ok=True)
                # have to use the same folder as output, because DLC:
                if not trial.do_2d_marker_video_files_exist():
                    deeplabcut.create_labeled_video(
                        dlc_config_path, list(trial.videos.values()),
                        destfolder=trial.markers_2D_dirname, draw_skeleton=True)


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Uses pretrained machine vision network to label videos.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite'))

    parser.add_argument(
        '--dlc_config_path',
        type=str, default=current_preset['dlc_config_path'],
        help='Location of the DLC config to use. Be sure to use the correct monkey config!')
    parser.add_argument(
        '--dont_analyze',
        action='store_false', dest='analyze',
        help='Do not analyze videos using a DLC network.')
    parser.add_argument(
        '--make_videos',
        action='store_true',
        help='Make videos with the labelled markers. ')

    args = parser.parse_args(args=argv)

    start_time = time.time()

    main(args.server, args.sessions, args.trials, args.temp, args.overwrite,
         args.dlc_config_path, args.analyze, args.make_videos)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
