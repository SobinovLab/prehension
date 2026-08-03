#!python3
# -*- coding: utf-8 -*-
"""
Functions for transforming videos into 2D marker positions and then 3D using NCams with a DLC
network.
Tested to run in py 3.7 (calibration and triangulation) and 3.8 (dlc) environments.
Since NCams is an optional dependency, it is not automatically loaded. Needs NCams to be installed
from https://github.com/CMGreenspon/NCams and deeplabcut.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/SobinovLab/prehension

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import os

import deeplabcut
import tqdm

from .. import meta_session
from ..tools import logs
from ..tools.logs import rs, ws


THORAX_BOUND_MARKERS = ('M_SternumTop', 'M_SternumBot')
PROXIMAL_MARKERS = ('M_SternumTop', 'M_SternumBot', 'M_RScapulaAnt', 'M_RScapulaPost')
# all ps points should be within this radius of the centroid
# calculated from ps side = 9 cm, max width = 5 cm, rounded up
PS_CENTROID_RADIUS = 80


def analyze_videos(server, sessions, trials_sel, temp, overwrite,
                   dlc_config_path, analyze, make_videos, preset):
    """Uses pretrained machine vision network to label videos.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {int} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} -- Overwrites the created files if they exist.
        dlc_config_path {str} --- Location of the DLC config to use. Be sure to use the correct
            monkey config!
        analyze {bool} --- Do not analyze videos using a DLC network.
        make_videos {bool} --- Make videos with the labelled markers.
    """
    logs.setup_logging(temp, sessions_dir=preset['processed_server'])

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
        processed_session = os.path.join(preset['processed_server'], session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(
                server_session, processed_session)
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


def parallel_analyze_videos(dlc_config_path, video, markers_2D_dirname):
    deeplabcut.analyze_videos(
        dlc_config_path, video,
        gputouse=0, save_as_csv=True, destfolder=markers_2D_dirname)
