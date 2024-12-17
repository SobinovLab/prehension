#!python3
# -*- coding: utf-8 -*-
"""
Makes videos from images using ffmpegio.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/BensmaiaLab/prehension

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
import copy
import shutil
import itertools
import warnings

import numpy as np
import matplotlib as mpl
import cv2
import tqdm
from reporting_pool import ReportingPool

from .. import meta_session
from ..tools import logs
from ..tools import filesystem
from ..tools.logs import rs, ws


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
            hide_banner=None,  # loglevel='error',
            # show_log=True,  # printing to stdout
            an=None, an_in=None,  # no sound for output or input
            **{
                'r': rate,
                'codec:v': 'h264_nvenc',
                'pix_fmt': 'rgb0',
                'preset:v': 'p7',
                'tune:v': 'hq',
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
        frame_filenames = filesystem.get_image_list(path=trial.images_dirnames[camera_serial])
        if len(frame_filenames) == 0:
            ws(f'Folder {trial.images_dirnames[camera_serial]} has no images.')
            continue
        # filename_ou = os.path.split(trial.videos[camera_serial])[1]

        # ncams.image_tools.images_to_video(
        #     frame_filenames, filename_ou,
        #     fps=mstruct['fps'], output_folder=dirname_ou, logger=None)
        new_make_video(frame_filenames, trial.videos[camera_serial], mstruct['fps'])

        # copy log
        shutil.copy2(trial.images_logs[camera_serial], trial.videos_logs[camera_serial])

        if False:  # clean:
            shutil.rmtree(trial.images_dirnames[camera_serial])


def compress_session_cameras(server, sessions, trials_sel, temp, processes, overwrite, clean):
    """Transforms images from a session into video.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed
            trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        clean {bool} --- DANGER! Remove directories from the server that were converted into videos.
    """
    logs.setup_logging(temp, sessions_dir=server)

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


def crop_video(ivp, ovp, frame_start, frame_end):
    # open reading file and get params
    cap = cv2.VideoCapture(ivp)
    img_size = [
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    # where start
    cap.set(1, frame_start)

    # open video for writing
    out = cv2.VideoWriter(
        ovp, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
        frame_rate, (img_size[0], img_size[1]))

    number_frames = frame_end - frame_start + 1

    for frame_num in tqdm.tqdm(range(number_frames)):
        ret, img = cap.read()
        img = img.astype(np.uint8)
        out.write(img)
    out.release()
    cap.release()


class VideoViewInterface:
    def __init__(self, fig, videoarea, filenames, run_setup=False):
        self.fig = fig
        self.videoarea = videoarea  # <left>, <bottom>, <right>, <top>
        self.filenames = filenames  # if a file does not exist, it is skipped
        self.start_frame = None
        self.x_padding = 0.001
        self.y_padding = 0.001
        self.n_rows = None
        self.n_cols = None
        self.i_frame = None  # 0 to num_frames-1 including

        # technical variables
        self._videosframe = None

        if run_setup:
            self.setup()

    def setup(self):
        self.load_videos()
        self.setup_axes()
        self.display()

    def load_videos(self):
        self.videos = []
        self.shortnames = []
        self.num_frames = np.nan
        nonexistant_files = []
        for ifile, filename in enumerate(self.filenames):
            if not os.path.isfile(filename):
                nonexistant_files.append(ifile)
                warnings.warn(f'File {filename} does not exist, skipping.')
                continue

            self.videos.append(cv2.VideoCapture(filename))
            video_nframes = int(self.videos[-1].get(cv2.CAP_PROP_FRAME_COUNT))
            if np.isnan(self.num_frames):
                self.num_frames = video_nframes
            elif self.num_frames != video_nframes:
                warnings.warn('Number of frames in videos does not match, truncating to shortest.')
                self.num_frames = min(self.num_frames, video_nframes)
            self.shortnames.append(os.path.basename(filename))

        for ifile in reversed(nonexistant_files):
            del self.filenames[ifile]

        if len(self.videos) == 0:
            raise ValueError("No videos found.")

    def calculate_tiling(self):
        '''calculate n_rows and n_cols from number of videos'''
        n_tot = len(self.videos)

        if self.n_cols is None:
            if self.n_rows is None:  # if both are unset
                self.n_cols = int(np.ceil(np.sqrt(n_tot)))
            else:
                self.n_cols = int(np.ceil(n_tot / self.n_rows))
        if self.n_rows is None:
            self.n_rows = int(np.ceil(n_tot / self.n_cols))

    def calculate_axes_borders(self):
        # self.videoarea: <left>, <bottom>, <right>, <top>
        x_size = (1 - self.videoarea[0] - self.videoarea[2]) / self.n_cols
        x_size_int = x_size - 2 * self.x_padding
        y_size = (1 - self.videoarea[1] - self.videoarea[3]) / self.n_rows
        y_size_int = y_size - 2 * self.y_padding

        axes_pos = []
        for i_row, i_col in itertools.product(range(self.n_rows), range(self.n_cols)):
            axes_pos.append([
                self.videoarea[0] + x_size * i_col + self.x_padding,
                self.videoarea[1] + y_size * (self.n_rows - i_row - 1) + self.y_padding,
                x_size_int, y_size_int])

        return axes_pos

    def setup_axes(self):
        self.calculate_tiling()
        axes_pos = self.calculate_axes_borders()

        self.video_axes = []
        for shortname, axis_pos in zip(self.shortnames, axes_pos):
            ax = self.fig.add_axes(axis_pos)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(0.5, 0.92, shortname,
                    color='k', ha='center', va='center', transform=ax.transAxes,
                    zorder=np.inf,
                    bbox={'boxstyle': 'round', 'fc': (1, 1, 1, 0.75), 'ec': (1, 1, 1, 0.75)})

            self.video_axes.append(ax)

    @staticmethod
    def display_videoframe(video, ax, cameraframe):
        video.set(cv2.CAP_PROP_POS_FRAMES, cameraframe)
        fe, frame = video.read()
        if fe is False:
            warnings.warn(f'Could not read the frame #{cameraframe}.')
            return
        frame_rgb = frame[..., ::-1].copy()
        return ax.imshow(frame_rgb)

    def display(self, i_frame=None):
        if i_frame is not None:
            self.i_frame = int(i_frame)
        if self.i_frame is None:
            self.i_frame = int(0.05 * self.num_frames)
        self.i_frame = max(0, self.i_frame)
        self.i_frame = min(self.i_frame, self.num_frames - 1)

        # remove previous
        if self._videosframe is not None:
            for vf in self._videosframe:
                vf.remove()

        # draw side
        self._videosframe = []
        for video, ax in zip(self.videos, self.video_axes):
            self._videosframe.append(self.display_videoframe(
                video, ax, self.i_frame))

        self.fig.canvas.draw_idle()


class VideoSlider():
    """docstring for VideoSlider"""
    def __init__(self, fig, sliderarea, num_frames, initial, onchange):
        self.fig = fig
        self.ax = self.fig.add_axes(sliderarea)
        self.slider = mpl.widgets.Slider(
            ax=self.ax,
            label='Frame #',
            valmin=0,
            valmax=num_frames-1,
            valinit=initial,
            valfmt='%d',
            dragging=False,
            valstep=1
        )

        self.slider.on_changed(onchange)

