#!python3
# -*- coding: utf-8 -*-
"""
Visualizes a video or multiple with control for the image displayed.

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
import warnings

import cv2


def load_video(filename):
    return cv2.VideoCapture(filename)


class VideosDisplayer():
    """Multiple videos synched"""
    def __init__(self, fig, videos, axes, titles=None):
        self.fig = fig
        if titles is None:
            titles = [None] * len(videos)
        self.video_displayers = []
        for ivid, (video, ax, title) in enumerate(zip(videos, axes, titles)):
            self.video_displayers.append(VideoDisplayer(video, ax, title=title))

        self.i_frame = 1

        self.checks()

    def display_frame(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        self.i_frame = i_frame

        for vd in self.video_displayers:
            vd.display_frame(i_frame=i_frame)

        if len(self.video_displayers):
            self.time = self.video_displayers[0].time

        self.fig.canvas.draw()

    def display_time(self, time):
        for vd in self.video_displayers:
            vd.display_time(time)

        if len(self.video_displayers):
            self.i_frame = self.video_displayers[0].i_frame

    def checks(self):
        if len(self.video_displayers):
            warnings.warn('No videos provided.')
            return
        fpss = [vd.fps for vd in self.video_displayers]
        if len(set(fpss)) > 1:
            warnings.warn(f'Some videos have different FPS: {fpss}.'
                          ' Use display_time to get synchronized videos.')
        num_framess = [vd.num_frames for vd in self.video_displayers]
        if len(set(num_framess)) > 1:
            warnings.warn(f'Some videos have different lengths: {num_framess}.'
                          ' One or two should not be an issue.')


class VideoDisplayer:
    def __init__(self, video, ax, fig=None, title=None):
        '''fig is needed to refresh the canvas'''
        self.video = video
        self.ax = ax
        self.fig = fig
        self.title = title

        self.i_frame = 1
        self.frame = None

        self._extract_video_constants()
        self._setup_axes()

    def _extract_video_constants(self):
        self.fps = int(self.video.get(cv2.CAP_PROP_FPS))
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))

    def _setup_axes(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        # Option: crop?
        # self.ax.set_ylim(top=400, bottom=1080)
        # option: hide spines?
        # self.ax.spines[['right', 'top']].set_visible(False)

    def display_frame(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        self.i_frame = i_frame
        self.time = i_frame / self.fps

        if self.frame is not None:
            self.frame.remove()

        self.video.set(cv2.CAP_PROP_POS_FRAMES, i_frame)
        fe, frame = self.video.read()
        if fe is False:
            warnings.warn('Could not read the frame #{}.'.format(i_frame))
            return
        frame_rgb = frame[..., ::-1].copy()
        self.frame = self.ax.imshow(frame_rgb)

        if self.title is not None:
            bbox = dict(boxstyle="round", fc="0.8")
            self.ax.annotate(
                self.title, (0.5, 0.9),
                xycoords='axes fraction', bbox=bbox,
                ha='center', va='center_baseline')

        if self.fig is not None:
            self.fig.canvas.draw()

    def display_time(self, time=None):
        if time is None:
            time = self.time
        self.i_frame = int(time * self.fps)
        self.display_frame()
