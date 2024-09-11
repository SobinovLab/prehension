#!python3.11
import os
import warnings
import itertools
import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2


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



def main():
    d = r'R:\ProjectFolders\Prehension\ProcessedData\Tot_Miller\sessions\2024_01_19\jarvis_videos\trial1'
    exts = ('.mp4', '.avi')
    filenames = []
    for ext in exts:
        filenames += glob.glob(os.path.join(d, '*' + ext))

    fig = plt.figure(figsize=(16, 9))
    fig.canvas.manager.set_window_title(d)
    vvi = VideoViewInterface(fig, [0.05, 0.2, 0.05, 0.05], filenames)
    vvi.setup()

    vs = VideoSlider(fig, [0.05, 0.05, 0.9, 0.15], vvi.num_frames, vvi.i_frame, vvi.display)

    plt.show()


if __name__ == '__main__':
    main()
