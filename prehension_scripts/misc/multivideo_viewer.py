#!python3
# -*- coding: utf-8 -*-
"""
Example use for visualization of videos from multiple cameras.

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
import glob

import matplotlib.pyplot as plt

from prehension.tools.video import VideoViewInterface, VideoSlider


def main():
    d = r'jarvis_videos\trial1'
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
