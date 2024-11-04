#!python3
# -*- coding: utf-8 -*-
"""
Example of how to crop a video using tools.video.

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

from prehension.tools import video


def main():
    in_video_paths = (
        r"cameras\cam001\trial0\cam001.avi",
        r"cameras\cam002\trial0\cam002.avi",
        r"cameras\cam003\trial0\cam003.avi",
        r"cameras\cam004\trial0\cam004.avi",
    )
    ou_video_paths = (
        r"cameras\cam001\trial1\cam001.avi",
        r"cameras\cam002\trial1\cam002.avi",
        r"cameras\cam003\trial1\cam003.avi",
        r"cameras\cam004\trial1\cam004.avi",
    )

    frame_start = 290
    frame_end = 580

    for ivp, ovp in zip(in_video_paths, ou_video_paths):
        video.crop_video(ivp, ovp, frame_start, frame_end)


if __name__ == "__main__":
    main()
