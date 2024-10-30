#!python3
# -*- coding: utf-8 -*-
"""
Constants and definitions.

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
import re

# matching geom or body
DIGITS = {
    'thumb': {'c': 'pink', 'exp': lambda v: re.search('[RL]A[0-9][MPD]1.*', v)},
    'index': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A[0-9][MPD]2.*', v)},
    'middle': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A[0-9][MPD]3.*', v)},
    'ring': {'c': 'lime', 'exp': lambda v: re.search('[RL]A[0-9][MPD]4.*', v)},
    'pinky': {'c': 'brown', 'exp': lambda v: re.search('[RL]A[0-9][MPD]5.*', v)},
    'None': {'c': 'blue grey', 'exp': lambda v: True}
}
UNCLAIMED_NAME = 'None'
UNCLAIMED_INDEX = list(DIGITS.keys()).index(UNCLAIMED_NAME)

SEGMENTS = {
    'thumb_mc': {'c': 'pink', 'exp': lambda v: re.search('[RL]A4M1.*', v)},
    'thumb_pp': {'c': 'pink', 'exp': lambda v: re.search('[RL]A5P1.*', v)},
    'thumb_dp': {'c': 'pink', 'exp': lambda v: re.search('[RL]A6D1.*', v)},

    'index_mc': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A4M2.*', v)},
    'index_pp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A5P2.*', v)},
    'index_mp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A6M2.*', v)},
    'index_dp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A7D2.*', v)},

    'middle_mc': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A4M3.*', v)},
    'middle_pp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A5P3.*', v)},
    'middle_mp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A6M3.*', v)},
    'middle_dp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A7D3.*', v)},

    'ring_mc': {'c': 'lime', 'exp': lambda v: re.search('[RL]A4M4.*', v)},
    'ring_pp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A5P4.*', v)},
    'ring_mp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A6M4.*', v)},
    'ring_dp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A7D4.*', v)},

    'pinky_mc': {'c': 'brown', 'exp': lambda v: re.search('[RL]A4M5.*', v)},
    'pinky_pp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A5P5.*', v)},
    'pinky_mp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A6M5.*', v)},
    'pinky_dp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A7D5.*', v)}
}

LPS_NAME = 'medial_sensor'
RPS_NAME = 'lateral_sensor'

# only needed for DOF extraction here
# and to remind user which model to start from
ORIGINAL_OPENSIM_MODEL = os.path.join(
    os.path.dirname(
        __file__), '..', '..', 'osim_models', 'default_model',
    'RightArmAndHand_NoMuscles.osim')
# CALIBRATIONS_DIR = os.path.join(
#     r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'DeepLabCut', 'CameraConfigs')
CALIBRATIONS_DIR = os.path.join(
    r'\\192.170.210.120', 'RawData', 'ProjectFolders', 'Prehension', 'CameraConfigs')

THORAX_DOF_NAMES = ('Thorax_tra1', 'Thorax_tra2', 'Thorax_tra3',
                    'Thorax_rot1', 'Thorax_rot2', 'Thorax_rot3')
