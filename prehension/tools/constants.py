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
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'osim_models',
    'default_model',
    'RightArmAndHand_NoMuscles.osim')
# CALIBRATIONS_DIR = os.path.join(
#     r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'DeepLabCut', 'CameraConfigs')
CALIBRATIONS_DIR = os.path.join(
    r'\\192.170.210.120', 'RawData', 'ProjectFolders', 'Prehension', 'CameraConfigs')

THORAX_DOF_NAMES = ('Thorax_tra1', 'Thorax_tra2', 'Thorax_tra3',
                    'Thorax_rot1', 'Thorax_rot2', 'Thorax_rot3')

DEPENDENT_DOFS = (
    'ra_sternoclavicular_r2_d', 'ra_sternoclavicular_r3_d', 'ra_unrotscap_r3_d',
    'ra_unrotscap_r2_d', 'ra_acromioclavicular_r2_d', 'ra_acromioclavicular_r3_d',
    'ra_acromioclavicular_r1_d', 'ra_unrothum_r1_d', 'ra_unrothum_r3_d', 'ra_unrothum_r2_d',
    'ra_shoulder1_r2_d', 'ra_proximal_distal_r1_d', 'ra_proximal_distal_r3_d')
ALL_DOFS = (
    'ra_sh_elv_angle', 'ra_sh_elv', 'ra_sh_rot',
    'ra_el_e_f',
    'ra_wr_sup_pro', 'ra_wr_rd_ud', 'ra_wr_e_f',
    'ra_cmc1_f_e', 'ra_cmc1_opp', 'ra_cmc1_ad_ab', 'ra_mcp1_e_f', 'ra_ip1_e_f',
    'ra_mcp2_e_f', 'ra_mcp2_ad_ab', 'ra_pip2_e_f', 'ra_dip2_e_f',
    'ra_mcp3_e_f', 'ra_mcp3_rd_ud', 'ra_pip3_e_f', 'ra_dip3_e_f',
    'ra_mcp4_e_f', 'ra_mcp4_ad_ab', 'ra_pip4_e_f', 'ra_dip4_e_f',
    'ra_mcp5_e_f', 'ra_mcp5_ad_ab', 'ra_pip5_e_f', 'ra_dip5_e_f')
ALL_DOFS_PROX_TO_DISTAL = (
    'ra_sh_elv_angle', 'ra_sh_elv', 'ra_sh_rot',
    'ra_el_e_f',
    'ra_wr_sup_pro', 'ra_wr_rd_ud', 'ra_wr_e_f',
    'ra_cmc1_f_e', 'ra_cmc1_opp', 'ra_cmc1_ad_ab',
    'ra_mcp1_e_f', 'ra_mcp2_e_f', 'ra_mcp3_e_f', 'ra_mcp4_e_f', 'ra_mcp5_e_f',
    'ra_mcp2_ad_ab', 'ra_mcp3_rd_ud', 'ra_mcp4_ad_ab', 'ra_mcp5_ad_ab',
    'ra_ip1_e_f', 'ra_pip2_e_f', 'ra_pip3_e_f', 'ra_pip4_e_f', 'ra_pip5_e_f',
    'ra_dip2_e_f', 'ra_dip3_e_f', 'ra_dip4_e_f', 'ra_dip5_e_f')
PROXIMAL_DOFS = (
    'ra_sh_elv_angle', 'ra_sh_elv', 'ra_sh_rot',
    'ra_el_e_f')
WRIST_DOFS = (
    'ra_wr_sup_pro', 'ra_wr_rd_ud', 'ra_wr_e_f')
DISTAL_DOFS = (
    'ra_wr_sup_pro', 'ra_wr_rd_ud', 'ra_wr_e_f',
    'ra_cmc1_f_e', 'ra_cmc1_opp', 'ra_cmc1_ad_ab', 'ra_mcp1_e_f', 'ra_ip1_e_f',
    'ra_mcp2_e_f', 'ra_mcp2_ad_ab', 'ra_pip2_e_f', 'ra_dip2_e_f',
    'ra_mcp3_e_f', 'ra_mcp3_rd_ud', 'ra_pip3_e_f', 'ra_dip3_e_f',
    'ra_mcp4_e_f', 'ra_mcp4_ad_ab', 'ra_pip4_e_f', 'ra_dip4_e_f',
    'ra_mcp5_e_f', 'ra_mcp5_ad_ab', 'ra_pip5_e_f', 'ra_dip5_e_f')
PER_DIGIT_DOFS = {
    'thumb': ('ra_cmc1_f_e', 'ra_cmc1_opp', 'ra_cmc1_ad_ab', 'ra_mcp1_e_f', 'ra_ip1_e_f'),
    'index': ('ra_mcp2_e_f', 'ra_mcp2_ad_ab', 'ra_pip2_e_f', 'ra_dip2_e_f'),
    'middle': ('ra_mcp3_e_f', 'ra_mcp3_rd_ud', 'ra_pip3_e_f', 'ra_dip3_e_f'),
    'ring': ('ra_mcp4_e_f', 'ra_mcp4_ad_ab', 'ra_pip4_e_f', 'ra_dip4_e_f'),
    'pinky': ('ra_mcp5_e_f', 'ra_mcp5_ad_ab', 'ra_pip5_e_f', 'ra_dip5_e_f')
}
