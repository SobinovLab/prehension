

'''
# INSTRUCTIONS
# Copy this file for your institution and call it preset.py
# In the preset dictionary create entries like the example provided
# In our setup we define default server as the location of the raw data
# While the processed_server is the location where the resultant data will be written during processing
# See the inline comments at each datafield for what that field should contain

All scripts in this directory will import this one to define the server on which to run the
analysis.

Note:
    'ps_dic': {
            'medial_sensor': VAL1
            'lateral_sensor': VAL2
        }
    Does not actually correspond to left and right, but to medial and lateral, respectively. Think
    of it as grabbing with the right hand - left will be medial (thumb), right - lateral (fingers).

'''

import os
import sys
import copy

# Make some stumps!
raw_stump = os.path.join(r'\\192.170.210.120', 'RawData', 'ProjectFolders',
                          'Prehension', 'Data')
dlc_stump = os.path.join(r'\\192.170.210.120', 'RawData', 'ProjectFolders',
                          'Prehension', 'DeepLabCut')
proc_stump = os.path.join(r'\\192.170.210.120', 'ProcessedData', 'ProjectFolders',
                           'Prehension', 'ProcessedData')

CURRENT_PRESET = 'mojito_left_hemisphere'
PRESETS = {

    # EXAMPLE PRESET
    'example_preset_1': {
        # The preset name. This is the string you will pass the processing script. There can be multiple values
        'names': ['example_preset', 'ex_pre', 'ep1'],
        # The hand used in the experiment 'left' or 'right'
        'hand': 'right',
        # The server containing the raw data to be processed, see https://github.com/nishbo/stereo_inverse_kinematics for details on our directory structure
        'default_server': os.path.join(raw_stump, 'MojitoLeftHemisphere', 'training_sessions_k1', 'sessions'),
        # The server to write the data to
        'processed_server': os.path.join(proc_stump, 'MojitoLeftHemisphere', 'training_sessions_k1', 'sessions'),
        # The path to the deep lab cut configuration yaml file
        'dlc_config_path': os.path.join(dlc_stump, 'MojitoJune2021', 'Mojito-CMG-2021-06-27', 'config.yaml'),
        'labeling': {
            'sessions': ('2021_04_08', '2021_04_29'),
            'labelers': ('CMG', 'AS', 'LO', 'NS'),
        },
        'scaling': {
            'session': '2021_04_29',
            'trial': 63
        },
        'period': ['2021_04_02', '2021_05_01'],
        'areas': ('3a', '2'),
        # The serial numbers for the medial and lateral sensors
        'ps_dic': {
            'medial_sensor': '00110-2743',
            'lateral_sensor': '00110-2746',
        },
        # The frames per second for camera data
        'fps': 50,
        # The columns to be created in the meta object file. You can add or remove these as you see fit
        'object_def_columns': (
            'pos_translation_z(mm)', 'pos_tilt(deg)', 'pos_aperture(mm)'),
        'straight_to_video': False,
    },
    ##### INSERT MORE PRESETS HERE ######

    # 'pimms_right_hemisphere': {
    #     'names': ['prhem', 'pimms_right_hem', 'pimms'],  # 'prh',
    #     'hand': 'left',
    #     'default_server': os.path.join(raw_stump, 'PimmsRightHemisphere', 'sessions'),
    #     'processed_server': os.path.join(proc_stump, 'PimmsRightHemisphere', 'training_sessions_k1'),
    #     'dlc_config_path': os.path.join(dlc_stump, 'Pimms_RightHem', 'Pimms-RightHem-2023-06-21', 'config.yaml'),
    #     'labeling': {
    #         'sessions': ('2022_03_01_Set1', ),
    #         'labelers': ('VA', 'EO', 'QH', 'AV', 'NS'),
    #     },
    #     'scaling': {
    #         'session': '2022_03_01_Set1',
    #         'trial': 100
    #     },
    #     'period': ['2021_11_11', '2022_03_04'],
    #     'areas': ('M1', ),
    #     'ps_dic': {
    #         'medial_sensor': '00110-2746',
    #         'lateral_sensor': '00110-2743',
    #     },
    #     'fps': 50,
    #     'object_def_columns': (
    #         'pos_translation_z(mm)', 'pos_tilt(deg)', 'pos_aperture(mm)', 'targetForce(N)'),
    #     'straight_to_video': False,
    # },
}

# === Helpers ===
def is_preset(name):
    return name in sum([[k] + v['names'] for k, v in PRESETS.items()], [])


def get_preset(name):
    for k, v in PRESETS.items():
        if name == k or name in v['names']:
            return k, v

    raise ValueError('Preset {} not found.'.format(name))


def process_args_for_preset():
    argv = copy.deepcopy(sys.argv)[1:]
    if len(argv) < 1 or not is_preset(argv[0]):
        current_preset_name = CURRENT_PRESET
        if '--help' in argv or '-h' in argv:
            pass
        else:
            answ = input('Default preset is {}. Continue with default? (y/[n]) '.format(
                current_preset_name))
            if len(answ) < 1 or answ[0].lower() != 'y':
                raise RuntimeError('Default preset was not selected.')
    else:
        current_preset_name = argv[0]
        del argv[0]

    current_preset_name, current_preset = get_preset(current_preset_name)
    print('Using {} preset settings.'.format(current_preset_name))

    return current_preset_name, current_preset, argv
