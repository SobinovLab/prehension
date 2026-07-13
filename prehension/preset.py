#!python3
# -*- coding: utf-8 -*-
"""
Loading minimal configuration of a specific dataset, like the location of the sessions. Tries to
load presets from a prehension_presets module if that is installed.

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
import sys
import copy
import importlib


PRESETS = {
    # EXAMPLE PRESET
    'example_preset_1': {
        # The preset name. This is the string you can pass to the processing script to load all
        # this configuration.
        'names': ['example_preset', 'ex_pre', 'ep1'],
        # The hand used in the experiment 'left' or 'right'
        'hand': 'right',
        # The server containing the raw data to be processed
        'default_server': os.path.join(
            'C:\\', 'Data', 'M1LeftHemisphere', 'sessions'),
        # The server to write the data to
        'processed_server': os.path.join(
            'C:\\', 'Data', 'M1LeftHemisphere', 'sessions'),
        # The path to the DeepLabCut configuration yaml file
        'dlc_config_path': os.path.join(
            'C:\\', 'Data', 'M1LeftHemisphere', 'M1-CMG-2021-06-27', 'config.yaml'),
        # which sessions were used for labeling for machine vision and who did the work
        'labeling': {
            'sessions': ('2021_04_08', '2021_04_29'),
            'labelers': ('CMG', 'AS', 'LO', 'NS'),
        },
        # which trial was used for scaling of the skeleton
        'scaling': {
            'session': '2021_04_29',
            'trial': 63
        },
        # The serial numbers for the medial and lateral sensors
        'ps_dic': {
            'medial_sensor': '00110-2743',
            'lateral_sensor': '00110-2746',
        },
        # The frames per second for camera data
        'fps': 50,
        # The columns identifying a unique object type
        'object_def_columns': ('pos_translation_z(mm)', 'pos_tilt(deg)', 'pos_aperture(mm)'),
    }
}


def attempt_loading_presets():
    '''Check if there is a presets module, and import it, loading presets from it.'''
    if importlib.util.find_spec('prehension_presets') is not None:
        import prehension_presets
        for k, v in prehension_presets.PRESETS.items():
            globals()['PRESETS'][k] = v


# Load local presets if they are available and overwrite the current files
attempt_loading_presets()


def is_preset(name):
    '''Checks if name corresponds to an existing preset.'''
    return name in sum([[k] + v['names'] for k, v in PRESETS.items()], [])


def get_preset(name):
    '''Returns the dictionary corresponding to the requested preset.'''
    for k, v in PRESETS.items():
        if name == k or name in v['names']:
            v['name'] = k
            return k, v

    raise ValueError('Preset {} not found.'.format(name))


def process_args_for_preset():
    '''Check if the first argument corresponds to a preset and loads its dictionary.'''
    argv = copy.deepcopy(sys.argv)[1:]
    if len(argv) < 1 or not is_preset(argv[0]):
        if '--help' in argv or '-h' in argv:
            pass
        else:
            raise RuntimeError('Preset was not selected.')
    else:
        current_preset_name = argv[0]
        del argv[0]

    current_preset_name, current_preset = get_preset(current_preset_name)
    print('Using {} preset settings.'.format(current_preset_name))

    return current_preset_name, current_preset, argv


# import_by_location(preset_fname, module_name=preset_name)
# def import_by_location(fname, module_name=None):
#     '''This allows importing a single file from location,
#     but screws with relative dependencies.'''
#     if module_name is None:
#         module_name = os.path.splitext(os.path.split(fname)[1])[0]
#     spec = importlib.util.spec_from_file_location(module_name, fname)
#     foo = importlib.util.module_from_spec(spec)
#     sys.modules[module_name] = foo
#     spec.loader.exec_module(foo)
