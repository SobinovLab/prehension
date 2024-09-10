#!python3
import os
import pickle
import re
from copy import deepcopy
from glob import glob

import yaml


def get_image_list(path=None, sort=True):
    '''Returns a list of all image filenames.

    Wrapper for 'utilities.get_file_list' with file_extensions defined as:
        ('jpg', 'jpeg', 'png', 'bmp')

    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list of strings} -- list of all image filenames.
    '''

    return get_file_list(('jpg', 'jpeg', 'png', 'bmp', 'Bmp'), path=path, sort=sort)


def get_file_list(file_extensions, path=None, sort=True):
    '''Returns a list of all filenames with a specific extension.

    Files with extensions .jpg, .jpeg, .png, .bmp are considered images. Searches shell-style for
        <path>/*.<file extension>

    Arguments:
        file_extensions {list of strings} -- file extensions to return, with or without the dot.
            If None or empty, returns all files with extensions.
    Keyword Arguments:
        path {string} -- directory to explore. (default: current directory)
        sort {bool} -- alphanumeric sort the output list (default: {True})
    Output:
        strings {list of strings} -- list of all filenames with specified extension.
    '''
    if isinstance(file_extensions, str):
        file_extensions = [file_extensions]

    if file_extensions is None or len(file_extensions) == 0:
        file_extensions = ('*', )
    else:
        file_extensions = [ifx.strip('.') for ifx in file_extensions]

    files = []
    for file_extension in file_extensions:
        if path is None:
            files += glob('*.' + file_extension)
        else:
            files += glob(os.path.join(path, '*.' + file_extension))

    if sort:
        files = alphanumeric_sort(files)

    return files


def alphanumeric_sort(strings):
    '''Alphanumeric sorter that considers parts of the numerical parts of the string independently.

    For example, 'text9moretext' < 'text10moretext' when using this sorting function.
    Useful for:
        sorting out very high framerate images
        sorting by framerate, because '11'<'9', but 11>9
        recording for more than 999.9999 seconds (in the current format the generic sort does not
            give the desired result).

    Arguments:
        strings {list of strings} -- list of strings (e.g. filenames to be sorted)
    Output:
        strings {list of strings} -- sorted list of strings
    '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(strings, key=alphanum_key)


def yaml_to_config(filename, overwrite_setup_path=None):
    '''Imports camera config from a YAML file.

    Arguments:
        filename {string} -- filename of the YAML ncams_config file.

    Keyword Arguments:
        overwrite_setup_path {bool or None} -- if set to True, automatically overwrites the setup
            path from the ncams_config with the actual path. If False, will not overwrite. If None,
            will ask the user for keyboard input. (default: None)

    Output:
        ncams_config {dict} -- see help(ncams.camera_tools).
    '''
    with open(filename, 'r') as yaml_file:
        ncams_config = yaml.safe_load(yaml_file)

    current_config_path = os.path.join(ncams_config['setup_path'], ncams_config['setup_filename'])

    check_filename = False
    if not os.path.exists(current_config_path):
        check_filename = True

    if not check_filename and not os.path.samefile(current_config_path, filename):
        check_filename = True

    if check_filename:
        if overwrite_setup_path is None:
            print('The setup path in the loaded ncams_config does not match its current location.')
            user_input_str = 'Would you like to overwrite the setup path? (yes/no)\n'
            user_input = input(user_input_str).lower()
            if user_input in ('yes', 'y'):
                do_overwrite = True
            else:
                do_overwrite = False
        else:
            do_overwrite = overwrite_setup_path
        if do_overwrite:
            (new_path, new_filename) = os.path.split(filename)
            ncams_config['setup_path'] = new_path
            ncams_config['setup_filename'] = new_filename
            print('The workspace variable has been overwritten but the original file has not.',
                  'Use the "config_to_yaml" function to overwrite the original file.')

    return ncams_config


def import_intrinsics(ncams_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in ncams_config. Alternatively, if a path
    is given then it will directly assume it is the path to the intrinsics config.

    Arguments:
        ncams_config {dict} -- information about camera configuration. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            setup_path {string} -- directory where the camera setup is located.
            intrinsic_path {string} -- relative path to where calibration information is stored
                from 'setup_path'.
            intrinsic_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
    Output:
        intrinsic_config {dict} -- information on camera calibration and the results of said
                calibraion. Order of each list MUST adhere to intrinsic_config['serials'] AND
                ncams_config['serials']. Should have following keys:
            serials {list of numbers} -- list of camera serials.
            distortion_coefficients {list of np.arrays} -- distortion coefficients for each camera
            camera_matrices {list of np.arrays} -- camera calibration matrices for each camera
            reprojection_errors {list of numbers} -- reprojection errors for each camera
            path {string} -- directory where calibration information is stored. Should be same as
                information in ncams_config.
            dicts {dict of 'camera_calib_dict's} -- keys are serials, values are
                'camera_calib_dict', see help(ncams.camera_tools).
            filename {string} -- name of the pickle file to store the config in/load from.
    '''
    # Get the path name
    if type(ncams_config) == dict:
        filename = os.path.join(ncams_config['setup_path'], ncams_config['intrinsic_path'],
                                ncams_config['intrinsic_filename'])
    elif type(ncams_config) == str:
        filename = ncams_config

    # Load the file
    with open(filename, 'rb') as f:
        _intrinsic_config = pickle.load(f)

    intrinsic_config = deepcopy(_intrinsic_config)
    # we want to keep whatever other info was stored just in case
    # Then add everything we usually use to a dictionary for easy lookup
    intrinsic_config['dicts'] = {}
    for serial in intrinsic_config['serials']:
        idx = _intrinsic_config['serials'].index(serial)

        intrinsic_config['dicts'][serial] = {
            'serial': serial,
            'distortion_coefficients': _intrinsic_config['distortion_coefficients'][idx],
            'camera_matrix': _intrinsic_config['camera_matrices'][idx],
            'reprojection_error': _intrinsic_config['reprojection_errors'][idx],
            'detected_markers': _intrinsic_config['detected_markers'][idx],
            'calibration_images': _intrinsic_config['calibration_images'][idx]
        }

    return intrinsic_config


def import_extrinsics(ncams_config):
    '''Imports camera calibration info for all cameras in the setup from a pickle file.

    Reorders the loaded information to adhere to 'serials' in ncams_config. Alternatively, if a path
    is given then it will directly assume it is the path to the intrinsics config.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.
            extrinsic_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        extrinsic_config {dict} -- information on estimation of relative position of all
                cameras and the results of said pose estimation. Order of each list MUST adhere to
                extrinsic_config['serials'] and ncams_config['serials']. Should
                have following keys:
            serials {list of numbers} -- list of camera serials.
            world_locations {list of np.arrays} -- world locations of each camera.
            world_orientations {list of np.arrays} -- world orientation of each camera.
            path {string} -- directory where pose estimation information is stored. Should be same
                as information in ncams_config.
            filename {string} -- name of the YAML file to store the config in/load from.
    '''
    # Get the path name
    if type(ncams_config) == dict:
        filename = os.path.join(ncams_config['setup_path'], ncams_config['extrinsic_path'],
                            ncams_config['extrinsic_filename'])
    elif type(ncams_config) == str:
        filename = ncams_config

    # Load the file
    with open(filename, 'rb') as f:
        _extrinsic_config = pickle.load(f)

    extrinsic_config = deepcopy(_extrinsic_config)
    # we want to keep whatever other info was stored just in case
    # Then add everything we usually use to a dictionary for easy lookup
    extrinsic_config['dicts'] = {}
    for serial in extrinsic_config['serials']:
        idx = _extrinsic_config['serials'].index(serial)

        extrinsic_config['dicts'][serial] = {
            'world_location': _extrinsic_config['world_locations'][idx],
            'world_orientation': _extrinsic_config['world_orientations'][idx]
        }

    return extrinsic_config


def load_calibrations(ncams_config):
    '''Safely loads pose estimation and camera calibration from files.

    Arguments:
        ncams_config {dict} -- see help(ncams.camera_tools). Should have following keys:
            serials {list of numbers} -- list of camera serials.
            setup_path {string} -- directory where the camera setup is located.
            intrinsic_path {string} -- directory where calibration information is stored.
            intrinsic_filename {string} -- name of the pickle file to store the calibration
                config in/load from.
            extrinsic_path {string} -- relative path to where pose estimation information is
                stored from 'setup_path'.
            extrinsic_filename {string} -- name of the pickle file to store the pose
                estimation config in/load from.
    Output:
        intrinsic_config {dict} -- see help(ncams.camera_tools), None if not found
        extrinsics_config {dict} -- see help(ncams.camera_tools), None if not found
    '''
    try:
        intrinsics_config = import_intrinsics(ncams_config)
        print('Camera calibration loaded.')
    except FileNotFoundError:
        intrinsics_config = None
        print('No camera calibration file found.')

    try:
        extrinsics_config = import_extrinsics(ncams_config)
        print('Pose estimation loaded.')
    except FileNotFoundError:
        extrinsics_config = None
        print('No pose estimation file found.')

    return (intrinsics_config, extrinsics_config)
