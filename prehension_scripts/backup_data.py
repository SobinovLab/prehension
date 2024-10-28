#!python
# -*- coding: utf-8 -*-
"""
Uploading data to a shared directory.

TODO: move back to SIK

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
import datetime
import inspect
import os
import re
import sys
import time

import tqdm

from prehension.tools import filesystem

# Not sure we need this, leave for now though ...
currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def print_ignore(src, names):
    """Simulates ignore, but only prints out arguments."""
    print(src)
    for name in names:
        print('\t', name)
    return []


def ignore_nothing(src, names):
    return []


def ignore_camera_images_dirs(src, names):
    basedir = os.path.split(src)[1]
    if basedir in 'cameras':
        # camera folders
        return [name for name in names if re.match('cam[0-9]{8}$', name) is not None]
    else:
        return []


def transfer_one(transfer_name, src, dst, dry_run, verbose_level,
                 add_ignore_list=None, ignore_cameras=True, overwrite_with_new=True):
    start_time = time.time()
    if add_ignore_list is None:
        if ignore_cameras:
            ignore = ignore_camera_images_dirs
        else:
            ignore = ignore_nothing
    else:
        if ignore_cameras:
            def ignore(src, names):
                return ignore_camera_images_dirs(src, names) + add_ignore_list
        else:
            def ignore(src, names):
                return add_ignore_list

    pcas = filesystem.PrintCopyAccumulateSize(dry_run, verbose_level - 1)
    make_dir = not dry_run

    if overwrite_with_new:
        overwrite_existing = filesystem.overwrite_existing_new
    else:
        overwrite_existing = filesystem.overwrite_existing_never

    filesystem.copytree(
        src, dst,
        symlinks=True,
        ignore=ignore, copy_function=pcas,
        make_dir=make_dir, dir_exist_ok=True,
        overwrite_existing=overwrite_existing,
        verbose=bool(verbose_level))

    timedelta = str(datetime.timedelta(seconds=time.time() - start_time))
    if dry_run:
        print('Found {} {} files for copying. Search took {}.'.format(
            pcas, transfer_name, timedelta))
    else:
        print('Copied {} {} files over {}.'.format(
            pcas, transfer_name, timedelta))


def main():

    src_base = os.path.join(r'\\midwaysmb.rcc.uchicago.edu', 'project2', 'nicho', 'pitt_collab',
                            'BCI02', 'BlackrockData')
    dst_base = os.path.join(r'T:\\', 'SessionData', 'BCI02', 'BlackrockData')

    dry_run = True
    verbose_level = 1
    overwrite_with_new = False

    transfers = {
        # 'test_bci': {
        #     'src': os.path.join('Baseline'),
        #     'dst': os.path.join('BaselineRecordings')
        # },
        # 'OLS': {
        #     'src':
        #     'dst':
        # }
    }
    for transfer_name, transfer in transfers.items():
        src = os.path.join(src_base, transfer['src'])
        dst = os.path.join(dst_base, transfer['dst'])

        transfer_one(transfer_name, src, dst, dry_run, verbose_level,
                     add_ignore_list=transfer.get('ignore'), overwrite_with_new=overwrite_with_new)


def subject_filname_pitt(subject, location, session):
    return os.path.join(
        r'\\share.files.pitt.edu\RNELShare\data_raw\human\crs_array',
        f'{subject}',
        'OpenLoopStim',
        f'{subject}{location}.data.{session:05d}')


def subject_filname_chicago(subject, location, session):
    return os.path.join(
        r'\\192.170.210.82\Data\SessionData',
        f'{subject}',
        'OpenLoopStim',
        f'{subject}{location}.data.{session:05d}')


def main_bci():
    start_time = time.time()
    dry_run = False
    verbose_level = 0
    overwrite_with_new = False

    crs02b_sessions = [
        882, 887, 892, 901, 902, 910, 911, 916, 920, 923, 926, 947, 959, 961, 965, 966]
    crs02b_locations = ['Lab']*len(crs02b_sessions)
    crs07_sessions = [68, 69, 73, 74, 74, 86, 87, 89, 92]
    crs07_locations = ['Lab', 'Lab', 'Home', 'Home',
                       'Lab', 'Home', 'Home', 'Home', 'Home']

    pcas = filesystem.PrintCopyAccumulateSize(dry_run, verbose_level - 1)
    make_dir = not dry_run
    if overwrite_with_new:
        overwrite_existing = filesystem.overwrite_existing_new
    else:
        overwrite_existing = filesystem.overwrite_existing_never

    subjects = ['CRS02b'] * len(crs02b_sessions) + \
        ['CRS07'] * len(crs07_sessions)
    sessions = crs02b_sessions + crs07_sessions
    locations = crs02b_locations + crs07_locations

    for subject, session, location in zip(tqdm.tqdm(subjects), sessions, locations):
        src = subject_filname_pitt(subject, location, session)
        dst = subject_filname_chicago(subject, location, session)
        filesystem.copytree(
            src, dst,
            symlinks=True,
            ignore=ignore_nothing, copy_function=pcas,
            make_dir=make_dir, dir_exist_ok=True,
            overwrite_existing=overwrite_existing,
            verbose=bool(verbose_level))

    timedelta = str(datetime.timedelta(seconds=time.time() - start_time))
    if dry_run:
        print('Found {} files for copying. Search took {}.'.format(
            pcas, timedelta))
    else:
        print('Copied {} files over {}.'.format(pcas, timedelta))


if __name__ == '__main__':
    main_bci()
