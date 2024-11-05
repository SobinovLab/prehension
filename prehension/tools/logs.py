#!python3
# -*- coding: utf-8 -*-
"""
Setting up and utilizing logging.

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
import sys
import uuid
import shutil
import atexit
import logging
import warnings
import datetime


def logging_atexit_function(logging_filename, destination_path, start_time):
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime('%Y.%m.%d-%H:%M:%S')
    time_passed_str = (end_time - start_time)
    logging.info(f'\n\tTimestamp: {end_time_str}\n'
                 f'\tTime passed: {time_passed_str}')
    shutil.copy(logging_filename, destination_path)


def setup_logging(temp, sessions_dir=None):
    '''
    Log filename, all arguments, and time
    '''
    os.makedirs(temp, exist_ok=True)
    # Move creation of random hash to this function
    random_hash = str(uuid.uuid4().hex)
    # Exectuting file name (for example: create_meta.py)
    arg_stump = ' '.join(sys.argv)
    exec_fname = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    # exec_fname = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    timestamp = datetime.date.today().strftime('%Y.%m.%d')
    start_time = datetime.datetime.now()
    timestamp_long = start_time.strftime('%Y.%m.%d-%H:%M:%S')
    logging_filename = os.path.join(temp, f'{timestamp}_{exec_fname}_{random_hash}.log')
    logging.basicConfig(filename=logging_filename, level=logging.INFO, force=True)
    logging.info('')

    # Define the cleanup action as upload to sessions_log dir
    if sessions_dir is not None:
        destination_path = os.path.join(sessions_dir, 'logs')
        os.makedirs(destination_path, exist_ok=True)
        assert os.path.exists(logging_filename)
        assert os.path.exists(destination_path)

        def laf():
            logging_atexit_function(logging_filename, destination_path, start_time)

        atexit.register(laf)

    logging.info(f'\n\tLog path: {logging_filename}\n'
                 f'\tScript call: {arg_stump}\n'
                 f'\tTimestamp: {timestamp_long}')
    print(f'{logging_filename} created.')


def rs(s):
    print(s)
    logging.info(s)


def ws(s):
    warnings.warn(s, stacklevel=2)
    logging.warning(s)
