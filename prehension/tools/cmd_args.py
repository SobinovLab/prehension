#!python3
# -*- coding: utf-8 -*-
"""
Functions for parsing command line arguments.

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


def add_default_arguments(parser, arguments):
    '''Possible arguments:
    sessions
    session -- singular
    trials
    trial -- singular required argument
    temp
    processes
    overwrite
    dry_run
    '''
    if isinstance(arguments, str):
        arguments = (arguments,)
    if 'sessions' in arguments:
        parser.add_argument(
            '--sessions',
            type=str,
            default=[],
            nargs='*',
            metavar='SESSION',
            help='List of directories for processing. If empty, find all unprocessed directories. '
            'Empty by default.',
        )

    if 'trials' in arguments:
        parser.add_argument(
            '--trials',
            type=int,
            default=[],
            nargs='*',
            metavar='TRIAL_NUMBER',
            help='List of trials for processing. If empty, find all unprocessed trials. '
            'Empty by default.',
        )

    temp_folder = os.path.join('C:\\', 'tmp')
    if 'temp' in arguments:
        parser.add_argument(
            '--temp',
            type=str,
            default=temp_folder,
            help='Folder for local temporary storage. Default: {}'.format(temp_folder),
        )

    processes = int(round(os.cpu_count() * 1.4))
    if 'processes' in arguments:
        parser.add_argument(
            '--processes',
            type=int,
            default=processes,
            help='Number of parallel processes in the pool. Defaults to {}.'.format(processes),
        )

    # affecting data
    if 'overwrite' in arguments:
        parser.add_argument(
            '--overwrite', action='store_true', help='Overwrites the created files if they exist.'
        )

    if 'dry_run' in arguments:
        parser.add_argument(
            '--dry_run',
            '--check',
            action='store_true',
            help='Check which trials have not been converted. Does not create data.',
        )

    if 'session' in arguments:
        parser.add_argument(
            'session', type=str, help='Session directory to use. Required argument.'
        )

    if 'trial' in arguments:
        parser.add_argument('trial', type=int, help='Trial to do adjustment on. Required argument.')

    if 'make_plots' in arguments:
        parser.add_argument(
            '--make_plots',
            action='store_true',
            help='Makes some inspection figures. Run with --processes 1.',
        )

    if 'debug' in arguments:
        parser.add_argument('--debug', action='store_true', help='Run script in debug mode')


def add_default_kwargument(parser, k, v):
    if k == 'server':
        parser.add_argument(
            '--server',
            type=str,
            default=v,
            help='Folder where the sessions are located. Default: {}'.format(v),
        )
        return

    if k == 'processed_server':
        parser.add_argument(
            '--processed_server',
            type=str,
            default=v,
            help='Folder where the processed data from sessions are located. Default: {}'.format(v),
        )
        return

    raise ValueError('Unknown keyword argument for parser: {}'.format(k))


def add_default_kwarguments(parser, kwarguments):
    '''Possible kwargs:
    server
    '''
    for k, v in kwarguments.items():
        add_default_kwargument(parser, k, v)
