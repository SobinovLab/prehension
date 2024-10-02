#!python
# -*- coding: utf-8 -*-
"""
Utilites for simplifying prehension processing.

Copyright (C) 2019-2024 Caleb Raman
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
import tqdm

from .. import meta_session
from . import logs
from .logs import rs, ws

import traceback
import sys


def apply_to_sessions_helper(rserv, pserv, preset, temp, func, args=None, sessions=[]):

    '''Apply a function to raw and processed session folders.
       Assumes that the function takes the following call signature:
       func(r_server_session, p_server_session, preset, session, *args)
    '''
    logs.setup_logging(temp, sessions_dir=pserv)

    if not os.path.exists(rserv):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            rserv))

    if len(sessions) == 0:
        found_sessions = meta_session.find_session_dirs(rserv)
    else:
        found_sessions = [s for s in sessions if os.path.exists(os.path.join(rserv, s))]

    found_sessions.sort()
    rs('Found {} sessions: {}'.format(len(found_sessions), ', '.join(found_sessions)))

    failed_sessions = []
    failed_sessions_errors = []
    for session in tqdm.tqdm(found_sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))

        r_server_session = os.path.normpath(os.path.join(rserv, session))
        p_server_session = os.path.normpath(os.path.join(pserv, session))

        if not os.path.exists(r_server_session):
            ws('Session {} does not exist on the server.'.format(r_server_session))
            continue

        if not os.path.exists(p_server_session):
            rs('Creating processed server session directory {}'.format(p_server_session))
            os.makedirs(p_server_session)

        try:
            func(
                r_server_session,
                p_server_session,
                preset,
                session,
                *(args if args is not None else ())
            )

        except Exception:
            print()
            ws('Function {} failed.'.format(session))
            _, exc_value, exc_traceback = sys.exc_info()
            error_str = ''.join(traceback.format_exception(None, exc_value, exc_traceback))
            ws(error_str)
            failed_sessions.append(session)
            failed_sessions_errors.append(error_str)

    if len(failed_sessions) > 0:
        print()
        ws('Failed running (func) for sessions:')
        for fs, fse in zip(failed_sessions, failed_sessions_errors):
            ws('\t{}: {}'.format(fs, fse))



