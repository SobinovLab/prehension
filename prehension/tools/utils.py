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

def _filter_pairs(raw_ss_list, proc_ss_list):
    pairs = []
    for pair in list(zip(raw_ss_list, proc_ss_list)):
        raw_exists = os.path.exists(pair[0])
        proc_exists = os.path.exists(pair[1])
        warn_msg = ''
        if not raw_exists:
            warn_msg += 'Raw server session {} does not exist.\n'.format(pair[0])
        if not proc_exists:
            warn_msg += 'Processed server session {} does not exist.\n'.format(pair[1])
        if warn_msg:
            ws(warn_msg)
        if raw_exists and proc_exists:
            pairs.append(pair)
    return pairs


def fetch_server_session_dirs(preset, sessions=[], filter=False):

    # Decide if we are finding sessions or using the provided sessions
    if not sessions:
        exp_session_names = meta_session.find_session_dirs(preset['default_server'])
        train_session_names = meta_session.find_session_dirs(preset['default_training_server'])
    else:
        exp_session_names = sessions
        train_session_names = sessions

    # Build initial list of raw and proc server session dirs
    raw_server_sessions = [os.path.normpath(os.path.join(preset['default_server'], os.path.basename(ss)))
                            for ss in exp_session_names]
    proc_server_sessions = [os.path.normpath(os.path.join(preset['processed_server'], os.path.basename(ss)))
                                for ss in exp_session_names]

    # now raw and proc ss lists should be the same length, read through and filter out/warn
    # on any that do not exist
    assert len(raw_server_sessions) == len(proc_server_sessions)
    if filter:
        experimental_raw_proc_pairs = _filter_pairs(raw_server_sessions, proc_server_sessions)
    else:
        experimental_raw_proc_pairs = zip(raw_server_sessions, proc_server_sessions)

    # Build initial list of raw and proc training session dirs
    raw_training_sessions = []
    proc_training_sessions = []
    if does_trianing_servers_exist(preset):
        raw_training_sessions += [os.path.normpath(os.path.join(
            preset['default_training_server'], os.path.basename(ss))) for ss in train_session_names]

        proc_training_sessions += [os.path.normpath(os.path.join(
            preset['processed_training_server'], os.path.basename(ss))) for ss in train_session_names]

    if filter:
        training_raw_proc_pairs = _filter_pairs(raw_training_sessions, proc_training_sessions)
    else:
        training_raw_proc_pairs = zip(raw_training_sessions, proc_training_sessions)

    return experimental_raw_proc_pairs, training_raw_proc_pairs



def does_trianing_servers_exist(preset):
    """Returns True if all training servers exist, prints warnings otherwise and returns False.
    """
    keys = ['default_training_server', 'processed_training_server']
    assert all(k in preset for k in keys), 'Missing keys in preset: {}'.format(keys)

    training_servers_not_none = all(v is not None for v in (preset[k] for k in keys))

    training_servers_exist = False
    if training_servers_not_none:
        training_servers_exist = all(os.path.exists(v) for v in (preset[k] for k in keys))

    if not training_servers_not_none:
        ws('one or more training servers are None in preset: {}'.format(preset['names'][0]))

    if not training_servers_exist:
        ws('one or more training servers do not exist in preset: {}'.format(preset['names'][0]))

    return training_servers_not_none and training_servers_exist


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



