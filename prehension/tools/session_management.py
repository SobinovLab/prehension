#!python3
# -*- coding: utf-8 -*-
"""
Classes and functions for managing sessions.

Copyright (C) 2019-2024 Caleb Raman
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
import tqdm
import traceback
import sys

from .. import meta_session
from . import logs
from .logs import rs, ws


def _filter_pairs(raw_ss_list, proc_ss_list, verbose=False):
    """Filters out pairs of raw session folders and processed session folders that do not exist,
    and returns them as a list of tuples. warns user if verbose == True

    Args:
        raw_ss_list (list of paths): the list of raw session folders to process
        proc_ss_list (list of paths): the list of processed session folders to process
        verbose (bool, optional): boolean for whether to print warnings. Defaults to False.

    Returns:
        list of tuples: pairs of raw and processed session folders that exist
        (raw session path, processed session path)
    """

    pairs = []
    for pair in list(zip(raw_ss_list, proc_ss_list)):
        raw_exists = os.path.exists(pair[0])
        proc_exists = os.path.exists(pair[1])
        warn_msg = ""
        if not raw_exists:
            warn_msg += "Raw server session {} does not exist.\n".format(pair[0])
        if not proc_exists:
            warn_msg += "Processed server session {} does not exist.\n".format(pair[1])
        if warn_msg and verbose:
            ws(warn_msg)
        if raw_exists and proc_exists:
            pairs.append(pair)
    return pairs


def _fetch_session_dirs(preset_server, preset_proc_server, sessions, remove_missing_sessions=False):
    """Returns a list of raw and processed session folders that exist for a given preset, only
    the actual experiments, no training

    Args:
        preset_server:
        preset_proc_server:
        sessions (list): list of sessions we want to process.
        remove_missing_sessions (bool, optional): remove out sessions that do not exist.
            Defaults to False.

    Returns:
        list[tuple(dir, dir)]: matched pairs of raw and processed session folders
    """
    # Build initial list of raw and proc server session dirs
    raw_server_sessions = [
        os.path.normpath(os.path.join(preset_server, os.path.basename(ss)))
        for ss in sessions]
    proc_server_sessions = [
        os.path.normpath(os.path.join(preset_proc_server, os.path.basename(ss)))
        for ss in sessions]

    # read through and remove_missing_sessions out/warn on any that do not exist
    if remove_missing_sessions:
        raw_proc_pairs = _filter_pairs(raw_server_sessions, proc_server_sessions)
    else:
        raw_proc_pairs = list(zip(raw_server_sessions, proc_server_sessions))

    return raw_proc_pairs


def fetch_exp_session_dirs(preset, sessions=None, remove_missing_sessions=False):
    if sessions:
        exp_session_names = sessions
    else:
        exp_session_names = meta_session.find_session_dirs(preset["default_server"])

    experimental_raw_proc_pairs = _fetch_session_dirs(
        preset["default_server"], preset["processed_server"], exp_session_names,
        remove_missing_sessions=remove_missing_sessions)

    return experimental_raw_proc_pairs


def fetch_server_session_dirs(preset, sessions=None, remove_missing_sessions=False):
    """Returns a list of raw and processed session folders that exist for a given preset

    Args:
        preset (dict): the preset in question
        sessions (list, optional): list of sessions we want to process.
            Defaults to [] (i.e. all sessions)
        remove_missing_sessions (bool, optional): remove sessions that do not exist.
            Defaults to False.

    Returns:
        list[tuple(dir, dir)], list[tuple[dir, dir]]: matched pairs of raw and processed
        session folders, first element is the experimental pairs, second is the training pairs
    """

    # Decide if we are finding sessions or using the provided sessions
    if sessions:
        exp_session_names = sessions
        train_session_names = sessions
    else:
        exp_session_names = meta_session.find_session_dirs(preset["default_server"])
        train_session_names = meta_session.find_session_dirs(preset["default_training_server"])

    experimental_raw_proc_pairs = _fetch_session_dirs(
        preset["default_server"], preset["processed_server"], exp_session_names,
        remove_missing_sessions=remove_missing_sessions)

    if does_training_servers_exist(preset):
        training_raw_proc_pairs = _fetch_session_dirs(
            preset["default_training_server"], preset["processed_training_server"],
            train_session_names, remove_missing_sessions=remove_missing_sessions)
    else:
        training_raw_proc_pairs = []

    return experimental_raw_proc_pairs, training_raw_proc_pairs


def does_training_servers_exist(preset, verbose=False):
    """Returns True if all training servers exist in preset

    Args:
        preset (dict): the preset in question
        verbose (bool, optional): Whether or not to print warnings. Defaults to False.

    Returns:
        bool: True if all training servers exist in preset else False
    """

    keys = ["default_training_server", "processed_training_server"]
    if not all(k in preset for k in keys):
        raise ValueError("Missing keys in preset: {}".format(keys))

    training_servers_not_none = all(v is not None for v in (preset[k] for k in keys))

    training_servers_exist = False
    if training_servers_not_none:
        training_servers_exist = all(os.path.exists(v) for v in (preset[k] for k in keys))

    if not training_servers_not_none and verbose:
        ws("One or more training servers are None in preset: {}".format(preset["names"][0]))

    if not training_servers_exist and verbose:
        ws("One or more training servers do not exist in preset: {}".format(preset["names"][0]))

    return training_servers_not_none and training_servers_exist


def apply_to_sessions_helper(rserv, pserv, preset, temp, func, args=None, sessions=None):
    """Apply a function to raw and processed session folders found in rserv and pserv. Will create
    the processed session folder if it does not exist.

    Args:
        rserv (str): Raw server sessions directory.
        pserv (str): Processed server sessions directory.
        preset (dict): Session preset.
        temp (str): Temporary directory for logging usually C:\tmp.
        func (callable): The function to apply.
        - Assumes that the function takes the following call signature:
        - func(r_server_session, p_server_session, preset, session, *args)
        args (list, optional): Additional arguments to pass to the function. Defaults to ().
        sessions (list, optional): List of sessions to process. Defaults to [] (i.e. all sessions).

    Raises:
        ValueError: if server directory does not exist or is inaccessible.

    Returns:
        list: List of failed sessions.
    """
    if args is None:
        args = ()
    if sessions is None:
        sessions = []

    logs.setup_logging(temp, sessions_dir=pserv)

    if not os.path.exists(rserv):
        raise ValueError("Server directory {} does not exist or is inaccessible.".format(rserv))

    if len(sessions) == 0:
        found_sessions = meta_session.find_session_dirs(rserv)
    else:
        found_sessions = [s for s in sessions if os.path.exists(os.path.join(rserv, s))]

    found_sessions.sort(reverse=True)
    rs("Found {} sessions: {}".format(len(found_sessions), ", ".join(found_sessions)))

    failed_sessions = []
    failed_sessions_errors = []
    for session in tqdm.tqdm(found_sessions, ncols=100, desc="Sessions"):
        print()
        rs("Processing session {}.".format(session))

        r_server_session = os.path.normpath(os.path.join(rserv, session))
        p_server_session = os.path.normpath(os.path.join(pserv, session))

        if not os.path.exists(r_server_session):
            ws("Session {} does not exist on the server.".format(r_server_session))
            continue

        if not os.path.exists(p_server_session):
            rs("Creating processed server session directory {}".format(p_server_session))
            os.makedirs(p_server_session)

        try:
            func(r_server_session, p_server_session, preset, session, *args)

        except Exception:
            print()
            ws("Function {} failed.".format(session))
            _, exc_value, exc_traceback = sys.exc_info()
            error_str = "".join(traceback.format_exception(None, exc_value, exc_traceback))
            ws(error_str)
            failed_sessions.append(session)
            failed_sessions_errors.append(error_str)

    if len(failed_sessions) > 0:
        print()
        ws("Failed running (func) for sessions:")
        for fs, fse in zip(failed_sessions, failed_sessions_errors):
            ws("\t{}: {}".format(fs, fse))
