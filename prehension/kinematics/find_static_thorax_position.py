#!python3
# -*- coding: utf-8 -*-
"""
Finds a median thorax position to fix the skeleton in place.

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
import copy
import multiprocessing

import numpy as np
import tqdm
from reporting_pool import ReportingPool

from .. import meta_session
from ..tools import logs
from ..tools import io
from ..tools.logs import rs, ws
from ..tools.constants import THORAX_DOF_NAMES

from . import inverse_kinematics


def get_median_thorax_position(trial, thorax_dof_names, median_positions, i_trial):
    dof_names, _, dofs = io.import_mot(trial.base_kinematic_filename)

    for i_dof, dof_name in enumerate(thorax_dof_names):
        median_positions[i_trial][i_dof] = np.nanmedian(dofs[dof_names.index(dof_name)])


def find_static_thorax_position(server, sessions, trials_sel, temp, processes, overwrite):
    """Calculates and saves median base body position (thorax) in the OpenSim model.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed
            trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
    """
    thorax_dof_names = THORAX_DOF_NAMES
    logs.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    failed_trial_reports = []
    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, mdof, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        if not overwrite and os.path.exists(mstruct['opensim_model_locked_base']):
            rs('Locked base model already exists, skipping.')
            continue

        # calculate median positions - parallel
        manager = multiprocessing.Manager()
        median_positions = manager.list()
        trials = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.does_post_base_ik_file_exists():
                continue
            median_positions.append(manager.list())
            for _ in thorax_dof_names:
                median_positions[-1].append(np.nan)
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        if len(trials) == 0:
            ws('Not enough trials, skipping.')
            continue

        p_args = list(zip(*[
            trials,
            [copy.deepcopy(thorax_dof_names) for _ in trials],
            [median_positions for _ in trials],
            list(range(len(trials))),
        ]))

        # run the pool
        if len(p_args) > 0:
            pool = ReportingPool(get_median_thorax_position, p_args, processes=processes,
                                 report_on_change=True, track_failures=True)
            pool.start()

            if len(pool.failed_i_jobs) > 0:
                print()
                ws('Failed to transform trials:')
                for v in pool.failed_i_jobs:
                    ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append('session {} trial {} error: {}'.format(
                        session, trials[v].trial_number, pool.error_reports[v]))

        # unpack and save
        median_position = {}
        for i_dof, dof_name in enumerate(thorax_dof_names):
            median_position[dof_name] = np.nanmedian([mp[i_dof] for mp in median_positions])

        rs('Found thorax median positions: {}'.format(
            ', '.join(['{}: {}'.format(k, v) for k, v in median_position.items()])))

        # change rotations to radians
        for dof_name in thorax_dof_names:
            if mdof[dof_name]['rot']:
                median_position[dof_name] *= np.pi / 180

        inverse_kinematics.set_opensim_model_default_position(
            mstruct['opensim_model'], mstruct['opensim_model_locked_base'], median_position,
            lock=True)

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
