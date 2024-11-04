#!python3.7
import copy
import multiprocessing
import os

import numpy as np
import tqdm
from reporting_pool import ReportingPool

from .. import io_tools
from .. import meta_session
from .. import tools
from ..tools import rs, ws


def ignore_dof_check(s):
    '''Dependent DOFs or thorax movement'''
    return (len(s) > 2 and s[-2:] == '_d') or (len(s) > 6 and s[:7] == 'Thorax_')


def calculate_optimal_time(trial, mdof, optimal_frames, i_trial):
    column_names, values = io_tools.import_csv(trial.post_kinematic_filename_csv)
    ja_times = values[column_names.index('time')]
    dof_names = []
    dofs = []
    for cn, vs in zip(column_names, values):
        if not ignore_dof_check(cn) and cn != 'time':
            dof_names.append(cn)
            dofs.append(vs)

    # load pressure sensors
    ps_matrices_reduced_all = []
    for ps_filename in trial.get_post_ps_filenames().values():
        _, ps_matrices = io_tools.import_matrices(ps_filename)
        ps_matrices_reduced_all.append(np.sum(ps_matrices, axis=(1, 2)))

    # finding best sensor optimization period
    # first iteration - normalized sum of all sensors
    metric = np.zeros(np.shape(ja_times))
    stability = np.zeros(np.shape(ja_times))
    for pmra in ps_matrices_reduced_all:
        metric += pmra / np.max(pmra)
    # second iteration - slight JA stability metric
    for dof_name, dof in zip(dof_names, dofs):
        diff_dof = np.abs(np.diff(dof))
        diff_dof = np.insert(diff_dof, 0, 0)
        diff_dof /= mdof[dof_name]['range'][1] - mdof[dof_name]['range'][0]
        stability += diff_dof
    stability /= np.max(stability)
    # stability is low when diff dof is high: stability = 1 - stability
    # calculate in-place
    stability *= -1
    stability += 1
    metric += stability
    # calculate
    maxval = np.amax(metric)
    max_metric_i = np.where(metric == maxval)[0][0]
    # max_metric_t = ja_times[max_metric_i]

    # return
    optimal_frames[i_trial] = max_metric_i


def find_optimal_frames(server, sessions, trials_sel, temp, processes, overwrite):
    """Find optimal frames that represent a static grasping posture.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(server))

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

        # TODO load calculated to reuse?
        # No real point since don't take too long to recalculate and not used any more
        optimal_frames_filename = os.path.join(server_session, 'optimal_frames.csv')
        if not overwrite and os.path.exists(optimal_frames_filename):
            rs('Optimal frames file already exists, skipping.')
            continue

        # load session meta
        try:
            _, mdof, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # calculate best period - parallel
        manager = multiprocessing.Manager()
        optimal_frames = manager.list()
        trials = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_all_post_files_exist():
                continue
            optimal_frames.append(-1)
            trials.append(trial)

        # Just continue if no trials
        if not trials:
            continue

        rs(
            'Found {} trials: {}'.format(
                len(trials), ', '.join([str(t.trial_number) for t in trials])
            )
        )

        p_args = list(
            zip(
                *[
                    trials,
                    [copy.deepcopy(mdof) for _ in trials],
                    [optimal_frames for _ in trials],
                    list(range(len(trials))),
                ]
            )
        )

        # run the pool
        if len(p_args) > 0:
            pool = ReportingPool(
                calculate_optimal_time,
                p_args,
                processes=processes,
                report_on_change=True,
                track_failures=True,
            )
            pool.start()

            if len(pool.failed_i_jobs) > 0:
                print()
                ws('Failed to transform trials:')
                for v in pool.failed_i_jobs:
                    ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append(
                        'session {} trial {} error: {}'.format(
                            session, trials[v].trial_number, pool.error_reports[v]
                        )
                    )

        # unpack and save
        meta_session.export_optimal_frames(
            optimal_frames_filename, [t.trial_number for t in trials], optimal_frames
        )
        rs('Exported optimal frames to {}.'.format(optimal_frames_filename))

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
