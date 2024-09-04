#!python3.8
import os

import tqdm
from reporting_pool import ReportingPool

from . import meta_session
from . import tools
from .tools import rs, ws


def run_ik_f(ik_file, log_file):
    # if log file exists, remove it
    if os.path.exists(log_file):
        os.remove(log_file)

    # needed for independent logging
    import opensim
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(log_file)
    opensim.Logger.setLevelString('warn')
    task = opensim.tools.InverseKinematicsTool(ik_file)

    task.run()


def inverse_kinematics(server, sessions, trials_sel, temp, processes, overwrite, base):
    """Runs the inverse kinematics OpenSim tool.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        base {bool} --- Runs inverse kinematics on the most proximal markers
            that can be used to estimate the default static thorax position.
    """
    tools.setup_logging(temp, sessions_dir=server)

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
            _, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        trials = []
        ik_files = []
        log_files = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if base:
                if not trial.do_pre_base_ik_files_exist():
                    continue
                if not overwrite and trial.does_post_base_ik_file_exists():
                    continue
                ik_files.append(trial.base_ik_filename)
            else:
                if not trial.do_pre_ik_files_exist():
                    continue
                if not overwrite and trial.does_post_ik_file_exists():
                    continue
                ik_files.append(trial.ik_filename)
            log_files.append(trial.ik_log_filename)
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(trial.trial_number) for trial in trials])))

        p_args = list(zip(*[ik_files, log_files]))

        if len(p_args) > 0:
            pool = ReportingPool(run_ik_f, p_args, processes=processes,
                                 report_on_change=True, track_failures=True)
            pool.start()

            if len(pool.failed_i_jobs) > 0:
                print()
                ws('Failed to transform trials:')
                for v in pool.failed_i_jobs:
                    ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append('session {} trial {} error: {}'.format(
                        session, trials[v].trial_number, pool.error_reports[v]))

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed converting trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
