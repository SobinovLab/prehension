#!python3.7
import os
import copy
import argparse
import time
import datetime
import multiprocessing
import tqdm
import numpy as np
import ncams
from reporting_pool import ReportingPool

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import io_tools
from prehension import meta_session



THORAX_DOF_NAMES = ('Thorax_tra1', 'Thorax_tra2', 'Thorax_tra3',
                    'Thorax_rot1', 'Thorax_rot2', 'Thorax_rot3')


def get_median_thorax_position(trial, thorax_dof_names, median_positions, i_trial):
    dof_names, _, dofs = ncams.io_utils.import_mot(trial.base_kinematic_filename)

    for i_dof, dof_name in enumerate(thorax_dof_names):
        median_positions[i_trial][i_dof] = np.nanmedian(dofs[dof_names.index(dof_name)])


def main(server, sessions, trials_sel, temp, processes, overwrite):
    thorax_dof_names = THORAX_DOF_NAMES
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

        ncams.inverse_kinematics.set_opensim_model_default_position(
            mstruct['opensim_model'], mstruct['opensim_model_locked_base'], median_position,
            lock=True)

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Calculates and saves median base body position (thorax) in the'
                     ' OpenSim model.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp, args.processes,
         args.overwrite)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
