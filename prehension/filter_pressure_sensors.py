#!python3.7
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import tqdm
from reporting_pool import ReportingPool

from . import io_tools
from . import meta_session
from . import tools
from .tools import rs, ws


# to be run in parallel
def transform_trial(trial, make_plots):
    # if the TSM does not exist, generate it from CSV
    for ps_name in trial.transformed_ps_filenames.keys():
        if not os.path.exists(trial.transformed_ps_filenames[ps_name]):
            # will break if the CSV does not exist - then run a MATLAB fsx->tsm program
            times, matrices = io_tools.import_csv_matrix_low(
                trial.transformed_ps_csv_filenames[ps_name])
            io_tools.export_tsm_matrix(
                trial.transformed_ps_filenames[ps_name], times, matrices, type='stamps')

    # remove the CSV if it exists
    for ps_name in trial.transformed_ps_filenames.keys():
        # double check to be safe
        if (os.path.exists(trial.transformed_ps_csv_filenames[ps_name]) and
                os.path.exists(trial.transformed_ps_filenames[ps_name])):
            os.remove(trial.transformed_ps_csv_filenames[ps_name])

    # filter the pressure sensor data
    for ps_name in trial.transformed_ps_filenames.keys():
        times, matrices = io_tools.import_matrices(trial.transformed_ps_filenames[ps_name])

        force_summed = np.sum(matrices, axis=(1, 2))
        force_summed_thr = np.max(force_summed) * 0.05

        # periods of active pressure
        ap_mask = np.zeros(np.size(times)).astype(bool)
        ap_mask[force_summed > force_summed_thr] = True
        nap_mask = np.logical_not(ap_mask)

        # if there was any touch
        if any(nap_mask):
            # remove sensel activity below the noise level - 90 percentile during no touch
            thrs = np.quantile(matrices[nap_mask, :, :], 0.9, axis=0)
            # for r in thrs:
            #     print(' '.join(str(e) for e in r))
            matrices[matrices <= thrs] = 0.

        # median filter
        for irow in range(len(matrices[0])):
            for icol in range(len(matrices[0][irow])):
                matrices[:, irow, icol] = scipy.ndimage.median_filter(
                    matrices[:, irow, icol], size=3, mode='constant', cval=0.)

        # export
        io_tools.export_tsm_matrix(trial.filtered_ps_filenames[ps_name], times, matrices,
                                   type='stamps')

    # testing plot
    if make_plots:
        fig, axs = plt.subplots(len(trial.transformed_ps_filenames.keys()), figsize=(16, 9))
        axs = axs.flatten()
        for ps_name, axs in zip(trial.transformed_ps_filenames.keys(), axs):
            times, matrices = io_tools.import_matrices(trial.transformed_ps_filenames[ps_name])
            axs.plot(times, np.sum(matrices, axis=(1, 2)), 'k')
            times, matrices = io_tools.import_matrices(trial.filtered_ps_filenames[ps_name])
            axs.plot(times, np.sum(matrices, axis=(1, 2)), 'r--')
            axs.set_xlabel('Time, s')
            axs.set_ylabel('Force, N')
            axs.set_title(ps_name)
        fig.set_suptitle('Trial {}'.format(trial.trial_number))


def filter_pressure_sensors(server, sessions, trials_sel, temp, processes, overwrite, make_plots, preset):
    """Compare manually-labeled to the automatically-labeled forces using sensor masks.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        make_plots {bool} --- Makes some inspection figures.
        preset {dict} --- Preset dictionary.
    """
    tools.setup_logging(temp, sessions_dir=preset['processed_server'])

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
        processed_session = os.path.join(preset['processed_server'], session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(server_session,
                                                                                  processed_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # accumulate data
        trials = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue

            # at least one of the files should exist
            if any([not os.path.exists(trial.transformed_ps_filenames[ps_name]) and
                    not os.path.exists(trial.transformed_ps_csv_filenames[ps_name])
                    for ps_name in trial.transformed_ps_filenames.keys()]):
                continue

            if not overwrite and all([os.path.exists(fpf)
                                      for fpf in trial.filtered_ps_filenames.values()]):
                continue
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        if len(trials) == 0:
            continue

        os.makedirs(mstruct['pre_ps_dir'], exist_ok=True)

        p_args = list(zip(*[
            trials,
            [make_plots]*len(trials)
        ]))

        # transform_trial(*(p_args[0]))
        # sys.exit()

        if len(p_args) > 0:
            pool = ReportingPool(transform_trial, p_args, processes=processes,
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
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
