#!python3
# -*- coding: utf-8 -*-
"""
Creates IK and Scaling files for OpenSim based on a period of trial.

Copyright (C) 2023-2024 Anton Sobinov, Caleb Raman
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

import numpy as np
import scipy
import scipy.ndimage
from reporting_pool import ReportingPool

from ..tools import io
from .. import meta_session
from ..tools.logs import rs, ws
from ..tools.session_management import apply_to_sessions_helper


# to be run in parallel
def transform_trial(trial):
    # if the TSM does not exist, generate it from CSV
    for ps_name in trial.transformed_ps_filenames.keys():
        if not os.path.exists(trial.transformed_ps_filenames[ps_name]):
            # will break if the CSV does not exist - then run a MATLAB fsx->tsm program
            times, matrices = io.import_csv_matrix_low(trial.transformed_ps_csv_filenames[ps_name])
            io.export_tsm_matrix(trial.transformed_ps_filenames[ps_name], times, matrices,
                                 type='stamps')

    # remove the CSV if it exists
    for ps_name in trial.transformed_ps_filenames.keys():
        # double check to be safe
        if os.path.exists(trial.transformed_ps_csv_filenames[ps_name]) and os.path.exists(
                trial.transformed_ps_filenames[ps_name]):
            os.remove(trial.transformed_ps_csv_filenames[ps_name])

    # filter the pressure sensor data
    for ps_name in trial.transformed_ps_filenames.keys():
        times, matrices = io.import_matrices(trial.transformed_ps_filenames[ps_name])

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
            matrices[matrices <= thrs] = 0.0

        # median filter
        for irow in range(len(matrices[0])):
            for icol in range(len(matrices[0][irow])):
                matrices[:, irow, icol] = scipy.ndimage.median_filter(matrices[:, irow, icol],
                                                                      size=3, mode='constant',
                                                                      cval=0.0)
        # export
        io.export_tsm_matrix(trial.filtered_ps_filenames[ps_name], times, matrices, type='stamps')



def fps_helper(raw_ss, proc_ss, _, session, trials_sel, overwrite, processes):
    """TODO - fill in
    """

    try:
        mstruct, _, _, msession = meta_session.load_meta_information(raw_ss, proc_ss)
    except Exception as e:
        ws('Could not load meta data from session {}, skipping.'.format(session))
        ws('Error message: {}'.format(e))
        return

    # transport sessions have no pressure sensor data, skip them
    if mstruct.get('experiment_type') == 'transport':
        ws('Session {} is a transport session, skipping.'.format(session))
        return

    output_dir = mstruct['pre_ps_dir']
    os.makedirs(output_dir, exist_ok=True)

    # accumulate trials
    trials = []
    for trial in msession:
        if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
            continue
        # Skip if missing input transformed tsm files
        if any([not os.path.exists(trial.transformed_ps_filenames[ps_name])
                for ps_name in trial.transformed_ps_filenames.keys()]):
            continue
        # Skip if output files exist and overwrite==False
        if not overwrite and all([os.path.exists(fpf)
                                  for fpf in trial.filtered_ps_filenames.values()]):
            continue
        trials.append(trial)

    # Just continue if no trials
    if not trials:
        return

    rs('Found {} trials: {}'.format(len(trials), ', '.join([str(t.trial_number) for t in trials])))

    failed_trial_reports = []
    p_args = list(zip(*[trials]))

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


def filter_pressure_sensors(current_preset, trials_sel, temp, overwrite, processes,
                            sessions_sel=[]):
    """Filter pressure sensor data - output folder called transformed_sensors
    """

    # Step 1: process experiment sessions
    apply_to_sessions_helper(current_preset['default_server'],
                             current_preset['processed_server'],
                             current_preset,
                             temp,
                             fps_helper,
                             args=(trials_sel, overwrite, processes),
                             sessions=sessions_sel)

    # Step 2: process training sessions
    if not current_preset['default_training_server']:
        ws('No raw training server specified in preset. Skipping...')
        return

    if not current_preset['processed_training_server']:
        ws('No processed training server specified in preset. Skipping...')
        return

    # Step 2: Process training sessions
    apply_to_sessions_helper(current_preset['default_training_server'],
                             current_preset['processed_training_server'],
                             current_preset,
                             temp,
                             fps_helper,
                             args=(trials_sel, overwrite, processes),
                             sessions=sessions_sel)
