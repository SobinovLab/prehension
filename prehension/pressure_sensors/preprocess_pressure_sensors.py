#!python3
# -*- coding: utf-8 -*-
"""
Creates IK and Scaling files for OpenSim based on a period of trial.

Copyright (C) 2023-2024 Caleb Raman
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
import ctypes
import inspect
import os
import sys

import clr  # pip install pythonnet (module is clr-loader)
import numpy as np
import pandas as pd
import reporting_pool
import tqdm
import timed_sparse_matrix as tsm

from .. import meta_session
from ..tools import logs
from ..tools.logs import rs, ws
from ..tools.utils import apply_to_sessions_helper

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))

dll_dir = os.path.join(currentdir, 'TekDLL')
sys.path.append(dll_dir)
clr.AddReference("TekAPI64")
from TekAPI import CTekAPI

# CONSTANTS
NEWTONS_PER_LBS = 4.4482216


def calibrationCheck(f, filename):
    fname, _ = os.path.splitext(os.path.basename(filename))
    if f.TekIsCalibrated() == CTekAPI.TEK_OK:
        return 0
    else:
        ws(f'File {fname} is NOT calibrated, aborting.')
        # Calibrations are always saved with a file. This blocker is to
        # make sure the correct calibrations are specified.
        return -1


def loadTrialLog(trial_log_filename, trials):
    f = pd.read_csv(trial_log_filename)
    trial_timestamps = []
    for trial in trials:
        row = f[f['trial_num'] == trial].iloc[0]
        trial_timestamps.append([row['startedRecording(ms)'],
                                 row['syncTrialStartTime(ms)'],
                                 row['syncTrialEndTime(ms)'],
                                 row['finishedRecording(ms)']])
    return np.array(trial_timestamps)


def loadFsxFile(filename):
    # Load the file
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File {filename} does not exist")

    handle = CTekAPI.TekLoadRecording(filename)
    assert handle is not None

    if calibrationCheck(handle, filename) < 0:
        return None, None, None, None

    # Extract values
    N = handle.TekGetFrameCount()
    _, sensel_area = handle.TekGetSenselArea()
    rows = handle.TekGetRows()
    cols = handle.TekGetColumns()

    times = np.zeros(N, dtype=np.float64)
    forces = np.zeros((N, rows, cols), dtype=np.float64)

    for iFrame in range(N):
        times[iFrame] = handle.TekGetFrameTimestamp(float(-1), iFrame)[1]

        placeholder = (ctypes.c_float * (rows * cols))()
        retcode, data = handle.TekGetCalibratedFrameData(placeholder, iFrame)
        assert retcode == 0, f'TekGetCalibratedFrameData[0] = {retcode}'

        data = np.ctypeslib.as_array(data).reshape((rows, cols))

        forces[iFrame][:, :] = data.T

    forces *= sensel_area  # in pounds
    forces *= NEWTONS_PER_LBS  # convert to newtons

    return times, forces


def fsxToTsm(filename, o_filename, trial_timestamps):
    # Import frame data
    times, forces = loadFsxFile(filename)

    # Resynchronize times
    sync_offset = (trial_timestamps[1] - trial_timestamps[0]) * 0.001
    times -= sync_offset

    # Export data to TSM format
    tsm.save(o_filename, 'stamps', times, forces, 0.0)


def process_trial(trial, trial_timestamp):
    for ps_name in trial.raw_ps_filenames.keys():
        fsxToTsm(trial.raw_ps_filenames[ps_name],
                 trial.transformed_ps_filenames[ps_name], trial_timestamp)


def ppps_helper(raw_ss, proc_ss, _, session, trials_sel, overwrite, processes):
    """preprocess_pressure_sensors helper function. Runs preprocess_pressure_sensors on single
    session.

    Args:
        raw_ss (dir): raw session folder - used for loading meta
        proc_ss (dir): processed session folder - used for loading meta
        _ (dict): Placeholder for preset, unused but passed by apply_to_sessions
        session (str): the session name (e.g. '2022_03_08_Set1
        trials_sel (list[int]): the selected trial numbers, defaults to [] (all trials)
        overwrite (bool): whether to overwrite existing files
        processes (int): number of processes in the pool
    """
    
    try:
        mstruct, _, _, msession = meta_session.load_meta_information(raw_ss, proc_ss)
    except Exception as e:
        ws('Could not load meta data from session {}, skipping.'.format(session))
        ws('Error message: {}'.format(e))
        return

    output_dir = mstruct['transformed_ps_dir']
    trial_log_filename = mstruct['ps_log_filename']
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.isfile(trial_log_filename):
        ws(f"Trial log {trial_log_filename} file for session {raw_ss} does not exist")
        return

    # accumulate trials
    trials = []
    for trial in msession:
        if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
            continue
        # Skip if missing input fsx files
        if any([not os.path.exists(trial.raw_ps_filenames[ps_name])
                for ps_name in trial.raw_ps_filenames.keys()]):
            continue
        # Skip if output files exist and overwrite==False
        if not overwrite and all([os.path.exists(fpf)
                                    for fpf in trial.transformed_ps_filenames.values()]):
            continue
        trials.append(trial)

    # Just continue if no trials
    if not trials:
        return

    rs('Found {} trials: {}'.format(
        len(trials), ', '.join([str(t.trial_number) for t in trials])))

    trial_timestamps = loadTrialLog(trial_log_filename, [t.trial_number for t in trials])

    p_args = list(zip(trials, trial_timestamps))

    failed_trial_reports = []

    if len(p_args) > 0:
        pool = reporting_pool.ReportingPool(process_trial, p_args, processes=processes,
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

def preprocess_pressure_sensors(current_preset, trials_sel, temp, overwrite, processes, sessions_sel=[]):
    """Creates meta information for a session.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all
            unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        processes {int} --- Number of parallel processes in the pool.
        preset {dict} --- Preset dictionary.
    """

    # Step 1: process experiment sessions
    apply_to_sessions_helper(
        current_preset['default_server'],
        current_preset['processed_server'],
        current_preset,
        temp,
        ppps_helper,
        args=(trials_sel, overwrite, processes),
        sessions=sessions_sel)

    # Step 2: process training sessions
    if current_preset['default_training_server']:
        ws('No raw training server specified in preset. Skipping...')
        return

    if current_preset['processed_training_server']:
        ws('No processed training server specified in preset. Skipping...')
        return

    # Step 2: Process training sessions
    apply_to_sessions_helper(
        current_preset['default_training_server'],
        current_preset['processed_training_server'],
        current_preset,
        temp,
        ppps_helper,
        args=(trials_sel, overwrite, processes))



