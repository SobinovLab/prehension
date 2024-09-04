#!python3.7
import os
import pandas as pd
import numpy as np
import tsm
import sys
import ctypes
import argparse
import clr  # pip install pythonnet (module is clr-loader)
import tqdm
import datetime
import time
import inspect
import reporting_pool

# Example call:
# py preprocess_pressure_sensors.py cr_local --sessions 2022_04_14_Set1
# py preprocess_pressure_sensors.py cr_test --server "C:\PrehensionDataLocal\MojitoRightHemisphere"
# --sessions 2022_04_19_Set1 --overwrite
# Linting: py -3.7 -m pycodestyle
# --first .\transform_pressure_sensor_data\preprocess_pressure_sensors.py --max-line-length=100
# include local library functions - TB included in NCams
# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import meta_session

currentdir = os.path.dirname(os.path.abspath(
    inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)

dll_dir = os.path.join(parentdir, 'TekDLL')
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


def main(server, sessions, trials_sel, temp, overwrite, processes, preset):
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

    # Iterate over dates
    failed_trial_reports = []
    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        raw_server_session = os.path.join(server, session)
        proc_server_session = os.path.join(preset['processed_server'], session)

        if not os.path.exists(raw_server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(raw_server_session, proc_server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        output_dir = mstruct['transformed_ps_dir']
        trial_log_filename = mstruct['ps_log_filename']
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.isfile(trial_log_filename):
            ws(f"Trial log {trial_log_filename} file for session {raw_server_session} does not exist")
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

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        if len(trials) == 0:
            continue

        trial_timestamps = loadTrialLog(trial_log_filename, [t.trial_number for t in trials])

        p_args = list(zip(trials, trial_timestamps))

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


if __name__ == '__main__':
    # Add arguments
    preset_name, current_preset, argv = preset.process_args_for_preset()
    parser = argparse.ArgumentParser(
        description=('Creates meta information for a session.'))

    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    tools.add_default_arguments(parser, ('sessions', 'trials', 'temp', 'overwrite', 'processes'))
    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp,
         args.overwrite, args.processes, current_preset)

    rs('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
