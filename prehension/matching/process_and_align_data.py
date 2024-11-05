#!python3.7
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tqdm
from reporting_pool import ReportingPool

from .. import io_tools
from .. import meta_session
from .. import tools
from ..tools import rs, ws

JA_FREQUENCY = 50
JA_TIME_PERIOD_MS = 1000 / JA_FREQUENCY


# to be run in parallel
def transform_trial(trial, mdof, ja_filter, object_def, make_plots=False):
    # load joint angles
    f_dof_names, ja_times, dofs = io_tools.import_mot(
        trial.pre_kinematic_filename)
    ja_times = np.array(ja_times)

    # offset ja_times, ja are not offset to TTL yet
    ja_times = ja_times + trial.ttl_to_ja_start
    ja_period = np.median(np.diff(ja_times))

    # process joint angles
    dofs2 = []
    for i_dof, (dof_name, dof_info) in enumerate(mdof.items()):
        dof = np.array(dofs[f_dof_names.index(dof_name)])

        # filter the data
        if ja_filter is not None:
            dof = ja_filter(dof)

        # put data into bounds
        dof = tools.enforce_rom(dof, dof_info['range'])

        dofs2.append(dof)
        # used to find the correct row
        dof_info['index'] = i_dof
    dofs = np.array(dofs2)

    # load pressure sensors
    ps_times_tot = []
    ps_matrices_tot = []
    # and find common range of time
    tmin = ja_times[0]
    tmax = ja_times[-1]
    for ps_filename in trial.get_pre_ps_filenames().values():
        ps_times, ps_matrices = io_tools.import_matrices(ps_filename)

        # range of time
        tmin = max((tmin, ps_times[0]))
        tmax = min((tmax, ps_times[-1]))

        ps_times_tot.append(ps_times)
        ps_matrices_tot.append(ps_matrices)
    common_times = np.arange(tmin, tmax, ja_period)
    n_times = len(common_times)

    # trim to the common time period
    time_slice = tools.get_slice_to_time_base(tmin, n_times, ja_times)
    dofs = dofs[:, time_slice]
    dofs_prefilter = copy.deepcopy(dofs)
    # downsample pressure sensors
    for i_ps, ps_times in enumerate(ps_times_tot):
        ps_matrices_tot[i_ps] = tools.downsample_at_timeseries(
            ps_times, ps_matrices_tot[i_ps], common_times)

    # find the active period - from the first crossing of 5% to the last
    ps_force_summed = np.zeros(np.size(common_times))
    for pmt in ps_matrices_tot:
        ps_force_summed += np.sum(pmt, axis=(1, 2))
    ps_force_summed_thr = np.max(ps_force_summed) * 0.05
    ps_force_above_thr = ps_force_summed >= ps_force_summed_thr
    ap_start = tools.find_first(ps_force_above_thr)
    ap_end = tools.find_last(ps_force_above_thr) + 1  # end + 1
    ap_mask = np.zeros(np.size(common_times)).astype(bool)
    ap_mask[ap_start:ap_end] = True

    # redo the pressure sensor position from average center position and recorded desired
    # rotation angle. Use minimum width between the sensor surfaces as minimum width measured with
    # finger pads?
    # width estimate:
    estimated_width_offset = 0  # mm
    estimated_object_width = (
        estimated_width_offset + object_def['pos_aperture(mm)'] / 1000.)
    # measuring from the data instead
    object_width = np.median(dofs[[mdof['ps_halfwidth_tra']['index'],
                                  mdof['ps_halfwidth_tra_d']['index']]]) * 2
    if abs(object_width - estimated_object_width) > 0.05:
        ws('Calculated object width ({} m) is different from what it should be ({} m).'.format(
            object_width, estimated_object_width))
    if abs(object_width - estimated_object_width) > 0.10:
        ValueError('Calculated object width ({} m) is critically different from what it should be'
                   ' ({} m).'.format(
                       object_width, estimated_object_width))
    dofs[mdof['ps_halfwidth_tra']['index']] = object_width / 2
    dofs[mdof['ps_halfwidth_tra_d']['index']] = object_width / 2
    # tra
    ps_tra_i_1 = mdof['ps_tra1']['index']
    ps_tra_i_2 = mdof['ps_tra2']['index']
    ps_tra_i_3 = mdof['ps_tra3']['index']
    # find the median of the final object position and replace with it the hold period position
    dofs[ps_tra_i_1, ap_mask] = np.median(dofs[ps_tra_i_1, ap_mask])
    dofs[ps_tra_i_2, ap_mask] = np.median(dofs[ps_tra_i_2, ap_mask])
    dofs[ps_tra_i_3, ap_mask] = np.median(dofs[ps_tra_i_3, ap_mask])
    # rot
    ps_rot_i_1 = mdof['ps_rot1']['index']
    ps_rot_i_2 = mdof['ps_rot2']['index']
    ps_rot_i_3 = mdof['ps_rot3']['index']
    # find the median of the final object position and replace with it the hold period position
    dofs[ps_rot_i_1, ap_mask] = np.median(dofs[ps_rot_i_1, ap_mask])
    dofs[ps_rot_i_2, ap_mask] = np.median(dofs[ps_rot_i_2, ap_mask])
    dofs[ps_rot_i_3, ap_mask] = np.median(dofs[ps_rot_i_3, ap_mask])

    # Save processed ja data
    io_tools.export_mot(trial.post_kinematic_filename_mot,
                        list(mdof.keys()), common_times, dofs)

    # export to CSV with rotational transformed to radians
    rots = np.array([dof_info['rot']
                    for dof_info in mdof.values()], dtype=bool)
    dofs[rots, :] = dofs[rots, :] / 180 * np.pi
    dofs_prefilter[rots, :] = dofs_prefilter[rots, :] / 180 * np.pi
    io_tools.export_csv(trial.post_kinematic_filename_csv,
                        ['time'] + list(mdof.keys()), [common_times] + dofs.tolist())

    # save processed pressure sensor data
    for ps_filename, ps_matrices in zip(trial.post_ps_tsm_filenames.values(), ps_matrices_tot):
        io_tools.export_tsm_matrix(
            ps_filename, common_times, ps_matrices, type='period')

    if make_plots:
        ps_dofs = [k for k in mdof.keys() if k[:3] ==
                   'ps_' and k[-2:] != '_d']
        dependent_dofs = [k for k in mdof.keys() if k[-2:] == '_d']
        anat_dofs = [k for k in mdof.keys(
        ) if k not in ps_dofs and k not in dependent_dofs]

        plot_some_dofs(
            'ps dofs trial {}'.format(
                trial.trial_number), common_times, dofs,
            dofs_prefilter, mdof, ps_dofs)
        plot_some_dofs(
            'anatomical dofs trial {}'.format(
                trial.trial_number), common_times, dofs,
            dofs_prefilter, mdof, anat_dofs)

        plt.figure()
        ax = plt.subplot(2, 1, 1)
        for dof_name in anat_dofs:
            ax.plot(common_times, dofs_prefilter[mdof[dof_name]['index']] -
                    dofs_prefilter[mdof[dof_name]['index']][0], 'k')
        ax.set_ylabel('DOFs, au')

        ax = plt.subplot(2, 1, 2, sharex=ax)
        for ps_matrices in ps_matrices_tot:
            ax.plot(common_times, np.sum(
                ps_matrices, axis=(1, 2)), 'k')
        ax.set_ylabel('PS, N')
        ax.set_xlabel('Time to TTL, s')


def plot_some_dofs(label, times, dofs, dofs_prefilter, mdof, dof_names, figsize=(16, 9)):
    xn_subplots, yn_subplots = tools.xy_numsubplots(len(dof_names))

    fig, axs = plt.subplots(
        xn_subplots, yn_subplots, sharex=True, figsize=figsize)
    axs = axs.flatten()
    plt.suptitle(label)
    axs_tra = []
    ax_rot = None

    for i_dof, (dof_name, ax) in enumerate(zip(dof_names, axs)):
        ax.plot(times, dofs_prefilter[mdof[dof_name]['index']], 'k')
        ax.plot(times, dofs[mdof[dof_name]['index']], 'r--')
        ax.set_ylabel(dof_name)
        ax.set_xlabel('Time, s')
        if mdof[dof_name]['rot']:
            if ax_rot is None:
                ax_rot = ax
            else:
                ax.sharey(ax_rot)
        else:
            axs_tra.append(ax)
    tools.match_yaxes_ranges(axs_tra)


def ja_filter(data):
    # butterworth filter
    sos = scipy.signal.butter(
        2, 20, btype='lowpass', output='sos', fs=JA_FREQUENCY)
    data = scipy.signal.sosfilt(sos, data)

    # gaussian
    gaussian_sd = 100  # ms
    gaussian_sd /= JA_TIME_PERIOD_MS
    data = scipy.ndimage.gaussian_filter1d(
        data, gaussian_sd, mode='reflect')

    # data = scipy.ndimage.median_filter(data, size=3, mode='nearest')

    return data


def process_and_align_data(server, sessions, trials_sel, temp, processes, overwrite, make_plots):
    """Filters, resamples, and aligns pressure sensor and kinematic data to grasp onset.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        preset {dict} --- Preset dictionary.
        make_plots {bool} --- Makes some inspection figures. Run with --processes 1.
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
    rs('Found {} sessions: {}'.format(
        len(sessions), ', '.join(sessions)))

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
            mstruct, mdof, mobject, msession = meta_session.load_meta_information(
                server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(
                session))
            ws('Error message: {}'.format(e))
            continue

        # accumulate data
        trials = []
        object_defs = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_all_pre_files_exist():
                continue
            if not overwrite and trial.do_all_post_files_exist():
                continue
            trials.append(trial)
            object_defs.append(copy.deepcopy(
                mobject[trial.object_id]['def']))

       # Just continue if no trials
        if not trials:
            continue

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        os.makedirs(mstruct['post_ja_dir'], exist_ok=True)
        os.makedirs(mstruct['post_ps_dir'], exist_ok=True)

        p_args = list(zip(*[
            trials,
            [copy.deepcopy(mdof) for _ in trials],
            [ja_filter for _ in trials],
            object_defs,
            [make_plots for _ in trials]
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
                    ws('\t{}: {}'.format(
                        trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append('session {} trial {} error: {}'.format(
                        session, trials[v].trial_number, pool.error_reports[v]))

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
