#!python3.7
import os
import random
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats
import tqdm

from .. import io_tools
from .. import meta_session
from .. import tools
from ..tools import rs, ws

SEGMENT_DIGIT_GROUPS = {
    'thumb': lambda v: re.search('[RL]A[0-9][MPD]1_.*', v),
    'index': lambda v: re.search('[RL]A[0-9][MPD]2_.*', v),
    'middle': lambda v: re.search('[RL]A[0-9][MPD]3_.*', v),
    'ring': lambda v: re.search('[RL]A[0-9][MPD]4_.*', v),
    'pinky': lambda v: re.search('[RL]A[0-9][MPD]5_.*', v),
    'None': lambda v: True
}


def calculate_digit_forces(mstruct, trial):
    ps_matrices = {}
    matched_contacts = {}
    segments_set = set()
    for ps_name in mstruct['ps_dic'].keys():
        ps_times, ps_matrices[ps_name] = io_tools.import_matrices(
            trial.get_post_ps_filenames()[ps_name])
        matched_contacts[ps_name] = io_tools.import_matched_contacts(
            trial.matched_contacts_filenames[ps_name])
        for mc in matched_contacts[ps_name]:
            segments_set = segments_set.union(list(mc.keys()))
    segments_set = sorted(list(segments_set), key=lambda v: int(v[4])*10+int(v[2]))

    # group up
    segment_digit_groups = []
    for segment in segments_set:
        for i_sdg, sdg_l in enumerate(SEGMENT_DIGIT_GROUPS.values()):
            if sdg_l(segment):
                segment_digit_groups.append(i_sdg)
                break

    # build up lists of forces per segment
    data = {}
    residual_force = {}
    for ps_name in mstruct['ps_dic'].keys():
        # sugar
        ps_matrix = ps_matrices[ps_name]
        matched_sensels = np.zeros(np.shape(ps_matrix), dtype=bool)
        matched_contact = matched_contacts[ps_name]
        dat = [[] for _ in segments_set]

        for psm, mc, ms in zip(ps_matrix, matched_contact, matched_sensels):
            for iseg, segment in enumerate(segments_set):
                if segment not in mc.keys():
                    dat[iseg].append(0)
                else:
                    dat[iseg].append(sum(psm[contact[0]][contact[1]] for contact in mc[segment]))
                    for contact in mc[segment]:
                        ms[contact[0], contact[1]] = True

        # save
        data[ps_name] = dat
        residual_force[ps_name] = np.sum(np.logical_not(matched_sensels) * np.array(ps_matrix),
                                         axis=(1, 2))

    # Group the data
    data_digits = {}
    for ps_name in mstruct['ps_dic'].keys():
        data_digits[ps_name] = [np.zeros(np.shape(ps_times)) for _ in SEGMENT_DIGIT_GROUPS]
        for i_sdg, d in zip(segment_digit_groups, data[ps_name]):
            data_digits[ps_name][i_sdg] = data_digits[ps_name][i_sdg] + d
        # adding residual to None
        data_digits[ps_name][-1] += residual_force[ps_name]

    # group across pressure sensors
    data_digits_aps = {}
    # assuming there is two pressure sensors
    for i_sdg, k in enumerate(SEGMENT_DIGIT_GROUPS):
        data_digits_aps[k] = data_digits['medial_sensor'][i_sdg] + data_digits['lateral_sensor'][i_sdg]

    # store the data
    trial.digit_forces_time = ps_times
    trial.digit_forces = data_digits_aps

    # assuming time is the same
    # load validation
    v_segnames, v_vals = io_tools.import_csv(trial.manually_labelled_filename)
    timeindex = v_segnames.index('time')
    v_times = v_vals[timeindex]
    del v_vals[timeindex]
    del v_segnames[timeindex]
    # rename old style
    if 'UNCLAIMED' in v_segnames:
        v_segnames[v_segnames.index('UNCLAIMED')] = 'None'

    # store
    trial.manual_digit_forces = {k: v for k, v in zip(v_segnames, v_vals)}


def calculate_differences(trial):
    tv_differences = {}  # time-varying
    differences = {}
    difference_units = {}

    # find active period - more than 5% of max force
    total_force = np.zeros(np.shape(trial.digit_forces_time))
    for digit, a_digit_force in trial.digit_forces.items():
        total_force += a_digit_force
    max_total_force = np.max(total_force)
    summed_force = np.sum(total_force) / 5  # per digit
    active_period_flag = total_force >= (0.05 * max_total_force)

    # integrated metric
    dt = np.median(np.diff(trial.digit_forces_time))
    total_force_integral = np.sum(dt * total_force[active_period_flag])

    # deviation
    tv_differences['deviation'] = {}
    tv_differences['normalized deviation'] = {}
    differences['deviation'] = {}
    differences['normalized deviation'] = {}
    differences['integrated deviation'] = {}
    differences['integrated normed deviation'] = {}
    differences['rms'] = {}
    differences['normalized rms'] = {}
    differences['r'] = {}
    differences['r2'] = {}
    # differences['normalized2 rms'] = {}
    difference_units['deviation'] = 'N'
    difference_units['normalized deviation'] = '%'
    difference_units['integrated deviation'] = 'N s'
    difference_units['integrated normed deviation'] = '%'
    difference_units['rms'] = 'N'
    difference_units['normalized rms'] = '%'
    difference_units['r'] = 'nu'
    difference_units['r2'] = 'nu'
    # difference_units['normalized2 rms'] = '%'
    for digit in trial.digit_forces.keys():
        a_digit_force = np.array(trial.digit_forces[digit])
        m_digit_force = np.array(trial.manual_digit_forces[digit])

        ap_a_df = a_digit_force[active_period_flag]
        ap_m_df = m_digit_force[active_period_flag]

        tv_differences['deviation'][digit] = np.abs(a_digit_force - m_digit_force)
        differences['deviation'][digit] = np.mean(np.abs(ap_a_df - ap_m_df))

        tv_differences['normalized deviation'][digit] = (
            tv_differences['deviation'][digit] / max_total_force) * 100.
        differences['normalized deviation'][digit] = (
            differences['deviation'][digit] / max_total_force) * 100.

        differences['integrated deviation'][digit] = np.sum(dt * np.abs(ap_a_df - ap_m_df))
        differences['integrated normed deviation'][digit] = (
            differences['integrated deviation'][digit] / total_force_integral) * 100.

        differences['rms'][digit] = np.sqrt(np.sum(np.square(ap_a_df - ap_m_df)))
        differences['normalized rms'][digit] = (
            differences['rms'][digit] / summed_force) * 100.

        res = scipy.stats.linregress(ap_m_df, ap_a_df)
        differences['r'][digit] = res.rvalue
        differences['r2'][digit] = res.rvalue**2
        # n2rms_f = (np.mean(ap_a_df) + np.mean(ap_m_df)) / 2
        # if n2rms_f < 0.01 * max_total_force:
        #     differences['normalized2 rms'][digit] = 0.
        # else:
        #     differences['normalized2 rms'][digit] = (
        #         differences['rms'][digit] / n2rms_f) * 100.

    # export
    trial.tv_differences = tv_differences
    trial.differences = differences
    trial.difference_units = difference_units


def compare_digit_forces(preset, sessions, trials_sel, temp, find_good, make_plots, find_good_n):
    """Compare manually-labeled to the automatically-labeled digit forces.

    Arguments:
        preset {dict} --- Preset holding the raw ('default_server') and processed
            ('processed_server') server locations.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        find_good {bool} --- Find good trials - candidates for labeling.
        make_plots {bool} --- Makes some inspection figures.
        find_good_n {bool} --- Default number of random good trials to select from a session.
    """
    rserv = preset['default_server']
    pserv = preset['processed_server']

    tools.logs.setup_logging(temp, sessions_dir=pserv)

    if not os.path.exists(rserv):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            rserv))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(rserv)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    trials_by_session = {}

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        raw_ss = os.path.join(rserv, session)
        proc_ss = os.path.join(pserv, session)

        if not os.path.exists(raw_ss):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                raw_ss, proc_ss, check_manual_log=True)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # accumulate data
        trials = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not find_good and (not trial.do_matched_contacts_files_exist() or
                                  not trial.does_manually_labelled_file_exists()):
                continue
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        if find_good:
            # print out a random subset of 20
            good_trials_select = sorted(random.sample(trials, find_good_n),
                                        key=lambda t: t.trial_number)
            rs('Selection of {} good trials: {}'.format(
                find_good_n, ', '.join([str(t.trial_number) for t in good_trials_select])))
            continue

        if len(trials) == 0:
            continue

        # get the force profiles
        for trial in tqdm.tqdm(trials, ncols=100, desc='Calculate digit forces'):
            calculate_digit_forces(mstruct, trial)

        # calculate summary statistics per trial
        for trial in tqdm.tqdm(trials, ncols=100, desc='Calculate differences'):
            calculate_differences(trial)
        difference_metrics = list(trials[0].differences.keys())

        # digits
        digit_names = SEGMENT_DIGIT_GROUPS.keys()
        yn_subplots = int(np.ceil(np.sqrt(len(digit_names))))
        xn_subplots = int(np.ceil(len(digit_names) / yn_subplots))
        figsize = (16, 9)

        # objects
        object_cmap = mpl.cm.get_cmap('gist_rainbow')

        def ocmap(object_id):
            return object_cmap(object_id / (len(mobject) - 1))

        # report on individual trials
        rs('Individual trial reports')
        lbls = ['trial number', 'object id'] + difference_metrics
        rs('|'.join(['{:20s}'.format(lbl) for lbl in lbls]))
        for trial in trials:
            v = ['{}'.format(lbl) for lbl in [trial.trial_number, trial.object_id]]
            values = [np.mean(list(trial.differences[dm].values())) for dm in difference_metrics]
            v += ['{:.4f}'.format(value) for value in values]
            rs('|'.join(['{:20s}'.format(lbl) for lbl in v]))

        # make plots
        for dm in difference_metrics:
            rs('{}, {}'.format(dm, trials[0].difference_units[dm]))
            rs('\t{:10}: {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
                'digit', 'mean', 'median', 'min', 'max', 'std'))
            if make_plots:
                fig = plt.figure(figsize=figsize)
            for i_digit, digit in enumerate(digit_names):
                if make_plots:
                    if i_digit:
                        plt.subplot(xn_subplots, yn_subplots, i_digit + 1, sharex=ax, sharey=ax)
                    else:
                        ax = plt.subplot(xn_subplots, yn_subplots, 1)

                tdd = [trial.differences[dm][digit] for trial in trials]

                # plot
                if make_plots:
                    plt.hist(tdd, color='k')
                    plt.title(digit)

                # report
                rs('\t{:10s}: {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
                    digit, np.mean(tdd), np.median(tdd), np.min(tdd), np.max(tdd), np.std(tdd)))
            tdd_total = sum([[trial.differences[dm][digit] for trial in trials]
                             for digit in digit_names], [])
            rs('\t{:10s}: {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
                'TOTAL', np.mean(tdd_total), np.median(tdd_total), np.min(tdd_total),
                np.max(tdd_total), np.std(tdd_total)))
            if make_plots:
                ax.set_xlabel('{}, {}'.format(dm, trials[0].difference_units[dm]))
                ax.set_ylabel('Number of trials')
                fig.suptitle('{}, session {}'.format(dm, session))

        # make time-varying plots
        if make_plots:
            for dm in trials[0].tv_differences.keys():
                fig = plt.figure(figsize=figsize)
                used_object_ids = set()
                for i_digit, digit in enumerate(digit_names):
                    if i_digit:
                        plt.subplot(xn_subplots, yn_subplots, i_digit + 1, sharex=ax, sharey=ax)
                    else:
                        ax = plt.subplot(xn_subplots, yn_subplots, 1)

                    # color by object id
                    for trial in trials:
                        plt.plot(trial.digit_forces_time, trial.tv_differences[dm][digit],
                                 color=ocmap(trial.object_id))
                        used_object_ids.add(trial.object_id)

                    plt.title(digit)

                ax.set_ylabel('{}, {}'.format(dm, trials[0].difference_units[dm]))
                ax.set_xlabel('Time, s')
                plt.sca(ax)
                for uoi in used_object_ids:
                    plt.plot(0, 0, color=ocmap(uoi), label='object type #{}'.format(uoi))
                plt.legend()
                fig.suptitle('{}, session {}'.format(dm, session))

        trials_by_session[session] = trials

    if 'difference_metrics' not in locals():
        ws('No sessions processed successfully.')
        return

    # report together for all sessions
    rs('Total {} trials.'.format(sum([len(trials) for trials in trials_by_session.values()])))
    rs('Individual trial reports')
    lbls = ['session', 'trial number', 'object id'] + difference_metrics
    rs('|'.join(['{:20s}'.format(lbl) for lbl in lbls]))
    for session, trials in trials_by_session.items():
        for trial in trials:
            v = ['{}'.format(lbl) for lbl in [session, trial.trial_number, trial.object_id]]
            values = [np.mean(list(trial.differences[dm].values())) for dm in difference_metrics]
            v += ['{:.4f}'.format(value) for value in values]
            rs('|'.join(['{:20s}'.format(lbl) for lbl in v]))
    # general metrics for all trials together
    for dm in difference_metrics:
        rs('{}, {}'.format(dm, trials[0].difference_units[dm]))
        rs('\t{:10}: {:>10} {:>10} {:>10} {:>10} {:>10}'.format(
            'digit', 'mean', 'median', 'min', 'max', 'std'))
        for i_digit, digit in enumerate(digit_names):
            tdd = sum([[trial.differences[dm][digit] for trial in trials]
                       for trials in trials_by_session.values()], [])
            rs('\t{:10s}: {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
                digit, np.mean(tdd), np.median(tdd), np.min(tdd), np.max(tdd), np.std(tdd)))
        tdd_total = sum([sum([[trial.differences[dm][digit] for trial in trials]
                              for trials in trials_by_session.values()], [])
                         for digit in digit_names], [])
        rs('\t{:10s}: {:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
            'TOTAL', np.mean(tdd_total), np.median(tdd_total), np.min(tdd_total),
            np.max(tdd_total), np.std(tdd_total)))

    # make a total figure
    fig = plt.figure(figsize=figsize)
    yn_subplots = int(np.ceil(np.sqrt(len(difference_metrics))))
    xn_subplots = int(np.ceil(len(difference_metrics) / yn_subplots))
    for idm, dm in enumerate(difference_metrics):
        ax = plt.subplot(xn_subplots, yn_subplots, idm + 1)
        data = sum([[np.mean(list(trial.differences[dm].values())) for trial in trials]
                    for trials in trials_by_session.values()], [])
        ax.hist(data, color='k', bins=30)
        tools.actual_vline(ax, np.median(data), color='r')
        rs('{} median {} {}.'.format(dm, np.median(data), trials[0].difference_units[dm]))
        ax.set_xlim(left=0)
        ax.set_xlabel('{}, {}'.format(dm, trials[0].difference_units[dm]))
        ax.set_ylabel('Number of trials')
