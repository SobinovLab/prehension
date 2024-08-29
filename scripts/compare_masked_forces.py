#!python3.7
import os
import argparse
import time
import datetime
import random
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import io_tools
from prehension import meta_session



DIGITS = tools.DIGITS
UNCLAIMED_NAME = tools.UNCLAIMED_NAME
UNCLAIMED_INDEX = tools.UNCLAIMED_INDEX


def load_maps(trial):
    if os.path.exists(trial.lps_map_filename):
        lps_digit_mask = np.array(
            io_tools.import_one_csv_matrix(trial.lps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    if os.path.exists(trial.rps_map_filename):
        rps_digit_mask = np.array(
            io_tools.import_one_csv_matrix(trial.rps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    return lps_digit_mask, rps_digit_mask


def load_forces(mstruct, trial):
    ps_matrices = {}
    matched_contacts = {}
    segments_set = set()
    for ps_name in mstruct['ps_dic'].keys():
        ps_times, ps_matrices[ps_name] = io_tools.import_matrices(
            trial.get_post_ps_filenames()[ps_name])
        matched_contacts[ps_name] = io_tools.import_matched_contacts(
            trial.matched_contacts_filenames[ps_name])

    # some basic force parameters
    dts = np.diff(ps_times)
    trial.dt = np.median(dts)
    # for varied dts, for each time point:
    trial.dts = (np.concatenate(([0], dts)) + np.concatenate((dts, [0]))) / 2
    trial.total_force = (np.sum(ps_matrices['medial_sensor'], axis=(1, 2)) +
                         np.sum(ps_matrices['lateral_sensor'], axis=(1, 2)))
    trial.max_total_force = np.max(trial.total_force)
    trial.summed_force = np.sum(trial.total_force)
    trial.summed_impulse = np.sum(trial.total_force * trial.dts)
    trial.active_period_flag = trial.total_force >= (0.05 * trial.max_total_force)

    # export
    trial.ps_times = ps_times
    trial.ps_matrices = ps_matrices
    trial.matched_contacts = matched_contacts

    # pool all distances
    trial.pooled_distances = []
    for mc_fl, mc_fr in zip(matched_contacts['medial_sensor'], matched_contacts['lateral_sensor']):
        # print(sum([mc for mc in mc_fl.values()], []))
        # print(sum([mc for mc in mc_fr.values()], []))
        trial.pooled_distances.append([mc[2] for mc in sum([mc for mc in mc_fl.values()], [])] +
                                      [mc[2] for mc in sum([mc for mc in mc_fr.values()], [])])

    # load maps
    manual_digit_maps = {}
    # lps_digit_mask, rps_digit_mask  # rigidly set for ps_names
    (manual_digit_maps['medial_sensor'],
     manual_digit_maps['lateral_sensor']) = load_maps(trial)

    # find mask-based difference between manual and automatic labels
    mask_based_diff_per_sensor = {}
    unclaimed_force = {}
    for ps_name in mstruct['ps_dic'].keys():
        manual_digit_map = manual_digit_maps[ps_name]
        mask_based_diff_per_sensor[ps_name] = []
        unclaimed_force[ps_name] = []
        for i_frame in range(len(ps_times)):
            # build auto mask
            auto_mask = (len(DIGITS) - 1) * np.ones(np.shape(manual_digit_map))
            for i_digit, d in enumerate(DIGITS.values()):
                if i_digit == len(DIGITS) - 1:
                    break
                digit_auto_mask = tools.get_matched_contact_frame_mask(
                    d['exp'], matched_contacts[ps_name][i_frame],
                    np.shape(manual_digit_map))
                auto_mask[digit_auto_mask] = i_digit

            # diff mask
            diff_mask = np.not_equal(auto_mask, manual_digit_map)
            ps_matrix_frame = ps_matrices[ps_name][i_frame]
            mask_based_diff_per_sensor[ps_name].append(np.sum(np.abs(ps_matrix_frame[diff_mask])))

            # manual unclaimed mask
            unclaimed_mask = manual_digit_map == (len(DIGITS) - 1)
            unclaimed_force[ps_name].append(np.sum(np.abs(ps_matrix_frame[unclaimed_mask])))

        mask_based_diff_per_sensor[ps_name] = np.array(mask_based_diff_per_sensor[ps_name])
        unclaimed_force[ps_name] = np.array(unclaimed_force[ps_name])
    # sum across sensors
    trial.mask_based_diff = (mask_based_diff_per_sensor['medial_sensor'] +
                             mask_based_diff_per_sensor['lateral_sensor'])
    trial.unclaimed_force = (unclaimed_force['medial_sensor'] +
                             unclaimed_force['lateral_sensor'])


TV_DIFF_F = {
    # 'misattributed force': {
    #     'f': lambda trial: trial.mask_based_diff,
    #     'unit': 'N'
    # },
    'distances': {
        'f': lambda trial: trial.pooled_distances,
        'unit': 'm'
    }
}
DIFF_F = {
    'misattributed impulse': {
        'f': lambda trial: np.sum(trial.mask_based_diff * trial.dts),
        'unit': 'N s'
    },
    'normalized misattributed impulse': {
        # 'f': lambda trial: np.sum(trial.mask_based_diff * trial.dts) / trial.summed_impulse * 100,
        # equivalent to the following, assuming dt is constant
        'f': lambda trial: np.sum(trial.mask_based_diff) / trial.summed_force * 100,
        'unit': '%'
    },
    'unclaimed impulse': {
        'f': lambda trial: np.sum(trial.unclaimed_force * trial.dts),
        'unit': 'N s'
    },
    'normalized unclaimed impulse': {
        # 'f': lambda trial: np.sum(trial.unclaimed_force * trial.dts) / trial.summed_impulse * 100,
        # equivalent to the following, assuming dt is constant
        'f': lambda trial: np.sum(trial.unclaimed_force) / trial.summed_force * 100,
        'unit': '%'
    }
}


def calculate_differences(trial):
    trial.tv_differences = {}  # time-varying
    trial.differences = {}

    for k, v in TV_DIFF_F.items():
        trial.tv_differences[k] = v['f'](trial)

    for k, v in DIFF_F.items():
        trial.differences[k] = v['f'](trial)


def main(server, sessions, trials_sel, temp, find_good, make_plots, find_good_n):

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

    trials_by_session = {}
    difference_metrics = list(DIFF_F.keys())
    tv_difference_metrics = list(TV_DIFF_F.keys())

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                server_session, check_manual_log=True)
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
        for trial in tqdm.tqdm(trials, ncols=100, desc='Load forces'):
            load_forces(mstruct, trial)

        # calculate summary statistics per trial
        for trial in tqdm.tqdm(trials, ncols=100, desc='Calculate differences'):
            calculate_differences(trial)

        # # objects
        # object_cmap = mpl.cm.get_cmap('gist_rainbow')

        # def ocmap(object_id):
        #     return object_cmap(object_id / (len(mobject) - 1))

        # report on individual trials
        rs('Individual trial reports')
        lbls = ['trial number', 'object id'] + difference_metrics
        rs('|'.join(['{:20s}'.format(lbl) for lbl in lbls]))
        for trial in trials:
            v = ['{}'.format(lbl) for lbl in [trial.trial_number, trial.object_id]]
            values = [trial.differences[dm] for dm in difference_metrics]
            v += ['{:.4f}'.format(value) for value in values]
            rs('|'.join(['{:20s}'.format(lbl) for lbl in v]))

        trials_by_session[session] = trials

    if len(trials_by_session) == 0:
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
            values = [trial.differences[dm] for dm in difference_metrics]
            v += ['{:.4f}'.format(value) for value in values]
            rs('|'.join(['{:20s}'.format(lbl) for lbl in v]))
    # general metrics for all trials together
    for dm, dmv in DIFF_F.items():
        rs('{}, {}'.format(dm, dmv['unit']))
        rs('\t{:10} {:>10} {:>10} {:>10} {:>10}'.format(
            'mean', 'median', 'min', 'max', 'std'))
        tdd = sum([[trial.differences[dm] for trial in trials]
                   for trials in trials_by_session.values()], [])
        rs('\t{:10.3f} {:10.3f} {:10.3f} {:10.3f} {:10.3f}'.format(
            np.mean(tdd), np.median(tdd), np.min(tdd), np.max(tdd), np.std(tdd)))

    # make a total figure
    figsize = (16, 9)
    fig = plt.figure(figsize=figsize)
    yn_subplots = int(np.ceil(np.sqrt(len(difference_metrics))))
    xn_subplots = int(np.ceil(len(difference_metrics) / yn_subplots))
    for idm, (dm, dmv) in enumerate(DIFF_F.items()):
        ax = plt.subplot(xn_subplots, yn_subplots, idm + 1)
        data = sum([[trial.differences[dm] for trial in trials]
                    for trials in trials_by_session.values()], [])
        ax.hist(data, color='k', bins=30)
        tools.actual_vline(ax, np.median(data), color='r')
        rs('{} median {} {}.'.format(dm, np.median(data), dmv['unit']))
        ax.set_xlim(left=0)
        ax.set_xlabel('{}, {}'.format(dm, dmv['unit']))
        ax.set_ylabel('Number of trials')

    # make a figure by pooling all time varying points
    fig = plt.figure(figsize=figsize)
    yn_subplots = int(np.ceil(np.sqrt(len(tv_difference_metrics))))
    xn_subplots = int(np.ceil(len(tv_difference_metrics) / yn_subplots))
    for idm, (dm, dmv) in enumerate(TV_DIFF_F.items()):
        ax = plt.subplot(xn_subplots, yn_subplots, idm + 1)
        data = []
        for trials in trials_by_session.values():
            for trial in trials:
                data += sum(trial.tv_differences[dm], [])
        ax.hist(data, color='k', bins=100)
        tools.actual_vline(ax, np.median(data), color='r')
        rs('{} median {} {}.'.format(dm, np.median(data), dmv['unit']))
        ax.set_xlim(left=0)
        ax.set_xlabel('{}, {}'.format(dm, dmv['unit']))
        ax.set_ylabel('Data points')


if __name__ == '__main__':

    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Compare manually-labeled to the automatically-labeled forces using sensor'
                     ' masks.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'make_plots'))

    parser.add_argument(
        '--find_good',
        action='store_true',
        help='Find good trials - candidates for labeling.')
    parser.add_argument(
        '--find_good_n',
        type=int, default=20,
        help='Default number of random good trials to select from a session.')

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp, args.find_good,
         args.make_plots, args.find_good_n)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
