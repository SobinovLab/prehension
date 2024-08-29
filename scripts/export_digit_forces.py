#!python3.7
import os
import argparse
import copy
import time
import datetime
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import ncams
from reporting_pool import ReportingPool

# New version: use prehension library
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import io_tools
from prehension import meta_session


DIGITS = tools.DIGITS
SEGMENTS = tools.SEGMENTS
UNCLAIMED_NAME = tools.UNCLAIMED_NAME
UNCLAIMED_INDEX = tools.UNCLAIMED_INDEX


def calculate_force_traces(
        trial, aligned_times, filtered_times, filtered_matrices, matched_contacts, groups):
    # find the closest timepoints in filtered times to aligned times
    aligned_timepoints = []
    atp = 0
    for ift, ft in enumerate(filtered_times):
        while (atp+1 < len(aligned_times) and
               abs(ft - aligned_times[atp+1]) < abs(ft - aligned_times[atp])):
            atp += 1
        aligned_timepoints.append(atp)

    # fill the data
    shp = np.shape(filtered_matrices[list(filtered_matrices.keys())[0]][0])
    names = []
    total_auto_force_traces = []
    for n, d in groups.items():
        if n == UNCLAIMED_NAME:
            continue
        names.append(n)

        auto_force_traces = []
        for itp, atp in enumerate(aligned_timepoints):
            af = 0
            for ps_name in filtered_matrices.keys():
                af += np.sum(
                    filtered_matrices[ps_name][itp],
                    where=tools.get_matched_contact_frame_mask(
                        d['exp'], matched_contacts[ps_name][atp], shp))
            auto_force_traces.append(af)
        total_auto_force_traces.append(auto_force_traces)

    return names, np.array(total_auto_force_traces)


def export_forces(trial, mstruct):
    filtered_times = None
    filtered_matrices = {}
    matched_contacts = {}
    segments_set = set()
    for ps_name in mstruct['ps_dic'].keys():
        aligned_times, _ = io_tools.import_matrices(
            trial.aligned_ps_filenames[ps_name])
        filtered_times_local, filtered_matrices[ps_name] = io_tools.import_matrices(
            trial.filtered_ps_filenames[ps_name])

        if filtered_times is None:
            filtered_times = filtered_times_local
        else:
            if any(filtered_times != filtered_times_local):
                if len(filtered_times) == len(filtered_times_local):
                    # fix the misaligned timepoints
                    misaligned_is = (filtered_times != filtered_times_local).nonzero()[0]
                    for mi in misaligned_is:
                        # figure out which one needs to be fixed
                        # choose the one that is closer to even rate:
                        if mi < 1:
                            # edge cases
                            ft_rate_dev = abs(
                                2 * filtered_times[mi+1] -
                                filtered_times[mi] - filtered_times[mi+2])
                            ftl_rate_dev = abs(
                                2 * filtered_times_local[mi+1] -
                                filtered_times_local[mi] - filtered_times_local[mi+2])
                        elif mi + 1 == len(filtered_times):
                            # edge cases
                            ft_rate_dev = abs(
                                2 * filtered_times[mi-1] -
                                filtered_times[mi-2] - filtered_times[mi])
                            ftl_rate_dev = abs(
                                2 * filtered_times_local[mi-1] -
                                filtered_times_local[mi-2] - filtered_times_local[mi])
                        else:
                            ft_rate_dev = abs(
                                2 * filtered_times[mi] -
                                filtered_times[mi-1] - filtered_times[mi+1])
                            ftl_rate_dev = abs(
                                2 * filtered_times_local[mi] -
                                filtered_times_local[mi-1] - filtered_times_local[mi+1])
                        if ft_rate_dev < ftl_rate_dev:
                            # missing/misaligned packet from filtered_times_local
                            filtered_times_local[mi] = filtered_times[mi]
                            # print(np.sum(np.abs(filtered_matrices[ps_name][mi] -
                            #                     filtered_matrices[ps_name][mi-1])))
                        else:
                            filtered_times[mi] = filtered_times_local[mi]
                    # the time points are pretty close, so no need to interpolate the data itself
                    # current is basically a nearest-neighbor

                    # some eval information
                    # print(len(filtered_times), len(filtered_times_local))
                    # print(all(filtered_times == filtered_times_local))
                    # n = min(len(filtered_times), len(filtered_times_local))

                    # plt.figure()
                    # diff = filtered_times[:n] - filtered_times_local[:n]
                    # print(max(abs(diff)))
                    # plt.plot(filtered_times[:n], diff, 'k')
                    # plt.xlabel('LPS time stamps')
                    # plt.ylabel('DIFF in time stamps')
                    # plt.show()

                else:
                    raise ValueError(
                        'Length of filtered time stamps does not match between sensors.')

        matched_contacts[ps_name] = io_tools.import_matched_contacts(
            trial.matched_contacts_filenames[ps_name])

    # DIGITS
    names, total_auto_force_traces = calculate_force_traces(
        trial, aligned_times, filtered_times, filtered_matrices, matched_contacts, DIGITS)

    ncams.io_utils.export_csv(trial.digit_forces_filename,
                              ['time'] + names, [filtered_times] + total_auto_force_traces.tolist())

    # plt.figure()
    # for name, taft in zip(names, total_auto_force_traces):
    #     plt.plot(filtered_times, taft, label=name)
    # plt.xlabel('Time, s')
    # plt.ylabel('Forces, N')
    # plt.legend()
    # plt.show()

    # SEGMENTS
    names, total_auto_force_traces = calculate_force_traces(
        trial, aligned_times, filtered_times, filtered_matrices, matched_contacts, SEGMENTS)

    ncams.io_utils.export_csv(trial.segment_forces_filename,
                              ['time'] + names, [filtered_times] + total_auto_force_traces.tolist())


def main(server, sessions, trials_sel, temp, overwrite, processes):
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
            if (not trial.do_matched_contacts_files_exist() or not trial.do_pre_ps_files_exist() or
                    not trial.do_post_ps_files_exist()):
                continue
            if (trial.does_digit_force_file_exist() and trial.does_segment_force_file_exist() and
                    not overwrite):
                continue
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        if len(trials) == 0:
            continue

        os.makedirs(mstruct['digit_forces_dir'], exist_ok=True)
        os.makedirs(mstruct['segment_forces_dir'], exist_ok=True)

        p_args = list(zip(*[
            trials,
            [copy.deepcopy(mstruct) for _ in trials],
        ]))

        # export_forces(*(p_args[0]))
        # sys.exit()

        pool = ReportingPool(export_forces, p_args, processes=processes,
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
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Compare manually-labeled to the automatically-labeled forces using sensor'
                     ' masks.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'overwrite', 'processes'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp, args.overwrite, args.processes)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
