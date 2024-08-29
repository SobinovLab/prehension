#!python3.7
import argparse
import os
import time

import tqdm
from reporting_pool import ReportingPool

from prehension import meta_session
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws


# to be run in parallel
def match_trial(executable_filename, trial, model_filename, adjustment_filename, visualize,
                skip_export, write_video, quality_threshold, verbose):
    # unpack
    ja_filename = trial.post_kinematic_filename_csv
    post_ps_filenames = trial.get_post_ps_filenames()
    leps_in = post_ps_filenames['medial_sensor']
    reps_in = post_ps_filenames['lateral_sensor']
    leps_ou = trial.matched_contacts_filenames['medial_sensor']
    rips_ou = trial.matched_contacts_filenames['lateral_sensor']

    command = (
        '{executable_filename} -m "{model_filename}" {visualize}{skip_export}{quality_threshold}'
        '  '  # --vertical_thorax
        '--ja_in "{ja_filename}" '
        '--leps_in "{leps_in}" --rips_in "{reps_in}" '
        '--leps_ou "{leps_ou}" --rips_ou "{rips_ou}"{adjustment_arg}{video_arg}{verbose}'.format(
            executable_filename=executable_filename,
            model_filename=model_filename,
            ja_filename=ja_filename,
            leps_in=leps_in, reps_in=reps_in,
            leps_ou=leps_ou, rips_ou=rips_ou,
            adjustment_arg=(' --adj "{}"'.format(adjustment_filename)
                            if adjustment_filename is not None else ''),
            visualize='' if visualize else ' --no_visuals ',
            skip_export=' --skip_export ' if skip_export else '',
            quality_threshold=(' --quality_threshold {} '.format(quality_threshold)
                               if quality_threshold is not None and trial.success else ''),
            video_arg=(f' --write_video {trial.mujoco_video} ' if write_video else ''),
            verbose=' --verbose ' if verbose else ''))
    if verbose:
        rs('Executing command:')
        rs(command)
    ret = os.system(command)
    # process output as error throw
    if (ret < 0):
        raise ValueError('Command returned with {} error message.'.format(ret))


def main(server, sessions, trials_sel, temp, processes, overwrite,
         executable_filename, visualize, skip_export, write_video, quality_threshold, verbose):

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
            mstruct, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # load session's adjustment filename
        adjustment_trials = meta_session.import_adjustment_trials(server_session)

        # accumulate data
        trials = []
        adjustment_filenames = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_all_post_files_exist():
                continue
            if not overwrite and not skip_export and trial.do_matched_contacts_files_exist():
                continue
            # find adjustment filename
            adjustment_filename = None
            if trial.trial_number in adjustment_trials.keys():
                adjustment_trialnum = adjustment_trials[trial.trial_number]
                for t in msession:
                    if t.trial_number == adjustment_trialnum:
                        adjustment_filename = t.adjustment_kinematic_filename
                        break
                if adjustment_filename is None:
                    ws('Could not find adjustment trial #{} for trial #{}'.format(
                        trial.trial_number, adjustment_trialnum))
                    continue

            trials.append(trial)
            adjustment_filenames.append(adjustment_filename)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        os.makedirs(mstruct['matched_contacts_dir'], exist_ok=True)

        if write_video:
            os.makedirs(mstruct['mujoco_videos_dir'], exist_ok=True)

        p_args = list(zip(*[
            [executable_filename for _ in trials],
            trials,
            [mstruct['mujoco_model_sensorized'] for _ in trials],
            adjustment_filenames,
            [visualize]*len(trials),
            [skip_export]*len(trials),
            [write_video]*len(trials),
            [quality_threshold]*len(trials),
            [verbose]*len(trials)
        ]))

        if len(p_args) > 0:
            if visualize:
                for p_arg in p_args:
                    match_trial(*p_arg)
            else:
                pool = ReportingPool(match_trial, p_args, processes=processes,
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


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    executable_filename = os.path.join(
        '../../stereo_inverse_kinematics', 'mjc_vs_code',
        'MuJoCoInverseDynamics', 'x64', 'Debug', 'MuJoCoInverseDynamicsProject.exe')

    parser = argparse.ArgumentParser(
        description=('Automatically matches sensels with hand segments using MuJoCo program.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('sessions', 'trials', 'temp', 'processes', 'overwrite'))

    # other
    parser.add_argument(
        '--executable_filename',
        type=str, default=executable_filename,
        help='Filename of the executable MuJoCo file. Default: {}.'.format(executable_filename))
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize the grasping motion. Disables parallel execution.')
    parser.add_argument(
        '--skip_export',
        action='store_true',
        help='Does not export the results. Useful when just trying to visualize the trial,'
        ' instead of specifying --overwrite.')
    parser.add_argument(
        '--write_video',
        action='store_true',
        help='Write video during force matching simulation, when running.'
    )
    parser.add_argument(
        '--quality_threshold',
        type=float, default=0.1,
        help='If the unmatched force exceeds this portion of total force, the trial will throw'
        ' an error. Useful to detect when the model is actually breaking bad.')
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose prints about program running.')

    args = parser.parse_args(args=argv)

    if (not args.visualize and args.write_video):
        raise Exception("Cannot specify --write_video without --visualize")

    start_time = time.time()
    main(args.server, args.sessions, args.trials, args.temp, args.processes,
         args.overwrite,
         args.executable_filename,
         args.visualize,
         args.skip_export,
         args.write_video,
         args.quality_threshold,
         args.verbose)

    rs('Program took {} s.'.format(time.time() - start_time))
