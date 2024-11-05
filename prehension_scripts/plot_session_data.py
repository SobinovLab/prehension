#!python3
# -*- coding: utf-8 -*-
"""
Create plots for many sessions.

Copyright (C) 2019-2024 Anton Sobinov, Caleb Raman
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


import argparse
import time
import os
import sys
import matplotlib as mpl

from tqdm import tqdm
from datetime import timedelta
from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from reporting_pool import ReportingPool
from matplotlib.colors import LinearSegmentedColormap

#from prehension import tools, meta_session
from prehension_presets.prehension_presets import PRESETS
from prehension.tools.logs import rs, setup_logging
from prehension.tools import cmd_args
from prehension.visualization.session_data_visualization import SessionWrapper, SessionGroup
from prehension.tools.session_management import (
    does_trianing_servers_exist,
    fetch_server_session_dirs,
)

# --- future features ---
# Plot performance should consider all sessions with red dot for experimental sessions
# Add slight x offset for training on same data
# Add transfer
# Add overwrite


def main(preset, sessions, temp, overwrite, transfer=True):
    """Does the following steps:
    1. Moves sessions from experiment to training if they don't have both cam and sensor data
    2. Groups these sessions into experimental sessions and training sessions
    3. For each group make the following plots
        3.1. Force traces
        3.2. Condition success x2(this session and all previous sessions in same group)
        3.3 Performance x2 (all time and last 10 days)

    Args:
        preset (dict): the preset in question
        sessions (list[str]): the sessions to process, if empty, process all
        temp (dir): the temporary directory for logging, usually C:\tmp
        overwrite (bool): if true, overwrite existing files
        transfer (bool, optional): if true, transfer sessions from experiment to training.
            This can be time consuming.Defaults to True.

    Raises:
        ValueError: If either the raw or processed servers do not exist
    """

def create_heatmap_from_trials_list(tr_list, cmap, max_discrete_conds=9, bin_width_N=1,
                                    savename=None):

    """Create heatmap for conditional success matrix based on a list of trials

    Parameters
    ----------
    tr_list : list
        List of Trial objects
    max_discrete_conds : int
        Maximum number of discrete conditions to plot

    Returns
    -------
    None
    """

    # List all force targets
    discrete_force_targets = sorted(set([tr.target_force for tr in tr_list]))

    # All unique session names
    session_names = sorted(set([tr.session for tr in tr_list]))

    if len(session_names) > 1:
        title = f'Avg Condition Success Matrices'
        title += f'\n{len(session_names)} sessions ({session_names[0]} - {session_names[-1]})'
        title += f'\n{len(tr_list)} trials'
    else:
        title = f'Condition Success Matrices'
        title += f'\nsession: {session_names[0]}'
        title += f'\n{len(tr_list)} trials'

    if len(discrete_force_targets) == 0:
        ws("No force targets found returning.")
        return

    # Decide if we should bin the force targets or use the exact values from each trial
    use_force_bins = len(discrete_force_targets) > max_discrete_conds
    if use_force_bins:
        rounded_min_cond = int(math.floor(min(discrete_force_targets)))
        rounded_max_cond = int(math.ceil(max(discrete_force_targets)))

        force_bins = []
        for lo_bound in range(rounded_min_cond,
                            rounded_max_cond,
                            bin_width_N):
            force_bins.append((lo_bound, lo_bound + bin_width_N))
    else:
        force_bins = discrete_force_targets

    # Determine the optimal grid layout based on the number of force targets
    num_rows = int(np.ceil(np.sqrt(len(force_bins))))
    num_cols = int(np.ceil(len(force_bins) / num_rows))

    # Create a grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle(title)

    # Flatten the axes array if necessary
    if num_rows * num_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs,]

    def _make_subplot(ax, sub_trial_list, use_force_bins, force_target):
        # Sub trial list contains only trials with the same force target

        # Get unique rotations and apertures
        unique_rots = sorted(set(tr.rotation for tr in sub_trial_list))
        unique_aps = sorted(set(tr.aperture for tr in sub_trial_list))

        # Create a heatmap matrix
        hm = []
        for i, rot in enumerate(unique_rots):
            row = []
            for j, ap in enumerate(unique_aps):

                trials_rot_ap_match = [tr for tr in sub_trial_list
                                       if tr.aperture == ap and tr.rotation == rot]
                if trials_rot_ap_match:
                    pct = sum([tr.success for tr in trials_rot_ap_match]) / len(trials_rot_ap_match)
                    row.append(pct)
                    color = 'white' if pct >= 0.5 else 'black'
                    txt = f'{pct:.2f}'
                    txt += f"\n(n={len(trials_rot_ap_match)})"
                    ax.text(j, i, txt, ha="center", va="center", color=color)
                else:
                    # No data found for the current combination
                    row.append(0)
            hm.append(row)

            # Plot the heatmap
        ax.imshow(hm, cmap=cmap, vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(unique_aps)))
        ax.set_xticklabels(unique_aps)
        ax.set_yticks(np.arange(len(unique_rots)))
        ax.set_yticklabels(unique_rots)
        title = (f"Force Range: {force_target[0]}-{force_target[1]} N" if use_force_bins
                  else f"Force: {force_target} N")
        ax.set_title(title)
        ax.set_ylabel('Aperture (mm)')
        ax.set_xlabel('Rotation (deg)')


    # Loop th over force bins
    for plot_idx, target_force in enumerate(force_bins):

        # Get the trials that either fall in the force range or are the exact target force
        if use_force_bins:
            force_range = (target_force[0], target_force[1])
            cond_trials = [tr for tr in tr_list if
                           tr.target_force >= force_range[0] and tr.target_force <= force_range[1]]
        else:
            cond_trials = [tr for tr in tr_list if tr.target_force == target_force]

        # make the subplot for each force condition
        _make_subplot(axs[plot_idx], cond_trials, use_force_bins, target_force)

    # Hide any unused subplots because we make a grid
    for i in range(len(force_bins), num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savename is not None:
        fig.savefig(savename)
        #rs(f'saving success matrix as {os.path.normpath(savename)}')
    else:
        plt.show()
    plt.close(fig)


class SessionGroup:

    """
    Wrapper for multiple session data and analysis,
    Includes methods for plotting the following
    - performance per session plots across time
    - conditional success matrix for all sessions
    """


    def __init__(self, l_session_wrappers, group_label=''):

        # Filter out sessions with no meta
        self.session_wrappers = [sw for sw in l_session_wrappers if sw.has_meta]

        # Sort by date and then set number ascending
        self.session_wrappers.sort(key=lambda x: (x.datetime, x.set_number))

        # Group label for tqdm later
        self.group_label = group_label

    def do_all_analysis(self, sessions=[]):
        self.do_single_sess_analysis(sessions=sessions)
        self.do_group_analysis(sessions=sessions)

    def do_single_sess_analysis(self, sessions=[]):
        # Experiment single sess analysis
        if sessions:
            to_process = [sw for sw in self.session_wrappers if sw.sess_name in sessions]
        else:
            to_process = self.session_wrappers
        for sw in tqdm.tqdm(to_process, 'Plotting Single Session Analysis'):
            if not SKIP_FORCE_TRACES:
                sw.plot_force_trace()
            sw.plot_cond_success_matrix()

    def do_group_analysis(self, sessions=[]):
        self._plot_avg_cond_success_matricies(sessions)
        self._plot_performance()

    def _plot_avg_cond_success_matricies(self, sessions=[]):

        # First sort all of the session wrappers by ascending datetime
        all_trials = []

        for sw in tqdm.tqdm(self.session_wrappers, desc='Plotting Average Conditional Success MTX'):
            all_trials += sw.msession ## Extend list of all trials

            if sessions:
                if sw.sess_name in sessions:
                    create_heatmap_from_trials_list(
                        all_trials,
                        'Greens',
                        savename=os.path.join(sw.results_dir, 'AvgCondSuccessMatrix.png')
                    )
            else:
                create_heatmap_from_trials_list(
                        all_trials,
                        'Greens',
                        savename=os.path.join(sw.results_dir, 'AvgCondSuccessMatrix.png')
                    )


    @staticmethod
    def plot_performace_fig(date_success_dict,
                            last_n_days_label=None, include_last_tick=False,
                            annotate_dates=False, savename=None):

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        if last_n_days_label:
            suptitle = f'Training Progress (last {last_n_days_label} days)'
        else:
            suptitle = 'Training Progress'
        fig.suptitle(suptitle)

        # Plot for Performance
        axs[0].set_title('Performance')
        axs[0].set_ylabel('Percent (%)')
        axs[0].set_ylim((0, 1))

        dates = sorted(date_success_dict.keys())
        pct_correct = [sum(date_success_dict[dt]) / len(date_success_dict[dt]) for dt in dates]
        total_trials = [len(date_success_dict[dt]) for dt in dates]
        num_correct = [sum(date_success_dict[dt]) for dt in dates]
        num_incorrect = [len(date_success_dict[dt]) - sum(date_success_dict[dt]) for dt in dates]

        axs[0].plot(dates, pct_correct, color='green', label='correct')
        axs[0].tick_params(axis='x', labelrotation=45)
        axs[0].legend()

        # Plot for Trial count
        axs[1].set_title('Trial count')
        axs[1].set_ylabel('Trials')
        axs[1].plot(dates, total_trials, color='black', label='total trials')
        axs[1].plot(dates, num_correct, color='green', label='num correct')
        axs[1].plot(dates, num_incorrect, color='orange', label='num incorrect')
        axs[1].tick_params(axis='x', labelrotation=45)
        axs[1].set_ylim(bottom=0)
        axs[1].legend()

        # -- DATE TICKS -- #
        desired_ticks = 5
        dt_range = max(dates) - min(dates)
        interval_options = [timedelta(days=1), timedelta(weeks=1), timedelta(weeks=2),
                            timedelta(weeks=4), timedelta(weeks=8), timedelta(weeks=12),
                            timedelta(weeks=26), timedelta(weeks=52)]

        interval_counts = [abs(desired_ticks - (dt_range / inter)) for inter in interval_options]
        i = np.argmin(np.array(interval_counts))
        interval = interval_options[i]

        date_ticks = [min(dates), ]
        while date_ticks[-1] < max(dates):
            date_ticks.append(date_ticks[-1] + interval)
        if not include_last_tick:
            date_ticks = date_ticks[:-1]

        for ax in axs:
            for date in date_ticks:
                ax.axvline(x=date, color='gray', linestyle='--', alpha=0.5)
            ax.set_xticks(date_ticks)
            ax.set_xticklabels([date.strftime('%y-%m-%d') for date in date_ticks], rotation=45)

        # -- LABEL POINTS IF LOOKBACK IS <= 2 WEEKS -- #
        if annotate_dates:
            for i, date in enumerate(dates):
                # Label each point on both subplots with the date (excluding time)
                label = date.strftime('%m/%d')

                axs[0].annotate(label, (dates[i], pct_correct[i]),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', rotation='vertical')

                axs[1].annotate(label, (dates[i], total_trials[i]),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', rotation='vertical')

        if savename is not None:
            fig.savefig(savename)
            #rs(f'saving performance as {os.path.normpath(savename)}')
        else:
            plt.show()

        plt.close(fig)

    def _plot_performance(self):

        # dictionary of form -> datetime: list[SessionWrapper]
        session_wrappers_by_date = {}
        for sw in self.session_wrappers:
            if sw.datetime not in session_wrappers_by_date:
                session_wrappers_by_date[sw.datetime] = [sw]
            else:
                ws(f'Duplicate session for {sw.datetime}')
                session_wrappers_by_date[sw.datetime].append(sw)

        # Build dictionary of form -> datetime: list[trial success bools for that date]
        # Note we need to aggregate all sessions for the same date

        date_l_trial_success_dict = {}

        for datetime in session_wrappers_by_date:
            trial_success_list_per_date = []
            for sw in session_wrappers_by_date[datetime]:
                trial_success_list_per_date += [tr.success for tr in sw.msession]

            date_l_trial_success_dict[datetime] = trial_success_list_per_date

        # Now loop through session wrappers and plot
        for sw in tqdm.tqdm(self.session_wrappers, desc='Plotting Performance'):
            # Find valid subset of keys within range for each session wrapper
            dates_total = [dt for dt in list(date_l_trial_success_dict.keys()) if dt <= sw.datetime]
            dates_10_day = [dt for dt in dates_total if dt >= sw.datetime - timedelta(days=10)]

            date_success_dict_total = {dt: date_l_trial_success_dict[dt] for dt in dates_total}
            date_success_dict_10_day = {dt: date_l_trial_success_dict[dt] for dt in dates_10_day}

            SessionGroup.plot_performace_fig(date_success_dict_total,
                                             savename=os.path.join(sw.results_dir,
                                                                   'Performance.png'))
            SessionGroup.plot_performace_fig(date_success_dict_10_day,
                                             last_n_days_label='10',
                                             include_last_tick=True,
                                             annotate_dates=True,
                                             savename=os.path.join(sw.results_dir,
                                                                   'PerformanceLast10Days.png'))



class SessionWrapper:

    """
    Wrapper for single session data and analysis,
    Includes methods for plotting the following
    - force traces for session
    - conditional success matrix for session
    """

    def __init__(self, raw_ss, proc_ss):

        self.raw_ss = raw_ss
        self.proc_ss = proc_ss

        self.sess_name = os.path.basename(self.raw_ss)
        self.datetime, self.set_number = SessionWrapper.date_from_folder(self.sess_name)

        # Create results and log folder
        self.results_dir = os.path.join(self.proc_ss, 'prehension_plots')
        os.makedirs(self.results_dir, exist_ok=True)
        self.has_meta = False

        potential_logs = glob.glob(os.path.join(self.raw_ss, 'behavior', '*.csv'))
        if len(potential_logs) > 1:
            ws(f'Found more than one log for session: {os.path.basename(self.raw_ss)}, skipping')
        if len(potential_logs) == 1:
            self.log_full = potential_logs[0]
        else:
            self.log_full = None

        # Load meta
        try:
            mres = meta_session.load_meta_information(self.raw_ss, self.proc_ss)
            self.mstruct, self.mdof, self.mobject, self.msession = mres
            self.has_meta = True
            if len(self.mstruct['auto_log']) > 0:
                self.log_full = self.mstruct['auto_log'][0]
        except meta_session.IncompleteMetaError as _:
            #ws(f'Incomplete meta information for session: {os.path.basename(self.raw_ss)}, skipping')
            return
        ## Other errors should stop the program -- this is intentional right now for testing

        # Bind conditional data to each trial within msession
        self.attach_fields_to_trials()


    def ensure_transfer_to_training_server(self, raw_training_server,
                                           proc_training_server, overwrite=False):

        if not self.is_training_session:
            return

        if not self.has_meta:
            return


        def _folder_contents_set(d1, d2):
            """return src/dst folder contents as sets"""

            g1 = glob.glob(os.path.join(d1, '**'), recursive=True)
            contents_rel_1 = set([os.path.relpath(path, d1) for path in g1])

            g2 = glob.glob(os.path.join(d2, '**'), recursive=True)
            contents_rel_2 = set([os.path.relpath(path, d2) for path in g2])

            return contents_rel_1, contents_rel_2


        def _move_helper(src_session_dir, dst_parent_dir):

            # inputs:
            # [src_session_dir] - raw_(experimental)_server_session: the directory to be transferred
            # [dst_parent_dir] - training server destination directory (not a session dir but a
            # dir containing sessions)
            # [overwrite] - bool to determine if we are overwriting

            # defined vars:
            # [dst_session_dir] - raw_(training)_server_session: the expected directory
            # name after transfer
            dst_session_dir = os.path.join(dst_parent_dir, os.path.basename(src_session_dir))

            # logic:
            src_exists = os.path.exists(src_session_dir)
            if not src_exists:             # This should always exist
                raise Exception(f'Source session directory {src_session_dir} does not exist')

            # Check if dst_session_dir exists
            already_transferred = False
            if os.path.exists(dst_session_dir):
                src_contents, dst_contents = _folder_contents_set(src_session_dir, dst_session_dir)
                if src_contents.issubset(dst_contents):
                    already_transferred = True

            # To keep track of transfer status
            transferred = False

            if not already_transferred or overwrite:
                #rs('transferring')
                # copy src to dst_parent_dir but remove dst first
                if os.path.exists(dst_session_dir):
                    try:
                        shutil.rmtree(dst_session_dir, ignore_errors=False)
                    except:
                        ws(f'failed to remove {dst_session_dir}')

                try:
                    shutil.move(src_session_dir, dst_parent_dir)
                    transferred = True
                except:
                    ws(f'failed to transfer {src_session_dir}')

            # Integrity check if we did a transfer
            if transferred:
                # Do integrity check and remove src if successful
                src_contents, dst_contents = _folder_contents_set(src_session_dir, dst_session_dir)
                assert src_contents.issubset(dst_contents)

            # now remove src locally because we should have already transferred
            if os.path.exists(src_session_dir):
                msg = 'Expected tranferred dir {} not found'.format(dst_session_dir)
                assert os.path.exists(dst_session_dir), msg
                try:
                    shutil.rmtree(src_session_dir, ignore_errors=False)
                except:
                    ws(f'failed to remove {src_session_dir}')

        # Transfer from experiment to training server
        _move_helper(self.raw_ss, raw_training_server)
        _move_helper(self.proc_ss, proc_training_server)


    @property
    def is_training_session(self):
        if not self.has_meta:
            return False
        # Check if it is an experiment or not
        experiment_expected_dirnames = [
            os.path.join(self.raw_ss, self.mstruct['videos_dir']),
            os.path.join(self.raw_ss, self.mstruct['raw_ps_dir'])
        ]
        is_training_session = not all([os.path.exists(dname) for dname
                                        in experiment_expected_dirnames])
        return is_training_session


    def attach_fields_to_trials(self):

        """Bind the following to each trial object in self.msession
            - target force
            - range delta (if exists)
            - rotation
            - aperture
        """

        for trial in self.msession:
            #trial.target_condition = target_condition
            stub = self.mobject[trial.object_id]['def']
            trial.target_force = float(stub['targetForce(N)'])
            bound_keys = ['targetForceRelRangeMin(N)', 'targetForceRelRangeMax(N)']
            # Check if keys exist in the stub
            trial.range_delta = None
            trial.target_range = None
            if all([k in stub.keys() for k in bound_keys]):
                trial.range_delta = tuple([float(stub[bk]) for bk in bound_keys])
                trial.target_range = tuple([trial.target_force + delta
                                            for delta in trial.range_delta])

            # Add aperture and rotation information
            trial.rotation = float(stub['pos_aperture(mm)'])
            trial.aperture = float(stub['pos_tilt(deg)'])

    @staticmethod
    def date_from_folder(folder):
        """Helper function to get the date from the folder name and the set number"""
        # Regex to capture the date part (YYYY_MM_DD) and optional SetN part
        match = re.match(r'(\d{4}_\d{2}_\d{2})(?:_Set(\d+))?', folder)

        if match:
            # Extract the date string and convert to datetime
            date_str = match.group(1)
            current_date = datetime.strptime(date_str, r'%Y_%m_%d')

            # Extract the set number if available, otherwise return -1
            set_number = int(match.group(2)) if match.group(2) else -1

        return current_date, set_number

    def plot_force_trace(self):
        tp_path = os.path.join(self.proc_ss, "timepoints.csv")

        if not os.path.isfile(tp_path):
            ws(f"Could not find timepoints csv {tp_path}, skipping force trace")
            return

        else:
            tp_df = pd.read_csv(tp_path)
            self.plot_force_traces(
                tp_df,
                self.results_dir,
                ref_events=['success_grasp_start', 'ttl_to_reward']
            )

    @staticmethod
    def get_trial_force(trial_info):

        # 1. Get tsm data
        latTsmFile = trial_info.filtered_ps_filenames['lateral_sensor']
        medTsmFile = trial_info.filtered_ps_filenames['medial_sensor']

        if not os.path.isfile(latTsmFile) or not os.path.isfile(medTsmFile):
            print(f'Skipping trial {trial_info.trial_number} | missing at least one'
                  f' ps file:\n{latTsmFile}\n{medTsmFile}')
            return ([], [])

        times, forces_summed = get_summed_force_data(latTsmFile, medTsmFile)

        # Return and THEN bind results to trial object
        return (times, forces_summed)

    def plot_force_traces(
        self,
        timepoints_df,
        savedir,
        ref_events=[],
        max_discrete_conds=9,
        time_bin_width=0.02,
        pre_event_pad=1,
        post_event_pad=1,
        processes=os.cpu_count() - 1,
        only_successful_trials=True
    ):

        # 1. get raw tsm data (times and forces) for each trial
        # feed trials from self.msession
        trials_flattened = self.msession[:]
        print(f"Plotting force traces for {len(trials_flattened)} trials")
        p_args = list(zip(*[trials_flattened]))
        results = ReportingPool(SessionWrapper.get_trial_force, p_args, processes=processes,
                    report_on_change=True, track_failures=True).start()

        # Get min and max force targets (needed for normalizing cmap)
        force_conditions = [tr.target_force for tr in trials_flattened]
        min_cond = min(force_conditions)
        max_cond = max(force_conditions)

        b_continous = len(set(force_conditions)) > max_discrete_conds
        print(f'b_continous: {b_continous}')

        # Bind times and forces to each trial
        for res, tr in zip(results, self.msession):
            if res is None:
                continue
            elif len(res) == 0:
                continue
            tr.tsm_times = res[0]
            tr.forces_summed = res[1]


        # Only include trials that have tsm data
        trials_flattened = [tr for tr in trials_flattened if
                            hasattr(tr, 'tsm_times') and hasattr(tr, 'forces_summed')]

        trials_flattened = [tr for tr in trials_flattened if
                            len(tr.tsm_times) > 0 and len(tr.forces_summed) > 0]

        # Create times to interp over since this will stay constant
        interp_times = np.arange(-pre_event_pad, post_event_pad + time_bin_width, time_bin_width)

        # Define cmap for plotting
        cmap = plt.cm.Oranges

        # Create a subset of the oranges colormap
        cmap = LinearSegmentedColormap.from_list(
            'subset_oranges', cmap(np.linspace(10 / 25.5, 1, 255 - 100 + 1))
        )

        norm = mcolors.Normalize(vmin=min_cond, vmax=max_cond)
        sm = ScalarMappable(norm=norm, cmap=cmap)

        # Define a function to get color based on a value
        def get_color(value):
            return cmap(norm(value))

        # Make a plot for each reference event
        for ref_event in ref_events:

            # Bind interpolated forces to each trial (this depends on reference event)
            for trial in trials_flattened:

                if only_successful_trials and not trial.success:
                    continue ## skip failed trial

                #trial.tsm_times = np.array(trial.tsm_times)
                msg = f'trial.tsm times is type ({type(trial.tsm_times)})'
                assert isinstance(trial.tsm_times, np.ndarray), msg

                # Set default value
                trial.force_interped = None

                # 1. zero to reference time (in place modification of trial object)
                if ref_event in timepoints_df.columns:
                    row = timepoints_df[timepoints_df['trial_number'] == trial.trial_number]
                    if len(row[ref_event].values) < 1:
                        ws('No zero time point reference value found in this csv {}, skipping')
                        continue
                    zeroTimeVal = row[ref_event].values[0]
                    if np.isnan(zeroTimeVal):
                        # Got to next trial...
                        continue

                    shifted_times = trial.tsm_times - float(zeroTimeVal)

                elif hasattr(trial, ref_event):
                    shifted_times = trial.tsm_times - float(getattr(trial, ref_event))

                else:
                    continue

                # 2. trim to pad region (in place)
                msg = f'shifted times check 1 is type ({type(trial.tsm_times)})'
                assert isinstance(shifted_times, np.ndarray), msg
                valid_idx = (shifted_times >= -pre_event_pad) & (shifted_times <= post_event_pad)
                msg = f'shifted times check 2 is type ({type(trial.tsm_times)})'
                assert isinstance(shifted_times, np.ndarray), msg
                shifted_times = shifted_times[valid_idx]
                interp_forces = trial.forces_summed[valid_idx]

                if len(interp_forces) < 2:
                    # Not enough data to interp over
                    continue

                # 3. Interpolate
                fxn = interp1d(shifted_times, interp_forces,
                                kind='linear', fill_value='extrapolate')
                trial.force_interped = fxn(interp_times).clip(min=0)
                # Clip to fix weird case of very negative values

            # Create line plot with avg
            fig, ax = plt.subplots(figsize=(15,10))
            total_trials = 0

            # all_bounds = set()
            # all_colors = list()

            # for tr in trials_flattened:
            #     # Check if the condition is a range or not
            #     if b_continous:
            #         f0, ff = tr.range_delta
            #         cond_color = get_color(float((f0 + ff) / 2))
            #         tf = 0  ## set for plotting bounds later
            #     else:
            #         # DEBUG
            #         tf = tr.target_force
            #         cond_color = get_color(tf)

            # Get a list of all interpolated force arrays
            filtered_trials = [tr for tr in trials_flattened if hasattr(tr, 'force_interped')]
            filtered_trials = [tr for tr in filtered_trials if tr.force_interped is not None]

            assert len(filtered_trials) > 0, 'No trials with interpolated force data'

            # Start plotting ...
            # 1. Normal mode (discrete targets)
            if not b_continous:
                # NORMAL MODE
                # For each force condition create the plot

                # keep track of all forces at a given target for avg
                dict_trials_per_target = {}

                for tr in filtered_trials:
                    ax.plot(interp_times, tr.force_interped, linewidth=0.75,
                            color=get_color(tr.target_force), alpha=0.2)
                    total_trials += 1

                    # Append trial to appropriate list
                    if tr.target_force not in dict_trials_per_target:
                        dict_trials_per_target[tr.target_force] = [tr]
                    else:
                        dict_trials_per_target[tr.target_force].append(tr)

                # plot avg per target force
                for tf in dict_trials_per_target:
                    interped_list = [tr.force_interped for tr in dict_trials_per_target[tf]]
                    force_sum = np.sum(interped_list, axis=0)
                    force_sum /= len(interped_list)

                    # Plot the force sum
                    ax.plot(interp_times, force_sum, linewidth=2,
                            color=get_color(tf), alpha=1, label=f'Force Target: {tf} N')

            # 2. Continuous mode (residual force)
            else:
                # Residual (difference from target force) mode
                for tr in filtered_trials:
                    # Plot residual force
                    ax.plot(interp_times, (tr.force_interped - tr.target_force), linewidth=0.75,
                            color=get_color(tr.target_force), alpha=0.2)

                    # TODO: Make avg force vs target force plot here
                    total_trials += 1

            # Add bounds to set to draw later
            # all_bounds |= {tr.range_delta for tr in filtered_trials}
            # all_colors.append(cond_color)

            # # Draw unique bounds
            # for bnds, color in zip(all_bounds, all_colors):
            #     ax.fill_between(interp_times, *bnds, color=color, alpha=0.1, edgecolor='none')

            title = f'Force Traces ({total_trials} Trials)'

            if b_continous:
                title = "Residual " + title
                fig.colorbar(sm, label="Target Force (N)", ax=ax)
                ax.set_ylabel('$\Delta F_{actual, target}$ (N)', fontsize=14)
                #cbar.set_clim(vmin=0, vmax=max_cond)
            else:
                ax.legend(loc='upper right')
                ax.set_ylabel(f"L/R summed force (N)", fontsize=14)

            ax.set_xlabel("$\Delta t_{event}$ (s)", fontsize=14)
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='--', label='test label')
            _, maxy = ax.get_ylim()
            ax.text(0, 0.6 * maxy, ref_event, rotation=90,
                    va='bottom', ha='right')

            savename = os.path.join(savedir, f'ForceTrace_from_{ref_event}')
            plt.savefig(savename)
            rs(f'saving forcetrace as {os.path.normpath(savename)}')
            plt.close(fig)

    def plot_cond_success_matrix(self):
        cond_success_matrix_path = os.path.join(self.results_dir, 'CondSuccessMatrix.png')
        self.__plot_single_cond_success_matrix(cmap='Purples', savename=cond_success_matrix_path)

    def __plot_single_cond_success_matrix(self, cmap='Blues', savename=None):
        create_heatmap_from_trials_list(
            self.msession,
            cmap,
            savename=savename
        )



def main(preset, sessions, temp, overwrite, transfer=False):

    # Check both raw and processed servers exist
    if not os.path.exists(preset['default_server']):
        raise ValueError(f"Default server directory {preset['default_server']} does not exist")

    if not os.path.exists(preset['processed_server']):
        raise ValueError("Default processed server directory"
                         f" {preset['default_server']} does not exist")

    # Setup logging
    setup_logging(temp, sessions_dir=preset['processed_server'])

    # Check if preset defines training servers, if so we try transferring sessions to training,
    has_training_server = does_trianing_servers_exist(preset)

    # Move sessions from experiment to training
    if transfer:
        rs('\n'*3 + '='*200)
        if has_training_server:
            rs('Moving training sessions')
            experimental_ss_pairs_pre_move, _ = fetch_server_session_dirs(preset, sessions,
                                                                          filter=False)
            for sw in tqdm([SessionWrapper(*exp_pair) for exp_pair in
                                 experimental_ss_pairs_pre_move]):
                sw.ensure_transfer_to_training_server(preset['default_training_server'],
                                                      preset['processed_training_server'],
                                                      overwrite)
            rs('\n'*3 + '='*200)
        else:
            rs(f"No training servers defined for preset {preset['names'][0]},"
               " skipping transfer step.")

    # Get raw/processed session dirs for preset
    # Note: we want to fetch all sessions here for performance analysis
    experimental_ss_pairs, training_ss_pairs = fetch_server_session_dirs(preset, filter=True)

    exp_session_wrappers = [SessionWrapper(*exp_pair) for exp_pair in experimental_ss_pairs]
    train_session_wrappers = [SessionWrapper(*train_pair) for train_pair in training_ss_pairs]

    exp_grp = SessionGroup(
        exp_session_wrappers, group_label='Experiment')
    train_grp = SessionGroup(
        train_session_wrappers, group_label='Training')

    exp_grp.do_all_analysis(overwrite, sessions=sessions)
    rs("\n" * 3 + "=" * 200)
    train_grp.do_all_analysis(overwrite, sessions=sessions)


# Entry
if __name__ == "__main__":

    preset_name = sys.argv[1]
    if preset_name not in PRESETS.keys():
        raise ValueError(f'preset_name {preset_name} not found in presets {list(PRESETS.keys())}')

    current_preset = PRESETS[preset_name]

    # Remove axis spines from plot
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.spines.top'] = False

    # Add arguments
    parser = argparse.ArgumentParser(
        description=("Create plots for a given monkey"))

    cmd_args.add_default_arguments(
        parser, ("sessions", "temp", "overwrite")
    )

    args = parser.parse_args(sys.argv[2:])
    start_time = time.time()

    main(
        current_preset,
        args.sessions,
        args.temp,
        args.overwrite,
    )

    rs('Program took {}.'.format(
        timedelta(seconds=time.time() - start_time)))
