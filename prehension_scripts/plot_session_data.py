#!python3.7
import argparse
import time
import os
import datetime
import sys
import tqdm
import math
import numbers
import shutil
import glob
import pandas as pd
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.cm import ScalarMappable
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
from reporting_pool import ReportingPool
from matplotlib.colors import LinearSegmentedColormap

#from prehension import tools, meta_session
from prehension_presets.prehension_presets import PRESETS
from prehension.tools.logs import rs, ws, setup_logging
from prehension import meta_session
from prehension.tools import cmd_args
from prehension.tools.forces import get_summed_force_data
from prehension.tools.utils import fetch_server_session_dirs, does_trianing_servers_exist


class SessionGroup:

    """
    Wrapper for multiple session data and analysis,
    Includes methods for plotting the following
    - performance per session plots across time
    - conditional success matrix for all sessions
    """

    def __init__(self, l_session_wrappers, warn_duplicates_per_date):
        self.session_wrappers = l_session_wrappers

        # dictionary of form -> datetime: list[SessionWrapper]
        self.session_wrappers_by_date = {}
        for sw in self.session_wrappers:
            if sw.datetime not in self.session_wrappers_by_date:
                self.session_wrappers_by_date[sw.datetime] = [sw]
            else:
                if warn_duplicates_per_date:
                    ws(f'Duplicate session for {sw.datetime}')
                self.session_wrappers_by_date[sw.datetime].append(sw)


    def plot_avg_cond_success_matricies(self):

        # First sort all of the session wrappers by ascending datetime
        msession_l = []
        mobject_l = []

        for datetime in tqdm.tqdm(sorted(self.session_wrappers_by_date.keys()), desc='Plotting Performance'):

            sessions_per_date = self.session_wrappers_by_date[datetime]

            # Combine date from multiple sessions from the same day
            datetime_msession = []
            datetime_mobject = []
            for sw in sessions_per_date:
                datetime_msession += sw.msession_l
                datetime_mobject += sw.mobject_l

            msession_l.append(datetime_msession), mobject_l.append(datetime_mobject)

            # Save a plot in all same day folders:
            for i, sw in enumerate(sessions_per_date):
                title = f'Daily performance ({i+1}/{len(sessions_per_date)}) sessions up to {datetime}'
                plot_cond_success_matrix(msession_l, mobject_l, title, cmap='Greens',
                                         savename=os.path.join(sw.results_dir, 'CondSuccessMatrix.png'))

    @staticmethod
    def plot_performace_fig(date_success_dict,
                            last_n_days_label=None, include_last_tick=False,
                            annotate_dates=False, savename=None):

        fig, axs = plt.subplots(1, 2, figsize=(15, 7))
        suptitle = f'Training Progress (last {last_n_days_label} days)' if last_n_days_label else 'Training Progress'
        fig.suptitle(suptitle)

        # Plot for Performance
        axs[0].set_title('Performance')
        axs[0].set_ylabel('Percent (%)')
        axs[0].set_ylim((0, 1))

        dates = sorted(date_success_dict.keys())
        pct_correct = [100 * sum(date_success_dict[dt]) / len(date_success_dict[dt]) for dt in dates]
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


    def plot_performance(self):
        # Build dictionary of form -> datetime: list[trial success bools for that date]
        # Note we need to aggregate all sessions for the same date

        date_l_trial_success_dict = {}

        for datetime in self.session_wrappers_by_date:
            trial_success_list_per_date = []
            for sw in self.session_wrappers_by_date[datetime]:
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
                                             savename=os.path.join(sw.results_dir, 'Performance.png'))
            SessionGroup.plot_performace_fig(date_success_dict_10_day,
                                             last_n_days_label='10',
                                             include_last_tick=True,
                                             annotate_dates=True,
                                             savename=os.path.join(sw.results_dir, 'PerformanceLast10Days.png'))



class SessionWrapper:

    """
    Wrapper for single session data and analysis,
    Includes methods for plotting the following
    - force traces for session
    - conditional success matrix for session
    """

    def __init__(self, raw_ss, proc_ss):
        assert os.path.isdir(raw_ss) and os.path.isdir(proc_ss)
        self.raw_ss = raw_ss
        self.proc_ss = proc_ss

        self.sess_name = os.path.basename(self.raw_ss)
        self.datetime = date_from_folder(self.sess_name)

        # Create results and log folder
        self.results_dir = os.path.join(self.proc_ss, 'prehension_plots')
        os.makedirs(self.results_dir, exist_ok=True)

        # Load meta
        self.mstruct, self.mdof, self.mobject, self.msession = meta_session.load_meta_information(self.raw_ss, self.proc_ss)


    def plot_force_trace(self, processes=17):
        tp_path = os.path.join(self.proc_ss, "timepoints.csv")

        if not os.path.isfile(tp_path):
            ws(f"Could not find timepoints csv {tp_path}, skipping force trace")
            return

        trial_cond_info = build_cond_trial_dict(self.mobject, self.msession, triple_key=False)

        if trial_cond_info is None:
            ws('trial condition info is None, skipping..')
        else:
            tp_df = pd.read_csv(tp_path)
            tp_df = pd.read_csv(tp_path)
            plot_force_traces(*trial_cond_info[:-1], tp_df, self.results_dir,
                              ref_events=['success_grasp_start', 'ttl_to_reward'],
                                processes=processes)


    def plot_cond_success_matrix(self):
        cond_success_matrix_path = os.path.join(self.results_dir, 'CondSuccessMatrix.png')
        SessionWrapper._plot_cond_success_matrix([self.msession,], [self.mobject,], title=self.sess_name,
                                    cmap='Purples', savename=cond_success_matrix_path)


    @staticmethod
    def _plot_cond_success_matrix(msession_l, mobject_l, title, cmap, savename):

        # Force conditions should be a list of [(lo, hi), ...]
        # or [target1, target2]
        cond_success_dict = {}

        def get_pct_true(l):
            return sum(l) / len(l)

        # Accumulate data for diff conditions
        n_trials = 0

        # Accumulate all conditions
        cond_all_trial_dict = {}

        for msession, mobject in zip(msession_l, mobject_l):
            # Get continuous force bins
            cond_trial_dict, _, _, _ = build_cond_trial_dict(mobject, msession, max_discrete_conds=0, triple_key=True)
            # Dictionary of form

            cond_all_trial_dict = merge_dicts(cond_all_trial_dict, cond_trial_dict)

        for cond, trials in cond_all_trial_dict.items():
            # Cound values
            n_trials += len(trials)
            cond_success_dict[cond] = (get_pct_true([tr.success for tr in trials]), len(trials))
            ## values are (pct success, ntrials)

        create_heatmaps_subplots(cond_success_dict,
                                f'Condition Success Matrices\n{title}\n(trials={n_trials})',
                                cmap,
                                savename=savename)




# --- HELPERS --- #
def plot_avg_target_ft():
    # TODO: for each trial plot an x,y point
    # x: target force
    # y: avg in region before event
    # draw a line of best fit over those points
    pass


def merge_dicts(dict1, dict2):
    merged_dict = {}
    # Merge dict1 into merged_dict
    for key, value in dict1.items():
        if key in dict2:
            merged_dict[key] = value + dict2[key]
        else:
            merged_dict[key] = value
    # Add keys from dict2 not present in dict1
    for key, value in dict2.items():
        if key not in dict1:
            merged_dict[key] = value
    return merged_dict


def get_trial_force(trial_info):

    # 1. Get tsm data
    latTsmFile = trial_info.filtered_ps_filenames['lateral_sensor']
    medTsmFile = trial_info.filtered_ps_filenames['medial_sensor']

    if not os.path.isfile(latTsmFile) or not os.path.isfile(medTsmFile):
        print(f'Skipping trial {trial_info.trial_number} | missing at least one' \
              f' ps file:\n{latTsmFile}\n{medTsmFile}')
        return ([], [])

    times, forces_summed = get_summed_force_data(latTsmFile, medTsmFile)

    # Return and THEN bind results to trial object
    return (times, forces_summed)


def date_from_folder(folder):
    elems = "_".join(folder.split('_')[:3])
    current_date = datetime.strptime(elems, r'%Y_%m_%d')
    return current_date


def get_date_folders(ref_server_session, mode='before',
                     lookback_timedelta=timedelta(days=3650), include_ref=True):
    # Return all folders in server dir before datetime of ref_server_session

    ref_dt = date_from_folder(os.path.basename(ref_server_session))
    min_dt = ref_dt - lookback_timedelta

    server = os.path.dirname(ref_server_session)

    valid_folders = []

    for folder in os.listdir(server):

        if not folder[0].isdigit():
            continue

        if not os.path.isdir(os.path.join(server, folder)):
            continue
        # For each folder name try parsing the datetime
        if len(folder) < 8:
            continue  # TO SHORT

        # Parse the string into a datetime object
        current_date = date_from_folder(folder)

        if mode == 'before':
            if current_date < ref_dt and current_date >= min_dt:
                valid_folders.append(os.path.join(server, folder))
        elif mode == 'after':
            if current_date > ref_dt:
                valid_folders.append(os.path.join(server, folder))
        else:
            raise ValueError(f"Mode must be before or after but got {mode}")

    if include_ref:
        valid_folders.append(ref_server_session)

    return valid_folders


def create_heatmaps_subplots(data, title, cmap, display_n=True, savename=None):
    # Group by first key
    force_targets = sorted(set([k[0] for k in data.keys()]))
    num_force_targets = len(force_targets)

    if num_force_targets == 0:
        ws("No force targets found returning.")
        return

    # Determine the optimal grid layout based on the number of force targets
    num_rows = int(np.ceil(np.sqrt(num_force_targets)))
    num_cols = int(np.ceil(num_force_targets / num_rows))

    # Create a grid of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    fig.suptitle(title)

    # Flatten the axes array if necessary
    if num_rows * num_cols > 1:
        axs = axs.flatten()
    else:
        axs = [axs,]

    for idx, force_target in enumerate(force_targets):
        # Get the relevant sub keys for the current force target
        sub_keys = [tup[1:] for tup in data.keys() if tup[0] == force_target]

        # Get unique rotations and apertures
        unique_rots = sorted(set(x[0] for x in sub_keys))
        unique_aps = sorted(set(x[1] for x in sub_keys))

        # Create a heatmap matrix
        hm = []
        for i, rot in enumerate(unique_rots):
            row = []
            for j, ap in enumerate(unique_aps):
                key = (force_target, rot, ap)
                if key in data.keys():
                    pct, n = data[key]
                    row.append(pct)
                    color = 'white' if pct >= 0.5 else 'black'
                    txt = f'{pct:.2f}'
                    if display_n:
                        txt += f"\n(n={n})"
                    axs[idx].text(j, i, txt, ha="center", va="center", color=color)
                else:
                    # No data found for the current combination
                    row.append(0)
            hm.append(row)

        # Plot the heatmap
        axs[idx].imshow(hm, cmap=cmap, vmin=0, vmax=1)
        axs[idx].set_xticks(np.arange(len(unique_aps)))
        axs[idx].set_xticklabels(unique_aps)
        axs[idx].set_yticks(np.arange(len(unique_rots)))
        axs[idx].set_yticklabels(unique_rots)
        title = f"Force: {force_target} N"
        if len(force_target) == 2:
            title = f"Force Range: {force_target[0]}-{force_target[1]} N"
        axs[idx].set_title(title)
        axs[idx].set_xlabel('Aperture (mm)')
        axs[idx].set_ylabel('Rotation (deg)')

    # Hide any unused subplots
    for i in range(num_force_targets, num_rows * num_cols):
        fig.delaxes(axs[i])

    plt.tight_layout()
    if savename is not None:
        fig.savefig(savename)
        #rs(f'saving success matrix as {os.path.normpath(savename)}')
    else:
        plt.show()
    plt.close(fig)

# --- PLOTTING FXNS --- #
def plot_avg_cond_success_matrix(raw_ss, proc_ss, savename):
    msession_l = []
    mobject_l = []

    for ss in get_date_folders(raw_ss, mode='before'):
        try:
            _, _, mo, ms = meta_session.load_meta_information(raw_ss, proc_ss)
        except FileNotFoundError as fnfe:
            ws(f'Skipping {ss} due to error loading meta: {fnfe}')
            continue
        msession_l.append(ms), mobject_l.append(mo)

    title = f'{len(msession_l)} sessions up to {os.path.basename(raw_ss)}'
    plot_cond_success_matrix(msession_l, mobject_l, title, cmap='Greens', savename=savename)


def plot_cond_success_matrix(msession_l, mobject_l, title, cmap, savename):

    # Force conditions should be a list of [(lo, hi), ...]
    # or [target1, target2]
    cond_success_dict = {}

    def get_pct_true(l):
        return sum(l) / len(l)

    # Accumulate data for diff conditions
    n_trials = 0

    # Accumulate all conditions
    cond_all_trial_dict = {}

    for msession, mobject in zip(msession_l, mobject_l):
        # Get continuous force bins
        cond_trial_dict, _, _, _ = build_cond_trial_dict(mobject, msession, max_discrete_conds=0, triple_key=True)
        # Dictionary of form

        cond_all_trial_dict = merge_dicts(cond_all_trial_dict, cond_trial_dict)

    for cond, trials in cond_all_trial_dict.items():
        # Cound values
        n_trials += len(trials)
        cond_success_dict[cond] = (get_pct_true([tr.success for tr in trials]), len(trials))
        ## values are (pct success, ntrials)

    create_heatmaps_subplots(cond_success_dict,
                             f'Condition Success Matrices\n{title}\n(trials={n_trials})',
                             cmap,
                             savename=savename)
    # Now merge based on force, rotation, aperture conditions
    # for oid in oid_success_dict.keys():


def plot_force_traces(
    cond_trial_dict,
    min_cond,
    max_cond,
    timepoints_df,
    savedir,
    ref_events=[],
    time_bin_width=0.02,
    pre_event_pad=1,
    post_event_pad=1,
    processes=8,
    draw_bounds=True
):

    # 1. get raw tsm data (times and forces) for each trial
    # Do this in parallel to avoid tsm memory problem
    trials_flattened = [
        x
        for xs in list(cond_trial_dict.values())
        for x in xs
    ]

    p_args = list(zip(*[trials_flattened]))
    results = ReportingPool(get_trial_force, p_args, processes=processes,
                  report_on_change=True, track_failures=True).start()

    # Bind properties after the fact
    for res, tr in zip(results, trials_flattened):
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
    # start_pct = 0.7  # Start cmap at 50%
    # min0 = max_cond - ((max_cond-min_cond)/(1-start_pct))
    #norm = mcolors.Normalize(vmin=min0, vmax=max_cond)
    # Create a subset of the oranges colormap
    cmap = LinearSegmentedColormap.from_list(
        'subset_oranges', cmap(np.linspace(10 / 25.5, 1, 255 - 100 + 1))
    )

    norm = mcolors.Normalize(vmin=min_cond, vmax=max_cond)
    sm = ScalarMappable(norm=norm, cmap=cmap)


    # Define a function to get color based on a value
    def get_color(value):
        return cmap(norm(value))

    # 2. single plot function
    def create_single_plot(ref_event, only_successful=True):
        # This should modify trials in place to:
        # 1. zero time to reference event time
        # 2. trim to pad region

        # 3. interpolate over common times
        # 4. create plot

        for trial in trials_flattened:

            if only_successful and not trial.success:
                continue ## skip failed trial

            #trial.tsm_times = np.array(trial.tsm_times)
            assert isinstance(trial.tsm_times, np.ndarray), f'trial.tsm times is type ({type(trial.tsm_times)})'

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
            assert isinstance(shifted_times, np.ndarray), f'shifted times check 1 is type ({type(trial.tsm_times)})'
            valid_idx = (shifted_times >= -pre_event_pad) & (shifted_times <= post_event_pad)
            assert isinstance(shifted_times, np.ndarray), f'shifted times check 2 is type ({type(trial.tsm_times)})'
            shifted_times = shifted_times[valid_idx]
            interp_forces = trial.forces_summed[valid_idx]

            if len(interp_forces) < 2:
                # Not enough data to interp over
                continue

            # 3. Interpolate
            fxn = interp1d(shifted_times, interp_forces,
                            kind='linear', fill_value='extrapolate')
            trial.force_interped = fxn(interp_times).clip(min=0)
            # Clip to fix wierd case of very negative values

        # Create line plot with avg
        fig, ax = plt.subplots(figsize=(15,10))
        total_trials = 0

        b_isrange = len(list(cond_trial_dict.keys())[0]) == 2

        all_bounds = set()
        all_colors = list()

        for cond, trial_list in cond_trial_dict.items():

            # Check if the condition is a range or not
            if b_isrange:
                f0, ff = cond
                cond_color = get_color(float((f0 + ff) / 2))
                tf = 0  ## set for plotting bounds later
            else:
                # DEBUG
                tf = cond[0]
                cond_color = get_color(tf)

            # Get a list of all interpolated force arrays
            good_trials = [tr for tr in trial_list if hasattr(tr, 'force_interped')]
            interped_list = [tr.force_interped for tr in good_trials if tr.force_interped is not None]

            if len(interped_list) == 0:
                ws(f'No interpolated forces for condition: {cond}. Continuing')
                continue

            if not b_isrange:
                # NORMAL MODE
                # For each force condition create the plot
                for f_interp in interped_list:
                    ax.plot(interp_times, f_interp, linewidth=0.75,
                            color=cond_color, alpha=0.2)
                    total_trials += 1

                # Compute avg force trace
                force_sum = np.sum(interped_list, axis=0)
                force_sum /= len(interped_list)

                # Plot the force sum
                ax.plot(interp_times, force_sum, linewidth=2,
                        color=cond_color, alpha=1, label=f'Force Target: {tf} N')

            else:
                # Residual (difference from target force) mode
                targets_list = [tr.target_force for tr in trial_list]
                for f_array, target_force in zip(interped_list, targets_list):
                    # Plot residual force
                    ax.plot(interp_times, (f_array - target_force), linewidth=0.75,
                             color=cond_color, alpha=0.2)

                    # TODO: Make avg force vs target force plot here
                    total_trials += 1

            # Add bounds to set to draw later
            interped_list = [tr for tr in good_trials if tr.range_delta is not None]
            all_bounds |= {(tf + tr.range_delta[0], tf + tr.range_delta[1]) for tr in interped_list}
            all_colors.append(cond_color)

        # Draw unique bounds
        if draw_bounds:
            for bnds, color in zip(all_bounds, all_colors):
                ax.fill_between(interp_times, *bnds, color=color, alpha=0.1, edgecolor='none')

        title = f'Force Traces ({total_trials} Trials)'

        if b_isrange:
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
        #rs(f'saving forcetrace as {os.path.normpath(savename)}')
        plt.close(fig)

    # 3. create a plot for each reference event desired
    for ref_event in ref_events:
        create_single_plot(ref_event)

    return


def plot_performance(
        raw_server,
        proc_server,
        ref_date,
        savename=None,
        lookback_timedelta=timedelta(days=3650),  ## 10 years
        include_last_tick=False
    ):

    # This will hold our results
    date_l_trial_success_dict = {}

    # LOOP 1: build a dictionary of (key, value) == (date, [list success bools])
    for dirname in os.listdir(raw_server):

        # Perform 4 checks on folder validity

        rss = os.path.join(raw_server, dirname)
        if not os.path.isdir(rss):
            continue

        if not dirname[0].isdigit():
            continue

        datetime = date_from_folder(dirname)
        if datetime > ref_date:
            continue  # Too new

        if datetime < ref_date - lookback_timedelta:
            continue  # Too old

        pss = os.path.join(proc_server, dirname)
        if not os.path.isdir(pss):
            continue

        # Load metadata
        try:
            _, _, _, msession = meta_session.load_meta_information(rss, pss)
        except FileNotFoundError as fnfe:
            #ws(f'Skipping {rss} due to error loading meta: {fnfe}')
            continue
        except meta_session.IncompleteMetaError as imfe:
            #ws(f'Skipping {rss} due to incomplete meta (TRY workaround later): {imfe}')
            continue

        # Add data to dictionary
        trial_success_list = [tr.success for tr in msession]

        if datetime not in date_l_trial_success_dict:
            date_l_trial_success_dict[datetime] = trial_success_list
        else:
            #rs('More than one session found for this date: {}'.format(dirname))
            date_l_trial_success_dict[datetime].extend(trial_success_list)

    # LOOP 2: plot the results
    dates = []
    total_trials = []
    pct_correct = []
    pct_incorrect = []
    num_correct = []
    num_incorrect = []

    for date, successful in sorted(date_l_trial_success_dict.items(), reverse=True):
        dates.append(date)
        ntrials = len(successful)
        total_trials.append(ntrials)
        pct_correct.append(sum(successful) / len(successful))
        pct_incorrect.append(1 - pct_correct[-1])
        num_correct.append(sum(successful))
        num_incorrect.append(ntrials - num_correct[-1])

    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle(f'Training Progress (last {lookback_timedelta.days} days)')

    # Plot for Performance
    axs[0].set_title('Performance')
    axs[0].set_ylabel('Percent (%)')
    axs[0].set_ylim((0, 1))

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
    if lookback_timedelta <= timedelta(weeks=2):
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


def build_cond_trial_dict(mobject, msession, bin_width_N=1, max_discrete_conds=10, triple_key=True):
    """Determine if there are more than max_discrete_conds number of force conditions.
    If there is, create a range of force bins to use as force conditions (continuous mode)
    Else use the existing force conditions as keys. Return a dictionary of
    -> {(force condition) : [trials that use given condition]}

    Args:
        mobject (meta.MOBJECT): Meta object
        msession (meta.MSESSION): Meta session (list of trial objects)
        bin_width_N (int, optional): Bin width in Newtons. Defaults to 1.
        max_discrete_conds (int, optional): The number of force conditions,
        above which we use continuous force modes. Defaults to 10.

    Returns:
        Dictionary: {force condition: [trials matching force condition]}
    """

    def attach_fields_cond_fields(trial, mobject, target_condition, target_force):
        # 1. bind force bound differences to trial
        # 2. bind target condition to trial
        # 3. bind target force to trial
        trial.target_condition = target_condition
        trial.target_force = target_force
        stub = mobject[trial.object_id]['def']
        bound_keys = ['targetForceRelRangeMin(N)', 'targetForceRelRangeMax(N)']
        # Check if keys exist in the stub
        trial.range_delta = None
        if all([k in stub.keys() for k in bound_keys]):
            trial.range_delta = [float(stub[bk]) for bk in bound_keys]

        # Add aperture and rotation information
        trial.rotation = stub['pos_aperture(mm)']
        trial.aperture = stub['pos_tilt(deg)']

    # Find target force for each trial
    trial_target_dict = {}
    for tr in msession:
        obj_stub = mobject[tr.object_id]['def']
        if 'targetForce(N)' not in obj_stub.keys():
            msg = f"Expected targetForce(N) in keys of object stub but only found: {obj_stub}"
            rs(msg)
            continue
        trial_target_dict[tr] = float(obj_stub['targetForce(N)'])

    # Find num discrete target forces
    discrete_targets = set(trial_target_dict.values())
    if not discrete_targets:
        return  [None, None, None, None] ## empty set

    rounded_min_cond = int(math.floor(min(discrete_targets)))
    rounded_max_cond = int(math.ceil(max(discrete_targets)))

    # Determine if we should use continous mode
    b_continuous = len(discrete_targets) > max_discrete_conds

    # Let the user know
    if b_continuous:
        # Create force range bins
        force_bins = []
        for lo_bound in range(rounded_min_cond,
                              rounded_max_cond,
                              bin_width_N):
            force_bins.append((lo_bound, lo_bound + bin_width_N))

        cond_trials_dict = {k: [] for k in force_bins}
        for trial, force in trial_target_dict.items():
            for condition, trials_in_condition in cond_trials_dict.items():
                if condition[0] <= force <= condition[1]:
                    attach_fields_cond_fields(trial, mobject, condition, force)
                    trials_in_condition.append(trial)

    # Discrete force ranges
    else:
        cond_trials_dict = {(tf, ): [] for tf in discrete_targets}
        for trial, force in trial_target_dict.items():
            attach_fields_cond_fields(trial, mobject, (force, ), force)
            cond_trials_dict[(force, )].append(trial)

    # If triple_key, reformat key to be target,
    cond_trials_dict_triple = {}
    if triple_key:
        for cond0, trials in cond_trials_dict.items():
            sub_d = {} # Should all have the same first element of key
            for tr in trials:
                tri_key = (cond0, tr.rotation, tr.aperture)
                if tri_key in sub_d.keys():
                    sub_d[tri_key].append(tr)
                else:
                    sub_d[tri_key] = [tr, ]
            cond_trials_dict_triple.update(sub_d)
        cond_trials_dict = cond_trials_dict_triple

    # Return (cond, [trials])
    # Sort by the first force level
    sort_fxn = lambda x: x if isinstance(x, numbers.Number) else sort_fxn(x[0])
    d = {k: v for k, v in sorted(cond_trials_dict.items(), key=sort_fxn)}
    retval = [d, rounded_min_cond, rounded_max_cond, b_continuous]
    return retval


# def process_session(raw_ss, proc_ss, overwrite, processes):

#     # Create output dir with plots
#     session = os.path.basename(raw_ss)
#     #rs("Processing session {}.".format(os.path.basename(session)))

#     # Create results and log folder
#     results_dir = os.path.join(proc_ss, 'prehension_plots')
#     os.makedirs(results_dir, exist_ok=True)

#     if not os.path.exists(proc_ss) or not os.path.exists(raw_ss):
#         ws("Session {} does not exist on the server.".format(session))
#         return

#     try:
#         mstruct, mdof, mobject, msession = meta_session.load_meta_information(raw_ss, proc_ss)
#     except Exception as e:
#         ws('Could not load meta data from session {} || {} ({}), skipping.'.format(raw_ss, proc_ss, repr(e)))
#         return

#     # Sort trials by force condition
#     trial_cond_info = build_cond_trial_dict(mobject, msession, triple_key=False)
#     ft_reference_events = ['success_grasp_start', 'ttl_to_reward']
#     ft_save_paths = [os.path.join(results_dir, f'ForceTrace_from_{evt}.png')
#                       for evt in ft_reference_events]
#     tp_path = os.path.join(proc_ss, "timepoints.csv")
#     cond_succ_path=os.path.join(results_dir, 'CondSuccessMatrix.png')
#     avg_succ_path = os.path.join(results_dir, 'AvgCondSuccessMatrix.png')
#     perf_path = os.path.join(results_dir, 'Performance.png')
#     perf_minus10_day_path = os.path.join(results_dir, 'PerformanceLast10Days.png')

#     # Determine what we need to write
#     # TODO reinstate expression when testing done
#     make_force_traces = False #not all([os.path.isfile(f) for f in ft_save_paths]) or overwrite
#     make_cond_succ_mat = not os.path.isfile(cond_succ_path) or overwrite
#     make_avg_succ_mat = not os.path.isfile(avg_succ_path) or overwrite
#     make_perf_plot = not os.path.isfile(perf_path) or overwrite
#     make_perf10day_plot = not os.path.isfile(perf_minus10_day_path) or overwrite

#     # Return if nothing to make
#     if not any([make_force_traces, make_cond_succ_mat, make_avg_succ_mat,
#                 make_perf_plot, make_perf10day_plot]):
#         return

#     # Else create a message for plotting
#     # msg = "Making"
#     # if make_force_traces: msg += " force traces,"
#     # if make_cond_succ_mat: msg += " conditional success plot,"
#     # if make_avg_succ_mat: msg += " average success plot,"
#     # if make_perf_plot: msg += " longitudinal performance plot"
#     # if make_perf10day_plot: msg += " last 10 days performance plot"

#     if not os.path.isfile(tp_path):
#         ws(f"Could not find timepoints csv {tp_path}, skipping force trace")
#     elif make_force_traces:
#         if trial_cond_info is None:
#             ws('trial condition info is None, skipping..')
#         else:
#             tp_df = pd.read_csv(tp_path)
#             plot_force_traces(*trial_cond_info[:-1], tp_df, results_dir,
#                             ft_reference_events, processes=processes)

#     # Success Matrices
#     if make_cond_succ_mat:
#         plot_cond_success_matrix([msession,], [mobject,], title=session,
#                                   cmap='Purples', savename=cond_succ_path)

#     # Acumulate msessions and mobjects for avg_cond_success_matrix
#     if make_avg_succ_mat:
#         plot_avg_cond_success_matrix(raw_ss, proc_ss, savename=avg_succ_path)

#     # Longitudenal performance
#     raw_server = os.path.dirname(raw_ss)
#     proc_server = os.path.dirname(proc_ss)
#     ref_date = date_from_folder(os.path.basename(raw_ss))

#     if make_perf_plot:
#         plot_performance(raw_server, proc_server, ref_date, savename=perf_path)

#     if make_perf10day_plot:
#         # Longitudenal performance (current - 10 days)
#         plot_performance(raw_server, proc_server, ref_date,
#                          savename=perf_minus10_day_path,
#                          lookback_timedelta=timedelta(days=10),
#                            include_last_tick=True)



# def get_sessions_from(server, sessions, from_session):
#     '''
#     from_session (str) : the folder name to parse the reference date from

#     Get all sessions with date greater than or equal to the reference date
#     returns full session paths
#     '''

#     if from_session == '':
#         return [os.path.join(server, sess) for sess in sessions]
#     else:
#         from_server_session = os.path.join(server, from_session)
#         assert os.path.isdir(from_server_session), f'From session not found: {from_server_session}'
#         total_sessions = len(sessions)
#         server_sessions = get_date_folders(from_server_session, mode='after')
#         rs(f'Processing {len(sessions)}/{total_sessions} sessions from {from_session} onwards')
#         return server_sessions



def transfer_to_training(preset, consider_for_transfer, overwrite=False, clean=True):

    import pdb; pdb.set_trace()

    # Check if training servers is defined
    if not does_trianing_servers_exist(preset):
        ws('WARNING: no training server location defined, not transfering session')
        return

    ## HACK clean up any training sessions that have already been transfered but that,
    # for some reason, still exist in sessions
    if clean:
        for raw_ss, proc_ss in consider_for_transfer:

            raw_training_dir = os.path.join(preset['default_training_server'], os.path.basename(raw_ss))
            if not os.path.isdir(raw_training_dir):
                #print(f'Could not find {raw_training_dir}, skipping')
                continue  ## skip if we don't have corresponding training dir

            #print(f'Considering removing {raw_ss}')

            # check raw session contents
            exp_contents = set([os.path.basename(pth) for pth in
                                glob.glob(os.path.join(raw_ss, '**'), recursive=True)])

            # see if they are a subset of raw_training contents
            train_contents = set([os.path.basename(pth) for pth in
                                  glob.glob(os.path.join(raw_training_dir, '**'), recursive=True)])

            if exp_contents <= train_contents:  ## Checks if experimental contents is a subset of training contents
                # If so delete the experimental contents as they have already been moved

                shutil.rmtree(raw_ss, ignore_errors=True)
                print(f'Attempted to remove {raw_ss}')

    rs('\n' * 2)
    transferred_pairs = []
    for raw_ss, proc_ss in tqdm.tqdm(consider_for_transfer, ncols=100, desc="Tranfering training sessions"):

        # Load meta information
        try:
            mstruct, _, _, _ = meta_session.load_meta_information(raw_ss, proc_ss)
        except Exception as e:
            ws('Could not load meta data from session {} ({}), skipping.'.format(raw_ss, repr(e)))
            continue

        # Check if it is an experiment or not
        experiment_expected_dirnames = [
            os.path.join(raw_ss, mstruct['videos_dir']),
            os.path.join(raw_ss, mstruct['raw_ps_dir'])
        ]
        is_training_session = not all([os.path.exists(dname) for dname
                                        in experiment_expected_dirnames])

        # Print a message if not
        sess_name = os.path.basename(raw_ss)
        if not is_training_session:
            rs(f'{sess_name} is an experiment session')
            continue

        # Do transfers
        rdst = os.path.join(preset['default_training_server'], sess_name)
        b_move_raw = os.path.isdir(raw_ss) and (overwrite or not os.path.isdir(rdst))
        if b_move_raw:
            premove_len = len(glob.glob(raw_ss, '**', recursive=True))

            # If we are overwriting, remove the old directory
            if overwrite:
                shutil.rmtree(rdst, ignore_errors=True)

            shutil.move(raw_ss, preset['default_training_server'])

            assert os.path.exists(rdst), 'Expected tranferred dir {} not found'.format(rdst)
            assert len(glob.glob(rdst)) == premove_len

            print('moved {raw_ss} to raw training server')

            if os.path.exists(raw_ss):
                print('Removing raw session {}'.format(raw_ss))
                shutil.rmtree(raw_ss, ignore_errors=True)

            transferred_pairs.append((raw_ss, rdst))

        pdst = os.path.join(preset['processed_training_server'], sess_name)
        b_move_proc = os.path.isdir(proc_ss) and (overwrite or not os.path.isdir(pdst))
        if b_move_proc:
            premove_len = len(glob.glob(proc_ss, '**', recursive=True))

            # If we are overwriting, remove the old directory
            if overwrite:
                shutil.rmtree(pdst, ignore_errors=True)

            shutil.move(proc_ss, preset['processed_training_server']) ## Note: dir moved INSIDE the destination path
            assert os.path.exists(pdst), 'Expected tranferred dir {} not found'.format(pdst)
            assert len(glob.glob(pdst)) == premove_len

            print('moved {proc_ss} to processed training server')

            if os.path.exists(proc_ss):
                shutil.rmtree(proc_ss, ignore_errors=True)

            transferred_pairs.append((proc_ss, pdst))


def main(preset, sessions, temp, overwrite, processes, dry_run=False):

    # Setup logging
    setup_logging(temp, sessions_dir=preset['processed_server'])

    if not os.path.exists(preset['default_server']):
        raise ValueError("Default server directory {} does not exist or is inaccessible.".format(preset['default_server']))

    experimental_ss_pairs_pre_move, _ = fetch_server_session_dirs(preset, sessions, filter=False)

    # Move sessions from experiment to training
    # Then perform move
    rs('\n'*3 + '='*200)
    rs('Moving training sessions')
    transfer_to_training(preset, experimental_ss_pairs_pre_move, overwrite)
    rs('\n'*3 + '='*200)

    experimental_ss_pairs, training_ss_pairs = fetch_server_session_dirs(preset, sessions, filter=True)

    # Process all sessions -- training and experiment
    if not dry_run:

        # process experiment sessions
        exp_session_wrappers = [SessionWrapper(*exp_pair) for exp_pair in experimental_ss_pairs]
        train_session_wrappers = [SessionWrapper(*train_pair) for train_pair in training_ss_pairs]

        # Experiment single sess analysis
        for exp_session in exp_session_wrappers:
            exp_session.plot_force_trace()
            exp_session.plot_cond_success_matrix()

        # Training single sess analysis
        # for train_session in train_session_wrappers:
        #     train_session.plot_force_trace()
        #     train_session.plot_cond_success_matrix()

        # # Experiment multi sess analysis
        # exp_group = SessionGroup(exp_session_wrappers)
        # exp_group.plot_avg_cond_success_matricies()
        # exp_group.plot_performance()

        # # Training multi sess analysis
        # exp_group = SessionGroup(train_session_wrappers)
        # exp_group.plot_avg_cond_success_matricies()
        # exp_group.plot_performance()

    else:
        rs('Skipping plot making because dry_run=True in main()')


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
        parser, ("sessions", "temp", "processes", "overwrite")
    )

    args = parser.parse_args(sys.argv[2:])

    start_time = time.time()

    main(
        current_preset,
        args.sessions,
        args.temp,
        args.overwrite,
        args.processes
    )

    rs('Program took {}.'.format(
        timedelta(seconds=time.time() - start_time)))