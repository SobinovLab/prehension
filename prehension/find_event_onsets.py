#!python3
# -*- coding: utf-8 -*-
"""
Finds events based on forces and kinematics if available and aligns them to the neural data.

Copyright (C) 2023-2024 Caleb Raman, Anton Sobinov
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
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas
import reporting_pool
import tqdm
from matplotlib.colors import ListedColormap
from scipy.signal import argrelmin

from . import meta_session
from .tools import logs
from .tools import io
from .tools.logs import rs, ws
from .tools.forces import get_summed_force_data

# ============================================ Notes ============================================= #
# Lint with: py -3.7 -m pycodestyle find_event_onsets.py --max-line-length 100 --ignore E402
# - trial_id - same as meta_session
# - shoulder_movement_onset - onset of movement of shoulder joints
# - elbow_movement_onset
# - wrist_movement_onset
# - fingers_movement_onset
# - maximum_aperture - we will have to try a few, but probably start
#     with the sum of index and thumb joints as approximation of aperture
# - grasp_start - use the same threshold as is written out in `align_`
#     script, it was something like 1% of 95th percentile
# - fingers_static - when fingers stop moving after grasp starts
# - release_start - when fingers leave the static posture. These two
#     might be hard to get reliably. If it is too hard, we will not have them.
#     Do not focus on them in the start.
# - release - same as grasp_start approach, just the last point
# - hand_retreated - when the arm returns to the resting position
#     (shoulder and elbow kinematics close to beginning)
# - (boolean) regrasp - should be 1 if the force drops to 0 at any
#     point between grasp and release.

# =========================================== Examples =========================================== #
# Example call:
# py find_event_onsets.py cr_test --server "C:\PrehensionDataLocal\MojitoRightHemisphere"
# --sessions 2022_04_19_Set1 --overwrite --processes 1

# Server:
# py find_event_onsets.py cr_test --server
# "S:\ProjectFolders\Prehension\Data\MojitoRightHemisphere\sessions"
#  --sessions 2022_04_27_Set1 --trials 34 --overwrite

# py find_event_onsets.py cr_test --server
#  "S:\ProjectFolders\Prehension\Data\MojitoRightHemisphere\sessions"
#  --sessions 2022_04_27_Set1 --overwrite --processes 1 --make_plots


# =========================================== Constants ========================================== #
# figure size
FIGSIZE = (15, 10)

# Min grasp time (for filtering grasp events)
MIN_GRASP_TIME_S = 0.2  # seconds

# Must exceed value of max force for a given trial to be valid
LOWER_FORCE_THRESHOLD = 1

# Fing static thresh
FING_STATIC_JA_THRESH = 0.4

# SHOULDER, ELBOW, WRIST, FINGER velocity thresholds
VELOCITY_THRESH_SHOULDER = 0.25
VELOCITY_THRESH_ELBOW = 0.25
VELOCITY_THRESH_FINGER = 0.25
VELOCTIY_THRESH_WRIST = 0.25

# ONSET / OFFSET thresholds
ONSET_FORCE_THRESH = 0.05
OFFSET_FORCE_THRESH = 0.05

# JA Constants
PREGRASP_POSITION_JAS = [
    "ra_sh_elv_angle",
    "ra_sh_elv",
    "ra_sh_rot",
]
SHOULDER_COLS = PREGRASP_POSITION_JAS
ELBOW_COLS = ["ra_el_e_f"]
WRIST_COLS = [
    "ra_wr_sup_pro",
    "ra_wr_rd_ud",
    'ra_wr_e_f'
]
FINGER_COLS = [
    "ra_cmc1_f_e",
    "ra_cmc1_opp",
    "ra_cmc1_ad_ab",
    "ra_mcp1_e_f",
    "ra_ip1_e_f",
    "ra_mcp2_e_f",
    "ra_mcp2_ad_ab",
    "ra_pip2_e_f",
    "ra_dip2_e_f",
    "ra_mcp3_e_f",
    "ra_mcp3_rd_ud",
    "ra_pip3_e_f",
    "ra_dip3_e_f",
    "ra_mcp4_e_f",
    "ra_mcp4_ad_ab",
    "ra_pip4_e_f",
    "ra_dip4_e_f",
    "ra_mcp5_e_f",
    "ra_mcp5_ad_ab",
    "ra_pip5_e_f",
    "ra_dip5_e_f",
]
THUMB_COLS = [
    "ra_cmc1_opp",
    "ra_cmc1_f_e",
]
INDEX_COLS = [
    "ra_mcp2_e_f"
]

TIMEPOINT_LABELS = [
    'shoulder_onset',  # shoulder_movement_onset,
    'elbow_onset',  # elbow_movement_onset,
    'wrist_onset',  # wrist_movement_onset,
    'finger_onset',  # finger_movement_onset
    'MGA_time',  # max aperture time
    'first_grasp_start',  # grasp_start
    'fingers_static',  # fingers static
    'release_start',  # release start
    'release',  # release
    'hand_retreat_time',  # hand retreated
    # Successful (as determined by meta ttl times)
    'success_grasp_start',
    'success_grasp_end',  # See above
    'regrasp_bool'  # regrasp (bool)

]


# ========================================= Functions ============================================ #
def get_empty_plot_dict():
    return {
        'normed_force_data': None,
        'time_ax': None,
        'grasp_pairs': None
    }


def create_timepoints_dict(trial_number):
    return dict([('trial_number', trial_number)] + [(k, np.nan) for k in TIMEPOINT_LABELS])


def norm_array(arr):
    """
    Normalize the elements of a NumPy array to the range [0, 1].

    Args:
        arr (numpy.ndarray): The input NumPy array to be normalized.

    Returns:
        numpy.ndarray: The normalized array.
    """
    arr -= np.nanmin(arr)
    mx = np.nanmax(arr)
    if mx != 0:
        arr /= mx
    return arr


def find_local_minima(x, y):
    """
    Find local minima in a dataset.

    Args:
        x (iterable): The x-values of the dataset.
        y (iterable): The y-values of the dataset.

    Returns:
        tuple: Two lists - local_min_x (x-values of local minima) and local_min_y
        (corresponding y-values).

    Raises:
        AssertionError: If the lengths of x and y are not the same.
    """

    local_min_x = []
    local_min_y = []
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length")
    # Exclude the endpoints because they can't be local minima
    for i in range(1, len(y) - 1):
        if y[i] < y[i - 1] and y[i] < y[i + 1]:
            local_min_x.append(x[i])
            local_min_y.append(y[i])
    return local_min_x, local_min_y


def find_local_maxima(x, y):
    """
    Find local maxima in a dataset.

    Args:
        x (iterable): The x-values of the dataset.
        y (iterable): The y-values of the dataset.

    Returns:
        tuple: Two lists - local_max_x (x-values of local maxima) and local_max_y
        (corresponding y-values).

    Raises:
        AssertionError: If the lengths of x and y are not the same.
    """
    local_max_x = []
    local_max_y = []
    if len(x) != len(y):
        raise ValueError("Arrays must have the same length")
    # Exclude the endpoints because they can't be local maxima
    for i in range(1, len(y) - 1):
        if y[i] > y[i - 1] and y[i] > y[i + 1]:
            local_max_x.append(x[i])
            local_max_y.append(y[i])
    return local_max_x, local_max_y


def get_abs_normed_velocity(position_data):
    """
    Calculate the absolute, normalized velocity of a position data sequence.

    Args:
        position_data (iterable): The sequence of position data.

    Returns:
        numpy.ndarray: An array representing the absolute, normalized velocity of the position data.

    """
    vels = np.diff(position_data)
    vels = np.abs(vels)
    vels -= np.min(vels)

    if np.max(vels) > 0:  # Should almost always be the case
        # Normalize so each value is weighted the same
        vels /= np.max(vels)

    return vels


def find_velocity_threshold_crossing_time(time_ax, ja_columns, thresh_dec, ax=None,
                                          title=None, col_names=None, from_minima=False):
    """
    Create velocity data from a set of joint-angle columns per timestep.
    Return a list of pairs of points that cross above and below the given threshold of the max.

    Args:
        time_ax (iterable): discrete timepoints in seconds.
        ja_columns (iterable of str): the column names that we wish to include in our total
        velocity curve
        thresh_dec (float -> [0, 1]): The target decimal threshold with which to determine
        crossing points.

    Returns:
        tuple: A tuple (cross_above, cross_below) where cross_above is the list of (x, y) points
        where the velocity
        curve crosses above the threshold and cross_below are is the list of (x, y) points where
        the velocity curve
        crosses below the threshold.
    """

    n_timesteps = len(time_ax)

    # Check each input col is acutally a numpy array
    if not np.all([isinstance(ja_col, np.ndarray) for ja_col in ja_columns]):
        raise TypeError("Joint angle column is not a numpy array")

    # Check that the len of each column is the same as timesteps
    if not np.all([len(ja_col) == n_timesteps for ja_col in ja_columns]):
        raise ValueError("Joint angle column length not equal to the number of timesteps")

    # Get normed velocity and time
    ja_abs_normed_vels = np.array([get_abs_normed_velocity(col) for col in ja_columns])
    ja_abs_normed_vels_sum = np.sum(ja_abs_normed_vels, axis=0)

    # Was: # np.percentile(ja_abs_normed_vels_sum, 95) -> before
    ja_abs_normed_vels_sum /= np.max(ja_abs_normed_vels_sum)
    # Get the velocity adjusted time axis
    time_ax_vel = time_ax[:-1] + (0.5 * np.median(np.diff(time_ax)))

    if not len(time_ax_vel) == len(ja_abs_normed_vels_sum):
        raise ValueError(
            "Time axis and ja velocities not equal length")

    # PLOTTING
    should_plot = ax is not None
    # If fig and ax provided add to plot
    if should_plot:
        legend_labels = [col_name for col_name in col_names]

        # Plot each column individually
        for col in ja_abs_normed_vels:
            ax.plot(time_ax_vel, col)

        ax.set_yticks([])
        if title is not None:
            ax.set_title(title)

        # Plot the total summed velocity
        ax.plot(time_ax_vel, ja_abs_normed_vels_sum, color='red', linestyle='--')
        ax.axhline(y=thresh_dec, color='gray', linestyle='--')

    # if from_minima is true, only consider range from closest prior local min (below thresh)
    if from_minima:

        # Sort local minima indices in descending order
        minima_indices = np.sort(
            argrelmin(ja_abs_normed_vels_sum)[0])[::-1]

        # Find greatest index corresponding to a value below threshold
        ll_idx = 0
        for idx in minima_indices:
            if ja_abs_normed_vels_sum[idx] < thresh_dec:
                ll_idx = idx
                break

        if should_plot:
            ax.scatter(time_ax_vel[minima_indices],
                       ja_abs_normed_vels_sum[minima_indices],
                       label='Local mins',
                       marker='x')
            ax.axvline(x=time_ax_vel[ll_idx], linestyle='--')

        # Now shape time axis and joint angle velocities accordingly
        # If it is stil -1 then it will just be the whole range
        time_ax_vel = time_ax_vel[ll_idx:]
        ja_abs_normed_vels_sum = ja_abs_normed_vels_sum[ll_idx:]

    cross_above, cross_below = find_threshold_crossing_points(
        time_ax_vel, ja_abs_normed_vels_sum, threshold=thresh_dec
    )

    if should_plot:
        # Last plotting step
        # Create a legend outside the plot

        # Trim legend labels if more than 9
        if len(legend_labels) > 9:
            legend_labels = legend_labels[:9]

        ax.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

        if len(cross_above) > 0:
            ax.axvline(x=cross_above[0][0],
                       linestyle='--', color='red')

    return cross_above, cross_below


def find_threshold_crossing_points(x_data, y_data, threshold):
    """
    Find points at which a dataset crosses a specified threshold value.

    Args:
        x_data (iterable): The x-values of the dataset.
        y_data (iterable): The y-values of the dataset.
        threshold (float): The threshold value to check for crossings.

    Returns:
        tuple: Two lists - cross_above (points where y_data crosses above the threshold)
               and cross_below (points where y_data crosses below the threshold).

    Raises:
        ValueError: If the crossing does not actually cross the specified threshold.

    Example:
    >>> x_values = [1, 2, 3, 4, 5]
    >>> y_values = [3, 2, 5, 7, 1]
    >>> threshold = 4
    >>> above, below = find_threshold_crossing_points(x_values, y_values, threshold)
    >>> print(above)
    [(3.0, 4), (4.0, 7)]
    >>> print(below)
    [(2.0, 5)]
    """

    post_cross_indices = np.where(np.diff((y_data > threshold).astype(int)))[0] + 1

    cross_above = []
    cross_below = []

    for i in post_cross_indices:

        # Using numpy interp
        # (flip the xp and yp because we want to evaluate at the threhshold on the y axis)
        yp = [x_data[i - 1], x_data[i]]
        xp = [y_data[i - 1], y_data[i]]

        cross_pt = (np.interp(threshold, xp, yp), threshold)

        if y_data[i] >= threshold and y_data[i - 1] <= threshold:
            cross_above.append(cross_pt)
        elif y_data[i] <= threshold and y_data[i - 1] >= threshold:
            cross_below.append(cross_pt)
        else:
            raise ValueError("Crossing does not cross threshold")

    return cross_above, cross_below


def get_max_thumb_index_aperture(df, time_window, ax=None):
    time_min, time_max = time_window

    if time_min >= time_max:
        raise ValueError(
            f"Time min > time max {time_min} > {time_max}")

    times = df["time"].values
    valid_idx = (times >= time_min) & (times <= time_max)

    if np.sum(valid_idx) == 0:
        raise ValueError(
            f"No valid indices found for window = {time_window}")

    times = times[valid_idx]

    # Plot index and middle sum norm abs vel
    # pos_middle = df['ra_mcp3_e_f'].values
    pos_index = df["ra_mcp2_e_f"].values[valid_idx]
    pos_thumb = df["ra_mcp1_e_f"].values[valid_idx] * -1

    # The valleys are where the aperture is biggest
    # (we plot the flipped data, but use this to find the local minima, aka max aperture point)
    thumb_index_abs_diffs = np.abs(np.subtract(pos_thumb, pos_index))
    thumb_index_diff = 1 - norm_array(thumb_index_abs_diffs)

    maxx_arr, maxy_arr = find_local_maxima(times, thumb_index_diff)

    MGA = np.nan
    t_MGA = np.nan
    if len(maxx_arr) > 0:
        max_idx = np.argmax(
            maxy_arr
        )  # The index of the minimum angle difference (actually the max but it's inverted)
        t_MGA = maxx_arr[max_idx]
        MGA = thumb_index_abs_diffs[max_idx]

    # Debug Plotting
    if ax is not None:
        ax.set_title("Maximum grasp aperture time")
        ax.set_yticks([])
        ax.plot(times, thumb_index_diff,
                label="Thumb index difference")
        ax.axvline(x=t_MGA, color='red',
                   linestyle='--', label="MGA Time")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

    return t_MGA, MGA



def merge_pairs(point_list, x_tolerance):
    """
    Merge pairs of points if the x-coordinate difference between consecutive points is within
    a specified tolerance.

    Args:
        point_list (list of tuples): A list of point pairs, where each pair is represented as
        ((x1, y1), (x2, y2)).
        x_tolerance (float): The maximum allowed x-coordinate difference for merging pairs.

    Returns:
        list of tuples: A list of merged point pairs.

    Example:
    >>> points = [((1, 2), (3, 4)), ((4.5, 5), (6, 7)), ((8, 9), (10, 11))]
    >>> tolerance = 2.0
    >>> merged = merge_pairs(points, tolerance)
    >>> print(merged)
    [((1, 2), (6, 7)), ((8, 9), (10, 11))]
    """
    merged_pairs = []

    # Sort the list of pairs by the x-coordinate of the first point
    sorted_point_list = sorted(
        point_list, key=lambda pair: pair[0][0])
    current_pair = sorted_point_list[0]

    for next_pair in sorted_point_list[1:]:
        if next_pair[0][0] - current_pair[1][0] <= x_tolerance:
            # Merge the pairs by taking the first point of the current pair
            # and the second point of the next pair
            current_pair = (current_pair[0], next_pair[1])
        else:
            merged_pairs.append(current_pair)
            current_pair = next_pair

    merged_pairs.append(current_pair)
    return merged_pairs


def find_grasp_events(
    time_ax,
    tsmSumsNormed,
    onset_thresh,
    offset_thresh,
    min_grasp_time_s=None,
    merge_gaps_less_than=None,
):
    """
    Find grasp events in a dataset of force measurements.

    Args:
        tsm1_file (str): The file path for the first TSM file (e.g., left hand).
        tsm2_file (str): The file path for the second TSM file (e.g., right hand).
        onset_thresh (float, optional): The onset threshold for detecting grasp events.
        Default is 0.1.
        offset_thresh (float, optional): The offset threshold for ending grasp events.
        Default is 0.02.
        min_grasp_time_s (float, optional): The minimum duration (in seconds) for a grasp event
        to be considered valid. Default is None.
        LR_active_thresh (float, optional): The threshold for sensor activity in the left and right
        hands. Default is None.
        merge_gaps_less_than (float, optional): Merge grasp events if they are closer than this time
        (in seconds). Default is None.

    Returns:
        tuple: A tuple containing grasp onsets (in seconds), grasp durations (in seconds), a figure,
        and an axis for visualization.

    Example:
    >>> tsm1_file = "tsm_left.dat"
    >>> tsm2_file = "tsm_right.dat"
    >>> onset_threshold = 0.1
    >>> offset_threshold = 0.02
    >>> min_grasp_duration = 2.0
    >>> grasp_onsets, grasp_durations, fig, ax = find_grasp_events(tsm1_file, tsm2_file,
                                              onset_threshold, offset_threshold, min_grasp_duration)
    >>> print(grasp_onsets)
    [2.0, 5.0, ...]
    >>> print(grasp_durations)
    [3.0, 4.0, ...]
    """

    # Find where we cross above onset threshold
    grasp_cross_above, _ = find_threshold_crossing_points(time_ax, tsmSumsNormed, onset_thresh)

    # Find where we cross below offset threshold
    _, grasp_cross_below = find_threshold_crossing_points(time_ax, tsmSumsNormed, offset_thresh)

    # Remove crosses below if they happen before the first cross above
    if len(grasp_cross_above) > 0:
        grasp_cross_below = [gcb for gcb in grasp_cross_below if gcb[0] > grasp_cross_above[0][0]]
    else:
        grasp_cross_below = []

    # Remove crosses above if they happen after the last cross below
    if len(grasp_cross_below) > 0:
        grasp_cross_above = [gca for gca in grasp_cross_above if gca[0] < grasp_cross_below[-1][0]]
    else:
        grasp_cross_above = []

    if not len(grasp_cross_above) == len(grasp_cross_below):
        raise ValueError(
            "The on-off grasp events are of different lengths")

    grasp_pairs = [
        (grasp_cross_above[i], grasp_cross_below[i]) for i in range(len(grasp_cross_above))
    ]

    # 'Merge' two grasp events if they are close together
    if merge_gaps_less_than:
        grasp_pairs = merge_pairs(grasp_pairs, merge_gaps_less_than)

    # Get grasp durations
    grasp_onsets = np.array([pt[0][0] for pt in grasp_pairs])
    grasp_durations = np.array(
        [(pt[1][0] - pt[0][0]) for pt in grasp_pairs])

    # Weed out any events whose duration falls under the min_grasp_time_s threshold
    if min_grasp_time_s is not None:
        # Create a boolean mask based on the condition
        valid_mask = grasp_durations >= min_grasp_time_s

        # Filter the arrays using the mask
        grasp_pairs = [pair for pair, is_valid in zip(grasp_pairs, valid_mask) if is_valid]
        grasp_onsets = grasp_onsets[valid_mask]
        grasp_durations = grasp_durations[valid_mask]

    return grasp_onsets, grasp_durations, grasp_pairs


def get_fingers_static_on_off(df, grasp_start, grasp_release, vel_threshold, vel_columns, ax=None):
    """
    Find the times when finger velocities dip below a certain
    threshold within a specified time window.

    Args:
        df (pandas.DataFrame): The DataFrame containing velocity data and a 'time' column.
        time_window (tuple): A tuple (start, end) defining the time window to search for static
          fingers.
        vel_threshold (float): The velocity threshold for detecting finger staticity.
        vel_columns (list of str): A list of column names containing finger velocities.
        ax (matplotlib.Axes, optional): The axes for plotting if provided. Default is None.

    Returns:
        tuple: A tuple containing the time when fingers become static (fingers_static) and the
        time when fingers begin to release (begin_release).

    Example:
    >>> import pandas as pd
    >>> data = {'time': [1, 2, 3, 4, 5], 'finger1_vel': [0.1, 0.2, 0.05, 0.02, 0.1],
    >>>          'finger2_vel': [0.08, 0.15, 0.03, 0.01, 0.1]}
    >>> df = pd.DataFrame(data)
    >>> time_range = (2, 4)
    >>> velocity_threshold = 0.1
    >>> velocity_columns = ['finger1_vel', 'finger2_vel']
    >>> fingers_static, begin_release = get_fingers_static_on_off(df, time_range,
    velocity_threshold, velocity_columns)
    >>> print(fingers_static)
    3
    >>> print(begin_release)
    4
    """

    # Find when the finger velocities are below a certain threshold within the time window
    times = df["time"].values[:-1] + (0.5 * np.median(np.diff(df["time"].values)))
    # Get the summed velocities over the whole time frame
    vels = norm_array(np.sum([np.abs(np.diff(df[col])) for col in vel_columns], axis=0))

    # --- Fingers Static --- #
    # Now find where the summed velocities dip below a threshold pct of the GLOBAL max
    idx_fing_static = (times >= grasp_start) & (
        times <= grasp_start + 0.5)
    # Finger static should happen within 0.5 s of grasp start
    _, cross_below = find_threshold_crossing_points(
        times[idx_fing_static], vels[idx_fing_static], vel_threshold * np.max(vels)
    )

    fingers_static = np.nan
    if len(cross_below) > 0:
        fingers_static = cross_below[0][0]

    # --- Begin Release --- #
    idx_begin_release = (times >= grasp_release - 0.5) & (times <= grasp_release)
    # Finger static should happen within 0.5 s of grasp start
    cross_above, _ = find_threshold_crossing_points(
        times[idx_begin_release], vels[idx_begin_release], vel_threshold * np.max(vels)
    )

    begin_release = np.nan
    if len(cross_above) > 0:
        # NOTE we take the LAST cross above the threshold in the grasp duration
        begin_release = cross_above[-1][0]

    # Debug plot
    if ax is not None:
        ax.set_yticks([])
        ax.plot(times[idx_fing_static], vels[idx_fing_static])
        ax.plot(times[idx_begin_release], vels[idx_begin_release])
        ax.set_title("Finger static period")
        ax.axhline(y=vel_threshold * np.max(vels), color='gray', linestyle='--')
        ax.axvline(x=fingers_static, linestyle='--', color='red', label='finger static')
        ax.axvline(x=begin_release, linestyle='--', color='green', label='release')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)

    return fingers_static, begin_release


def find_return_to_init_position(pregrasp_movement_window_s, postgrasp_return_window_s,
                                 timepoints_d, last_grasp_end, df, time_ax, ax):

    init_pos_cols = [
        df[col_name].values for col_name in PREGRASP_POSITION_JAS]

    def get_vector(t):
        if not (t >= time_ax[0] and t <= time_ax[-1]):
            raise ValueError("Time argument is out of range")
        return np.array([np.interp(t, time_ax, ja_col) for ja_col in init_pos_cols])

    P0 = get_vector(timepoints_d['first_grasp_start'] - pregrasp_movement_window_s)

    search_window = (last_grasp_end, last_grasp_end + postgrasp_return_window_s)
    search_i = np.where((time_ax > search_window[0]) & (time_ax < search_window[1]))
    search_times = time_ax[search_i]

    # This is the vector difference from the initial position to time t
    # We want to plot this if fig and ax are provided
    Dt = np.array([np.sqrt(np.sum((P0 - get_vector(t))**2)) for t in search_times])
    loc_mins_x, loc_mins_y = find_local_minima(search_times, Dt)

    # Find the lowest local min
    if len(loc_mins_y) > 0:
        i_min = np.argmin(loc_mins_y)
        timepoints_d["hand_retreat_time"] = loc_mins_x[i_min]

    # Plotting for debug plot
    if (ax is not None):
        ax.plot(search_times, Dt, label='Difference from t0')
        ax.set_yticks([])
        ax.set_title('Return to pregrasp state')
        ax.axvline(timepoints_d["hand_retreat_time"], label='Hand retreat time',
                   linestyle='--', color='red')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=3)


def find_finger_onset(timepoints_d, df, pre_grasp_idx, times_pre, ax=None):
    finger_cols = [df[col_name].values[pre_grasp_idx] for col_name in FINGER_COLS]
    finger_vel_cross_above, _ = find_velocity_threshold_crossing_time(
        times_pre, finger_cols, VELOCITY_THRESH_FINGER, ax, 'finger onset', FINGER_COLS,
        from_minima=True
    )

    if len(finger_vel_cross_above) > 0:
        timepoints_d['finger_onset'] = finger_vel_cross_above[0][0]
    else:
        # TODO reintroduce with verbosity level
        # ws("Can't compute finger onset because no velocity threshold exceed point")
        pass


def find_wrist_onset(timepoints_d, df, pre_grasp_idx, times_pre, ax=None):
    wrist_cols = [df[col_name].values[pre_grasp_idx] for col_name in WRIST_COLS]
    wrist_vel_cross_above, _ = find_velocity_threshold_crossing_time(
        times_pre, wrist_cols, VELOCTIY_THRESH_WRIST, ax, "wrist onset", WRIST_COLS,
        from_minima=True
    )

    if len(wrist_vel_cross_above) > 0:
        timepoints_d['wrist_onset'] = wrist_vel_cross_above[0][0]
    else:
        # TODO reintroduce with verbosity level
        # ws("Can't compute wrist onset because no velocity threshold exceed point")
        pass


def find_elbow_onset(timepoints_d, df, pre_grasp_i, times_pre, ax=None):
    el_cols = [df[col_name].values[pre_grasp_i]
               for col_name in ELBOW_COLS]
    elbow_vel_cross_above, _ = find_velocity_threshold_crossing_time(
        times_pre, el_cols, VELOCITY_THRESH_ELBOW, ax, "elbow onset", ELBOW_COLS,
        from_minima=True
    )

    if len(elbow_vel_cross_above) > 0:
        timepoints_d['elbow_onset'] = elbow_vel_cross_above[0][0]
    else:
        # TODO reintroduce with verbosity level
        # ws("Can't compute elbow onset because no velocity threshold exceed point")
        pass


def find_shoulder_onset(timepoints_d, df, pre_grasp_idx, times_pre, ax=None):

    sh_cols = [df[col_name].values[pre_grasp_idx] for col_name in SHOULDER_COLS]

    if not np.all([len(col) == len(times_pre) for col in sh_cols]):
        raise ValueError("Length of joint angle data is not equal to the pre grasp time data")

    shoulder_vel_cross_above, _ = find_velocity_threshold_crossing_time(
        times_pre, sh_cols, VELOCITY_THRESH_SHOULDER, ax, "shoulder onset",
        SHOULDER_COLS, from_minima=True
    )

    timepoints_d['shoulder_onset'] = (
        shoulder_vel_cross_above[0][0] if len(shoulder_vel_cross_above) > 0 else np.nan
    )


def find_trial_timepoints(
    trial,
    make_trial_plots,
    pregrasp_movement_window_s=1,
    postgrasp_return_window_s=3,
):
    """
    Create a list of timepoints for a single trial (TrialInfo) to track various events and onsets.

    make this so we only throw exceptions if something is really whack,
    else continue and just print a warning
    No message returned (1 return value)

    """

    # Define return value
    timepoints_d = create_timepoints_dict(trial.trial_number)

    # Additional data for plotting later
    plot_addons = get_empty_plot_dict()

    # ================= GRASP START/END ================== #
    keys = set(trial.transformed_ps_filenames.keys())

    # W1: Check if pressure sensor data is not found
    expected_ps_files = [
        trial.transformed_ps_filenames[k] for k in keys]
    if np.any([not os.path.exists(k) for k in expected_ps_files]):
        ws(f"trial {trial.trial_number}: No pressure sensor data found. Searched"
           f" {expected_ps_files} Skipping.")
        return timepoints_d, plot_addons, None

    # Get the normalized summed L/R force
    time_ax, tsmSums = get_summed_force_data(
        *[trial.transformed_ps_filenames[k] for k in keys])

    # Check if the max force is super low
    if np.max(tsmSums) <= LOWER_FORCE_THRESHOLD:
        # TODO enable when verbosity is introduced
        # ws(f"trial {trial.trial_number}:f_max < thres({LOWER_FORCE_THRESHOLD:.2f} N) Skipping.")
        return timepoints_d, plot_addons, None

    # Get normalized force sum (important to do this after checking the force threshold above)
    tsmSumsNormed = norm_array(tsmSums)
    plot_addons['normed_force_data'] = tsmSumsNormed
    plot_addons['time_ax'] = time_ax

    grasp_onsets, grasp_durations, grasp_pairs = find_grasp_events(
        time_ax,
        tsmSumsNormed,
        ONSET_FORCE_THRESH,
        OFFSET_FORCE_THRESH,
        min_grasp_time_s=MIN_GRASP_TIME_S,
    )

    plot_addons['grasp_pairs'] = grasp_pairs

    num_grasps = len(grasp_onsets)
    if num_grasps == 0:
        # TODO reintroduce with verbosity level
        # ws(f"trial {trial.trial_number}: No grasp events found. Skipping trial.")
        return timepoints_d, plot_addons, None

    last_grasp_end = grasp_onsets[-1] + grasp_durations[-1]

    timepoints_d['first_grasp_start'] = grasp_onsets[0]
    timepoints_d['release'] = grasp_onsets[-1] + grasp_durations[-1]
    timepoints_d['regrasp_bool'] = len(grasp_durations) > 1

    # CR instead of taking the first force threshold crossing to be the time onset
    # Find force thresh crossings the contain the success grasp
    meta_success_time = np.nan
    if 'ttl_to_success_grasp' in trial.other_info.keys():
        meta_success_time = trial.other_info['ttl_to_success_grasp']

    for pair in grasp_pairs:
        if pair[0][0] <= meta_success_time and pair[1][0] >= meta_success_time:
            timepoints_d['success_grasp_start'] = pair[0][0]
            timepoints_d['success_grasp_end'] = pair[1][0]
            break

    # Check if we have the joint angles csv
    kin_file = trial.post_kinematic_filename_csv
    if not os.path.exists(kin_file):
        return timepoints_d, plot_addons, None

    df = pandas.read_csv(kin_file)
    times = df["time"].values

    # -- movement onsets -- #
    debug_fig = None
    axs = [None, None, None, None, None, None, None]
    if make_trial_plots:
        debug_fig, axs = plt.subplots(
            7, 1, sharex=True, figsize=FIGSIZE)
        debug_fig.suptitle(f"Trial {trial.trial_number} debug plot")
        axs[-1].set_xlabel('Seconds since TTL')

    pre_grasp_idx = np.where(
        (times < timepoints_d['first_grasp_start']) & (
            times > timepoints_d['first_grasp_start'] - pregrasp_movement_window_s)
    )
    times_pre = df["time"].values[pre_grasp_idx]

    if len(times_pre) < 1:
        raise ValueError("No pregrasp time period found.")

    # Shoulder onset
    find_shoulder_onset(
        timepoints_d, df, pre_grasp_idx, times_pre, axs[0])

    # Elbow onset
    find_elbow_onset(timepoints_d, df, pre_grasp_idx,
                     times_pre, axs[1])

    # Wrist onset
    find_wrist_onset(timepoints_d, df, pre_grasp_idx,
                     times_pre, axs[2])

    # Finger onset
    find_finger_onset(timepoints_d, df,
                      pre_grasp_idx, times_pre, axs[3])

    # Return to init position.
    find_return_to_init_position(pregrasp_movement_window_s, postgrasp_return_window_s,
                                 timepoints_d, last_grasp_end, df, times, axs[4])

    # Max grasp aperture
    MGA_window = (timepoints_d['first_grasp_start'] - 0.5, timepoints_d['first_grasp_start'])
    timepoints_d['MGA_time'], _ = get_max_thumb_index_aperture(df, MGA_window, axs[5])

    # Finger static & release start
    # TODO do we use end of first grasp or end of last grasp
    args = (df, timepoints_d['first_grasp_start'],
            timepoints_d['first_grasp_start'] + grasp_durations[0],
            FING_STATIC_JA_THRESH, FINGER_COLS, axs[6])
    timepoints_d['fingers_static'], timepoints_d['release_start'] = get_fingers_static_on_off(*args)

    if make_trial_plots:
        plt.tight_layout()

    return timepoints_d, plot_addons, debug_fig


def create_plot_from_dictionary(timepoints_d, trial,
                                grasp_pairs=None, on_off_thresholds=None,
                                time_ax=None, normed_force_data=None):
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # Choose a colormap
    cmap = plt.get_cmap('brg')
    sub_d = {k: timepoints_d[k] for k in timepoints_d.keys() - {'trial_number', 'regrasp_bool'}}
    # Filter out any nans
    # pdb.set_trace()
    sub_d = {k: v for k, v in sub_d.items() if not np.isnan(v)}
    # add meta_session timepoints
    for tk, tv in trial.other_info.items():
        if tk[:7] == 'ttl_to_':
            sub_d['meta_' + tk[7:]] = tv
    # Set time limits on plot
    vals = list(sub_d.values())

    if len(vals) == 0:
        plt.close(fig)
        return

    t1 = np.nanmin(vals) * 0.95
    t2 = np.nanmax(vals) * 1.05
    ax.set_xlim(t1, t2)
    ax.set_xlabel('Time from TTL (s)')

    msg = ''
    if (normed_force_data is None and time_ax is None and grasp_pairs is None):
        msg = '(Missing: force and grasp events)'
    ax.set_title(
        f"Trial {timepoints_d['trial_number']} Timepoints {msg}")

    # Plot all existing timepoints in the trial
    for i, (k, v) in enumerate(sub_d.items()):
        clr = cmap(i / len(sub_d))
        ax.axvline(x=v, linestyle='--', label=k, color=clr)
        ax.annotate(k, [v, 0.5 + 0.5 * (len(sub_d) - i) / len(sub_d)],
                    color=clr, ha='left', va='top')

    if normed_force_data is not None and time_ax is not None:
        ax.plot(time_ax, normed_force_data, label='Normalized force data', color='blue')
        ax.set_ylabel('Normalized total force (au)')

    # Optional but nice to have plotting only available if not run in 'create from csv' mode
    if grasp_pairs is not None:
        kwargs = {'color': 'beige',
                  'alpha': 0.8,
                  'label': 'grasp_event'}
        for pt in grasp_pairs:
            ax.scatter(*pt[0], color="green")
            ax.scatter(*pt[1], color="red")
            ax.fill_between([pt[0][0], pt[1][0]], 0, 1,
                            **kwargs)
            if 'label' in kwargs.keys():
                # So we don't relabel the grasp events
                del kwargs['label']

    if on_off_thresholds is not None:
        onset_thresh, offset_thresh = on_off_thresholds
        ax.axhline(y=onset_thresh, linestyle="--", color="gray")
        ax.axhline(y=offset_thresh, linestyle="--", color="gray")

    fig.legend()

    return fig


def find_event_onsets(preset, sessions, trials_sel, temp, overwrite,
                      processes, make_plots, store_plots, make_trial_plots, show_plots):
    """Outputs a csv of movement onset times for each session.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed
            trials.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        processes {int} --- Number of parallel processes in the pool.
        make_plots {bool} --- Makes some inspection figures.
        store_plots {bool} --- Save plots to disk.
        make_trial_plots {bool} --- Makes more inspection figures.
        show_plots {bool} --- Show the generated plots.
        preset {dict} --- Preset dictionary.

    Example:
    >>> server_directory = "/data/trials"
    >>> session_directories = ["session1", "session2"]
    >>> temp_dir = "/temp/logs"
    >>> overwrite_output = True
    >>> num_processes = 4
    >>> create_plots = True
    >>> find_event_onsets(server_directory, session_directories, temp_dir, overwrite_output,
        num_processes, create_plots)
    """

    proc_dir = preset['processed_server']
    raw_dir = preset['default_server']

    logs.setup_logging(temp, sessions_dir=proc_dir)

    if not os.path.exists(raw_dir):
        raise ValueError("Server directory {} does not exist or is inaccessible.".format(raw_dir))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(raw_dir)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    if make_trial_plots:
        ws('Plots for individual trials selected, restricting to ONE process.')
        processes = 1

    # sort
    sessions.sort()
    rs("Found {} sessions: {}".format(
        len(sessions), ", ".join(sessions)))

    failed_trial_reports = []

    # For each session
    for session in tqdm.tqdm(sessions, ncols=100, desc="Sessions"):
        print()
        rs("Processing session {}.".format(session))
        raw_ss = os.path.join(raw_dir, session)
        proc_ss = os.path.join(proc_dir, session)

        if not os.path.exists(raw_ss):
            ws("Session {} does not exist on the server.".format(raw_ss))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(
                raw_ss,
                proc_ss,
            )
        except meta_session.IncompleteMetaError as imfe:
            ws(f'Skipping {raw_ss} due to incomplete meta: {imfe}')
            continue
        except Exception as e:
            ws('Could not load meta data from session {} ({}), skipping.'.format(session, repr(e)))
            continue

        # If store_plots create the timepoints plots directory
        directory_path = None
        if store_plots:
            directory_path = mstruct['timepoint_plots_dir']
            os.makedirs(directory_path, exist_ok=True)

        # if running a subset of trials, then save to a temporary file which is deleted at the end
        output_csv = mstruct['timepoint_csv_filename']
        if len(trials_sel) > 0:
            rs('Only a subset of trials was selected, the results will not be written into a file.')

        # accumulate trials
        if len(trials_sel) == 0:  # default behavior
            if not os.path.exists(output_csv) or overwrite:
                trials = msession
            else:
                trials = []
        else:
            trials = [
                t for t in msession if t.trial_number in trials_sel]

        # Just continue if no trials
        if not trials:
            continue

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(t.trial_number) for t in trials])))

        # FIND THE TIMEPOINTS
        df = None
        plot_kwargs = None
        debug_figs = None

        if len(trials) > 0:
            p_args = list(zip(*[
                trials,
                [make_trial_plots] * len(trials)
            ]))

            if len(p_args) > 0:
                pool = reporting_pool.ReportingPool(
                    find_trial_timepoints,
                    p_args,
                    processes=processes,
                    report_on_change=True,
                    track_failures=True,
                )

                # If this fails the result will be None for a row.
                # If this is the case we want to just fill in a blank row with trial number.
                pool_results = pool.start()

                # Handle exceptions
                if len(pool.failed_i_jobs) > 0:
                    print()
                    ws("Failed to find timepoints for the following trials:")
                    for v in pool.failed_i_jobs:
                        ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                        failed_trial_reports.append('session {} trial {} error: {}'.format(
                            session, trials[v].trial_number, pool.error_reports[v]))

                # Extract plot kwargs
                plot_kwargs = {elem[0]['trial_number']: elem[1]
                               for elem in pool_results if elem is not None}

                debug_figs = {elem[0]['trial_number']: elem[2]
                              for elem in pool_results if elem is not None}

                # 1. Check if we found anything and if so write new csv
                if len(pool_results) > 0:
                    # Replace the Nones with empty pool_results
                    timepoints = []
                    for pr, trial in zip(pool_results, trials):
                        if pr is None:
                            timepoints.append(create_timepoints_dict(trial.trial_number))
                        else:
                            timepoints.append(pr[0])

                    df = pandas.DataFrame(timepoints, columns=timepoints[0].keys())
                    # Write to csv only if the whole session was processed
                    if len(trials_sel) == 0:
                        df.to_csv(output_csv, index=False)

        # MAKE REPORTS AND PLOTS
        # If the data has not been created this cycle, check if we have a csv to plot from and
        # load it
        if df is None:
            if os.path.exists(output_csv):
                df = pandas.read_csv(output_csv)
            else:
                # If no data was generated and nothing to load, go to next session
                continue

        # print percent success
        portion_success = sum(
            [t.success for t in msession]) / len(msession)
        print(f"Success:\t {portion_success:.2%} trials")

        for column in df.columns:
            if column == 'trial_number':
                continue
            non_nan_count = df[column].notna().sum()
            percentage_non_nan = (non_nan_count / len(df))

            # Align the output using string formatting
            print(f"\t{column:20s}: {percentage_non_nan:.2%} trials")

        # Heatmap
        if make_plots and (store_plots or show_plots):
            # Create a heatmap showing missing values across all trials in the session
            df2 = df.iloc[:, 1:]
            binary_array = ~df2.isnull()

            # Create a colormap with two distinct colors
            cmap = ListedColormap(['black', 'white'])

            # Set a larger figure size and make it square
            heatfig = plt.figure(figsize=(8, 11))

            # Create a heatmap using matplotlib with equal aspect ratio
            plt.imshow(binary_array, cmap=cmap, aspect='auto')

            # Adjust spacing around the subplots
            plt.subplots_adjust(
                left=0.1, right=0.9, top=0.8, bottom=0.2)

            # 0 at the bottom
            plt.gca().invert_yaxis()

            # Customize the plot
            plt.title('Binary Heatmap - Null Values Black. Session {}.'.format(session))
            plt.xlabel('Timepoints')
            plt.ylabel('Trials')

            # Set x ticks and labels
            plt.xticks(range(len(df2.columns)), df2.columns, rotation='vertical')

            if store_plots and len(trials_sel) == 0:
                plt.savefig(os.path.join(mstruct['timepoint_plots_dir'], 'Session_Heatmap.png'))

            if not show_plots:
                plt.close(heatfig)

        # Trial plots
        if make_trial_plots and (store_plots or show_plots):
            for row in tqdm.tqdm(
                    df.itertuples(index=False, name=None), desc="Building plots", total=len(df)):

                row_d = dict(zip(df.columns, row))
                trial_number = row_d['trial_number']
                plot_kwarg = (plot_kwargs[trial_number]
                              if trial_number in plot_kwargs.keys() else {})

                debug_fig = (debug_figs[trial_number]
                             if trial_number in plot_kwargs.keys() else None)

                fig = create_plot_from_dictionary(
                    row_d,
                    meta_session.find_trial(
                        msession,
                        trial_number),
                    on_off_thresholds=None,
                    **plot_kwarg)

                if store_plots:
                    fig.savefig(os.path.join(mstruct['timepoint_plots_dir'],
                                             f"EventsPlot_{trial_number}.png"))
                    if debug_fig is not None:
                        debug_fig.savefig(os.path.join(mstruct['timepoint_plots_dir'],
                                                       f"DebugPlot_{trial_number}.png"))

                if not show_plots:
                    plt.close(fig)
                    plt.close(debug_fig)

    if show_plots:
        plt.show()
