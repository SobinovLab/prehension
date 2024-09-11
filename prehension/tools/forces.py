#!python3
# -*- coding: utf-8 -*-
"""
Commonly used functions for loading and processing forces.

Copyright (C) 2019-2024 Anton Sobinov
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

import numpy as np

from . import constants
from . import io
from .logs import rs, ws


def get_matched_contact_frame_mask(exp, mcf, frame_size):
    frame_mask = np.zeros(frame_size, dtype=bool)
    for k, v in mcf.items():
        if exp(k):
            for ve in v:
                frame_mask[ve[0]][ve[1]] = True
    return frame_mask


def load_maps(trial):
    if os.path.exists(trial.lps_map_filename):
        lps_digit_mask = np.array(
            io.import_one_csv_matrix(trial.lps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    if os.path.exists(trial.rps_map_filename):
        rps_digit_mask = np.array(
            io.import_one_csv_matrix(trial.rps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    return lps_digit_mask, rps_digit_mask


def load_forces(mstruct, trial):
    ps_matrices = {}
    matched_contacts = {}

    for ps_name in mstruct['ps_dic'].keys():
        ps_times, ps_matrices[ps_name] = io.import_matrices(
            trial.get_post_ps_filenames()[ps_name])
        matched_contacts[ps_name] = io.import_matched_contacts(
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
            auto_mask = (len(constants.DIGITS) - 1) * np.ones(np.shape(manual_digit_map))
            for i_digit, d in enumerate(constants.DIGITS.values()):
                if i_digit == len(constants.DIGITS) - 1:
                    break
                digit_auto_mask = get_matched_contact_frame_mask(
                    d['exp'], matched_contacts[ps_name][i_frame],
                    np.shape(manual_digit_map))
                auto_mask[digit_auto_mask] = i_digit

            # diff mask
            diff_mask = np.not_equal(auto_mask, manual_digit_map)
            ps_matrix_frame = ps_matrices[ps_name][i_frame]
            mask_based_diff_per_sensor[ps_name].append(np.sum(np.abs(ps_matrix_frame[diff_mask])))

            # manual unclaimed mask
            unclaimed_mask = manual_digit_map == (len(constants.DIGITS) - 1)
            unclaimed_force[ps_name].append(np.sum(np.abs(ps_matrix_frame[unclaimed_mask])))

        mask_based_diff_per_sensor[ps_name] = np.array(mask_based_diff_per_sensor[ps_name])
        unclaimed_force[ps_name] = np.array(unclaimed_force[ps_name])
    # sum across sensors
    trial.mask_based_diff = (mask_based_diff_per_sensor['medial_sensor'] +
                             mask_based_diff_per_sensor['lateral_sensor'])
    trial.unclaimed_force = (unclaimed_force['medial_sensor'] +
                             unclaimed_force['lateral_sensor'])



def get_summed_force_data(tsm1_file, tsm2_file, verbose=False):
    # Handle different start/end times
    # load time and ps data for each file
    ps_times1, ps_matrices1 = io.import_tsm_matrix(tsm1_file)  # import_matrices(tsm1_file)
    ps_times2, ps_matrices2 = io.import_tsm_matrix(tsm2_file)  # import_matrices(tsm2_file)

    # Get the sums at each timestep
    ps_sum1 = np.sum(ps_matrices1, axis=(1, 2))
    ps_sum2 = np.sum(ps_matrices2, axis=(1, 2))

    # MEM ISSUE FIX
    ps_matrices1 = None
    ps_matrices2 = None

    # Sanity check that time and pressure data are the same size
    if ps_times1.size != ps_sum1.size:
        raise ValueError("Size of tsm1 time and pressure data not equal")
    if ps_times2.size != ps_sum2.size:
        raise ValueError("Size of tsm2 time and pressure data not equal")

    # find smallest common time range
    tmin = max([ps_times1[0], ps_times2[0]])
    tmax = min([ps_times1[-1], ps_times2[-1]])

    # trim both datasets to that range
    valid_idx1 = (ps_times1 >= tmin) & (ps_times1 <= tmax)
    ps_times1 = np.array(ps_times1[valid_idx1])
    ps_sum1 = np.array(ps_sum1[valid_idx1])

    valid_idx2 = (ps_times2 >= tmin) & (ps_times2 <= tmax)
    ps_times2 = np.array(ps_times2[valid_idx2])
    ps_sum2 = np.array(ps_sum2[valid_idx2])

    # Take union of times to get all common times
    U_times = np.union1d(ps_times1, ps_times2)
    max_size = max(ps_times1.size, ps_times2.size)
    added_times_pct = abs(U_times.size - max_size) / max_size
    if added_times_pct > 0.05:
        if verbose:
            ws(f"Times union size is greater 5% of the max \
            ps_times array: pct diff = ({added_times_pct})")

    # Interp the missing sums
    ps_sum1_fill = np.interp(U_times, ps_times1, ps_sum1)
    ps_sum2_fill = np.interp(U_times, ps_times2, ps_sum2)

    if ps_sum1_fill.size != ps_sum2_fill.size:
        raise ValueError(
            f"Len of ps sum 1 and 2 not equal "
            f"({ps_sum1_fill.size} / {ps_sum2_fill.size})"
        )

    # Left/Right force sums
    fss_total = np.sum([ps_sum1_fill, ps_sum2_fill], axis=0)

    return (
        U_times,
        fss_total
    )
