#!python3
# -*- coding: utf-8 -*-
"""
Filtering and resampling data.

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
import warnings
import numpy as np
import scipy


def downsample_at_timeseries(times, data, times_new):
    '''times_new has to have a much lower frequency. Neither have to be uniform.
    times_new cannot be wider than times'''
    times = np.array(times)
    data = np.array(data)
    times_new = np.array(times_new)

    data_new = []
    diff_times_new_hvd = np.diff(times_new) / 2
    t_froms = np.insert(
        times_new[1:] - diff_times_new_hvd, 0, times_new[0])
    t_tos = np.insert(
        times_new[:-1] + diff_times_new_hvd, len(diff_times_new_hvd), times_new[-1])
    for t_from, t_to in zip(t_froms, t_tos):
        slc = np.logical_and(times >= t_from, times < t_to)
        if sum(slc) == 0:
            warnings.warn(
                'downsample_at_timeseries: No corresponding interval found.')
            data_new.append(np.zeros(np.shape(data)[1:]))
        else:
            data_new.append(np.median(data[slc], axis=0))

    # # testing
    # plt.figure()
    # plt.plot(times, reduce_force_matrices(data), 'k')
    # plt.plot(times_new, reduce_force_matrices(data_new), 'r')
    # plt.show()

    return np.array(data_new)


# TODO instead of decimate use time-based median filter bc sensor times are not consistent
def downsample(ps_times, data, ja_period):
    '''Downsamples pressure sensor data to joint angle frequency'''
    ps_period = np.median(np.diff(ps_times))

    # downsample
    q = int(round(ja_period / ps_period))
    data = scipy.signal.decimate(data, q, axis=0, ftype='fir')

    ps_times_new = np.arange(ps_times[0], ps_times[-1], ja_period)

    # HACK sometimes there is an off-by-one error for decimate requiring q to be an integer
    if len(ps_times_new) > len(data):
        data = np.append(data, [data[-1]], axis=0)

    return ps_times_new, data


def get_slice_to_time_base(tmin, n_times, times):
    # tmin must be within [times[0], times[1]] period
    # otherwise it will throw an exception
    start = next(x for x, val in enumerate(times) if val >= tmin)
    return slice(start, start + n_times)


def enforce_rom(dof, rng):
    dof[dof < rng[0]] = rng[0]
    dof[dof > rng[1]] = rng[1]
    return dof


def reduce_force_matrices(matrices, reduction=np.sum):
    '''Consider replacing with np.sum(matrices, axis=(1, 2)) on numpy versions >= 1.7.0'''
    red = []
    for matrix in matrices:
        red.append(reduction(matrix))
    return red


def nanmedianfilt(input_vector, kernel_width):
    '''Median filter that ignores nan values'''
    if kernel_width % 2 == 0:
        kernel_width = kernel_width + 1

    kernel_offset = int((kernel_width-1)/2)

    output_vector = np.empty(input_vector.shape)
    output_vector.fill(np.nan)

    init_idx = int(np.ceil(kernel_width/2))
    term_idx = int(len(input_vector) - np.ceil(kernel_width/2))

    output_vector[:init_idx] = input_vector[:init_idx]
    output_vector[term_idx:] = input_vector[term_idx:]
    for idx in np.arange(init_idx, term_idx):
        vals_to_filt = input_vector[idx -
                                    kernel_offset:idx+kernel_offset+1]
        num_nans = sum(np.isnan(vals_to_filt))
        if num_nans < kernel_offset:
            output_vector[idx] = np.nanmedian(vals_to_filt)

    return output_vector
