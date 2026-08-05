#!python3
# -*- coding: utf-8 -*-
"""
Filtering and resampling data.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/SobinovLab/prehension

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

from . import misc


########### Slices
def get_slice_to_time_base(tmin, n_times, times):
    '''Returns a slice with n_times length that starts at tmin in n_times. '''
    # tmin must be within [times[0], times[1]] period
    # otherwise it will throw an exception
    start = next(x for x, val in enumerate(times) if val >= tmin)
    return slice(start, start + n_times)


def get_slice_around_point(times, point, b_tp, a_tp):
    '''Returns a slice that matches the period [point-b_tp;point+a_tp] in times.'''
    np_times = np.array(times)
    fr = misc.find_first((np_times - (point - b_tp)) >= 0)
    to = misc.find_last((np_times - (point + a_tp)) <= 0)
    return slice(int(fr), int(to))


def enforce_rom(dof, rng, padding=0):
    '''Enforces that the values in 'dof' are within the range specified by 'rng'.

    Padding moves them further away from the edge.
    '''
    rng_min = rng[0] + padding * (rng[1] - rng[0])
    rng_max = rng[1] - padding * (rng[1] - rng[0])
    dof[dof < rng_min] = rng_min
    dof[dof > rng_max] = rng_max
    return dof


########### General filters and resampling
def downsample_at_timeseries(times, data, times_new):
    '''Downsamples data (for example, pressure sensors) to a different set of times.

    times_new has to have a much lower frequency. Neither have to be uniform.
    times_new cannot be wider than times'''
    times = np.array(times)
    data = np.array(data)
    times_new = np.array(times_new)

    data_new = []
    diff_times_new_hvd = np.diff(times_new) / 2
    t_froms = np.insert(times_new[1:] - diff_times_new_hvd, 0, times_new[0])
    t_tos = np.insert(times_new[:-1] + diff_times_new_hvd, len(diff_times_new_hvd), times_new[-1])
    for t_from, t_to in zip(t_froms, t_tos):
        slc = np.logical_and(times >= t_from, times < t_to)
        if sum(slc) == 0:
            warnings.warn('downsample_at_timeseries: No corresponding interval found.')
            data_new.append(np.zeros(np.shape(data)[1:]))
        else:
            data_new.append(np.median(data[slc], axis=0))

    # # testing
    # plt.figure()
    # plt.plot(times, reduce_force_matrices(data), 'k')
    # plt.plot(times_new, reduce_force_matrices(data_new), 'r')
    # plt.show()

    return np.array(data_new)


def downsample(ps_times, data, ja_period):
    '''Downsamples pressure sensor data (for example) to joint angle frequency

    It is better to use the instead of time-based median filter bc sensor times are not consistent.
    See downsample_at_timeseries function
    '''
    ps_period = np.median(np.diff(ps_times))

    # downsample
    q = int(round(ja_period / ps_period))
    data = scipy.signal.decimate(data, q, axis=0, ftype='fir')

    ps_times_new = np.arange(ps_times[0], ps_times[-1], ja_period)

    # HACK sometimes there is an off-by-one error for decimate requiring q to be an integer
    if len(ps_times_new) > len(data):
        data = np.append(data, [data[-1]], axis=0)

    return ps_times_new, data


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
        vals_to_filt = input_vector[idx-kernel_offset:idx+kernel_offset+1]
        num_nans = sum(np.isnan(vals_to_filt))
        if num_nans < kernel_offset:
            output_vector[idx] = np.nanmedian(vals_to_filt)

    return output_vector


########### Causal smoothing
def causal_halfgaussian_kernel(sigma_s, bin_width, n_sd=4):
    '''Return a normalized causal half-gaussian smoothing kernel (in bins).

    The kernel spans the current bin and the preceding ~n_sd*sigma bins (no future
    bins), so filtering with it never uses activity from after a bin -- appropriate
    for a read-out that sweeps forward through time.  The weights are the right half
    of a Gaussian (index 0 = current bin) normalized to sum to 1.  A non-positive
    sigma degenerates to a pass-through kernel ([1.0]).
    '''
    sigma_bins = float(sigma_s) / float(bin_width)
    if sigma_bins <= 0:
        return np.array([1.0])
    half_len = int(np.ceil(n_sd * sigma_bins))
    delays = np.arange(0, half_len + 1)
    kernel = np.exp(-0.5 * (delays / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def apply_causal_filter(rate, kernel):
    '''Causally smooth a 1-D rate trace with `kernel` (index 0 weights the current bin).

    Output bin i is sum_d kernel[d] * rate[i - d] for d >= 0, i.e. only the present
    and past bins contribute; the trace length is preserved.
    '''
    return np.convolve(rate, kernel, mode='full')[:len(rate)]


############## Deprecated
def reduce_force_matrices(matrices, reduction=np.sum):
    '''Consider replacing with np.sum(matrices, axis=(1, 2)) on numpy versions >= 1.7.0'''
    red = []
    for matrix in matrices:
        red.append(reduction(matrix))
    return red
