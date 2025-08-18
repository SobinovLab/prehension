#!python3
# -*- coding: utf-8 -*-
"""
Loading and saving a variety of files. Images are in a separate module

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
import csv
import re
import math

import numpy as np

import timed_sparse_matrix as tsm


def import_matrices(filename, equalize_period=-np.inf,
                    rowscols=None, zero_time_start=False, force_positive=False):
    '''Load a CSV or TSM matrix.

    Switches between TSM and CSV based on extension.

    Arguments:
        filename {str} -- [description]

    Keyword Arguments:
        equalize_period {float} -- subtract from the whole period maximum element value during the
            period from 0 to this value. (default: {-inf})
        rowscols [(int, int) or int or None] -- number of rows and columns in the matrix. If None,
            estimated from the first line. (default: {None})
        zero_time_start {bool} -- subtract the first value from the time series. (default: {False})
        force_positive {bool} -- makes all values less than 0 equal to 0. (default: {False})

    Returns:
        times -- [itime]
        matrices -- [itime][irow][icol]
    '''
    if os.path.splitext(filename)[1] == '.csv':
        times, matrices = import_csv_matrix_low(filename, rowscols=rowscols)
    elif os.path.splitext(filename)[1] == '.tsm':
        times, matrices = import_tsm_matrix(filename)
    else:
        raise ValueError('Unknown file extension: ', filename)

    # make times start from 0
    if zero_time_start:
        times = times - times[0]

    # equalize the beginning
    if equalize_period >= times[0]:
        rows = len(matrices[0])
        cols = len(matrices[0][0])
        equalizer_range = range(next(x for x, val in enumerate(times) if val > equalize_period))
        equalizer = []
        for irow in range(rows):
            equalizer.append([])
            for icol in range(cols):
                equalizer[irow].append(np.max([matrices[i][irow][icol] for i in equalizer_range]))

        for itime in range(len(times)):
            for irow in range(rows):
                for icol in range(cols):
                    matrices[itime][irow][icol] -= equalizer[irow][icol]

    if force_positive:  # does not seem to do anything, left from old stuff
        matrices[matrices < 0] = 0

    return times, matrices


def import_tsm_matrix(filename):
    times, matrices = tsm.load(filename)
    return times, matrices


def export_tsm_matrix(filename, times, matrices, type='stamps'):
    tsm.save(filename, type, times, matrices)


# to work with deprecated raw matrices
def import_csv_matrix(*args, **kwargs):
    return import_matrices(*args, **kwargs)


def export_csv(filename, column_names, values):
    '''Exports from a structure into a csv file.

    Arguments:
        filename {str} -- filename to write.
        column_names {list of M str} -- list of all column names.
        values {list [M][N]} -- list of all column values. First index corresponds to
            column number.
    '''
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)

        wrr.writerow(column_names)

        for itrial in range(len(values[0])):
            lo = [values[k][itrial] for k in range(len(column_names))]
            wrr.writerow(lo)


def import_csv(filename, cast=float):
    '''Imports csv into a simple structure.

    Arguments:
        filename {str} -- filename to import.
    Keyword Arguments:
        cast {callable} -- class of variables to return. Defaults to float.
    Returns a tuple of:
        column_names {list of M str} -- list of all column names.
        values {list [M][N] of cast type if possible, str otherwise} -- list of all column values.
            First index corresponds to column number.
    '''
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        line = next(rdr)
        column_names = [i.strip() for i in line]

        values = [[] for _ in column_names]

        for li in rdr:
            for idof, vdof in enumerate(li):
                try:
                    v = cast(vdof)
                except ValueError:
                    v = vdof
                values[idof].append(v)

    # clear empty
    for idof in reversed(range(len(column_names))):
        if (len(column_names[idof]) == 0 and
                all(isinstance(v, str) and len(v) == 0 for v in values[idof])):
            del column_names[idof]
            del values[idof]
    return column_names, values


def import_csv_as_dic(filename, cast=float):
    '''Imports csv into a simple structure. The column names are the dictionary keys.

    Arguments:
        filename {str} -- filename to import.
    Keyword Arguments:
        cast {callable} -- class of variables to return. Defaults to float.
    Returns a dictionary of:
        column_names: values {str: N of cast type if possible, str otherwise}
    '''
    return {cn: v for cn, v in zip(*import_csv(filename, cast=cast))}


def import_timed_csv(filename):
    '''Imports a csv and separates the time column.

    Arguments:
        filename {str} -- filename to import.
    Returns a tuple of:
        times {list of float} -- time points.
        names {list of M str} -- list of all column names.
        values {list [M][N] of float} -- list of all values.
            First index corresponds to column number.
    '''
    column_names, values = import_csv(filename)
    time_index = column_names.index('time')
    times = values[time_index]
    del column_names[time_index]
    del values[time_index]
    return times, column_names, values


def import_joint_angles(filename):
    '''Imports a csv and separates the time column.

    Arguments:
        filename {str} -- filename to import.
    Returns a tuple of:
        times {list of float} -- time points.
        ja_names {list of M str} -- list of all joint angle names.
        values {list [M][N] of float} -- list of all joint angle.
            First index corresponds to column number.
    '''
    return import_timed_csv(filename)


def import_torques(filename):
    '''Imports a csv and separates the time column.

    How many copies does one need?

    Arguments:
        filename {str} -- filename to import.
    Returns a tuple of:
        times {list of float} -- time points.
        ja_names {list of M str} -- list of all joint angle names.
        values {list [M][N] of float} -- list of all torques.
            First index corresponds to column number.
    '''
    return import_timed_csv(filename)


def import_csv_matrix_low(filename, rowscols=None):
    '''[summary]

    TODO probably should be rewritten using csv import functions

    The file is expected to be organized with the following columns:
    timestamp, row 1 column 1, row 2 column 1, .., row <rowcols[0]> column 1, ..,
        row <rowcols[0]> column <rowcols[1]>

    If rowcols is None, it is estimated from the name of the last element, which is assumed to be
    <TEXT>rowcols[0]<TEXT>rowcols[1]

    Arguments:
        filename {str} -- [description]

    Returns:
        times -- [itime]
        matrices -- [itime][irow][icol]
    '''
    times = []
    matrices = []
    with open(filename, 'r') as fin:
        rdr = csv.reader(fin)

        li = next(rdr)

        if rowscols is None:
            # extract number of rows and columns from the last element of the first line
            rows, cols = [int(i) for i in re.findall('[0-9]+', li[-1])]
        else:
            if isinstance(rowscols, (list, tuple)):
                rows = rowscols[0]
                cols = rowscols[1]
            else:
                rows = rowscols
                cols = rowscols

        # check
        if not len(li) == rows*cols + 1:
            raise ValueError(('Number of rows {} and columns {} is inconsistent with the number of'
                              ' columns in CSV file {}.').format(rows, cols, len(li)))

        for li in rdr:
            times.append(float(li[0]))
            matrices.append(np.zeros((rows, cols)))
            for irow in range(rows):
                for icol in range(cols):
                    matrices[-1][irow][icol] = float(li[1+irow+rows*icol])
    times = np.array(times)
    matrices = np.array(matrices)

    return times, matrices


def export_csv_matrix(filename, times, matrices):
    rows = np.shape(matrices)[1]
    cols = np.shape(matrices)[2]

    m_new = np.zeros((len(times), cols, rows))
    for i, m in enumerate(matrices):
        m_new[i] = m.transpose()

    matrices = m_new.reshape((len(times), rows*cols)).transpose()

    values = np.concatenate((times.reshape((1, len(times))), matrices), axis=0)
    column_names = ['times'] + ['r{}c{}'.format(r+1, c+1)
                                for c in range(cols)
                                for r in range(rows)]

    export_csv(filename, column_names, values)


def export_one_csv_matrix(filename, matrix):
    with open(filename, 'w', newline='') as f:
        wrr = csv.writer(f)

        for row in matrix:
            wrr.writerow(row)


def import_one_csv_matrix(filename, dtype=float):
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        values = []
        for li in rdr:
            values.append([dtype(v) for v in li])
    return values


def import_matched_contacts(filename):
    matched_contacts = []
    with open(filename, 'r') as fin:
        rdr = csv.reader(fin)

        for li in rdr:
            matched_contacts.append({})
            for mc in li:
                if len(mc) == 0:
                    continue
                indices, hand_segment, dist = mc.split(':')
                r, c = indices.split('.')
                if hand_segment not in matched_contacts[-1].keys():
                    matched_contacts[-1][hand_segment] = []
                matched_contacts[-1][hand_segment].append((int(r), int(c), float(dist)))
    return matched_contacts


def dic_from_csv(fname, keyword, value, key_cast=None, value_cast=None):
    '''Imports two columns from a CSV file as a dictionary.

    Arguments:
        fname {str} -- CSV filename.
        keyword {str} -- name of the column to use as a keyword of the dictionary. Should be unique
            in the CSV. If not unique, the dictionary will return the last 'value' corresponding to
            the repeating keyword.
        value {str} -- name of the column to use as a value of the dictinary.

    Keyword Arguments:
        key_cast {function or class} -- a cast to apply to the element from the 'keyword' column to
            create the key of the dictionary, for example 'key_cast=int'.
            (default: lambda x: x.strip())
        value_cast {function or class} -- a cast to apply to the element from the 'value' column to
            create the value for the dictionary, for example 'value_cast=float'.
            (default: lambda x: x.strip())

    Returns:
        dict -- returns the element from the 'value' column in response to the element
    '''
    if key_cast is None:
        def key_cast(x):
            return x.strip()
    if value_cast is None:
        def value_cast(x):
            return x.strip()

    dic = {}
    with open(fname, 'r') as f:
        fd = csv.DictReader(f)
        for li in fd:
            dic[key_cast(li[keyword])] = value_cast(li[value])

    return dic



def dic_to_csv(fname, dic, column_names=None):
    '''Writes a dictionary into a csv.

    Arguments:
        fname {str} -- CSV filename.
        dic {dict} -- dic with values. No check on values is performed.
    Keyword Arguments:
        column_names {None or list of str} -- the first line of the csv (nothing if None).
    '''
    with open(fname, 'w', newline='') as f:
        fw = csv.writer(f)

        if column_names is not None:
            fw.writerow(column_names)

        for k, v in dic.items():
            fw.writerow([k, v])


def load_roms(filename, dof_names=None):
    '''Reads ranges of motion from a file.

    Depending on input either returns a list of dof_names and their ranges [min, max], or just
    the ranges.
    '''
    column_names, values = import_csv(filename)

    i_dofname = column_names.index('dof_name')
    i_rmin = column_names.index('range_min')
    i_rmax = column_names.index('range_max')
    if 'rotation' in column_names:
        i_rot = column_names.index('rotation')
    else:
        i_rot = -1

    if dof_names is None:
        ranges = [[rmin, rmax] for rmin, rmax in zip(values[i_rmin], values[i_rmax])]
        if i_rot >= 0:
            return values[i_dofname], ranges, values[i_rot]
        else:
            return values[i_dofname], ranges

    ranges = []
    rots = []
    for dof_name in dof_names:
        i_dof = values[i_dofname].index(dof_name)
        ranges.append([values[i_rmin][i_dof], values[i_rmax][i_dof]])
        if i_rot >= 0:
            rots.append(values[i_rot][i_dof])
    if i_rot >= 0:
        return ranges, rots
    else:
        return ranges


def import_triangulated_csv(filename):
    '''Returns data as dictionary:
        bodypart: nFrames X 3
    '''
    data = {}
    with open(filename, 'r') as f:
        rdr = csv.reader(f)

        li1 = next(rdr)
        li2 = next(rdr)

        # read the csv
        frame_numbers = []
        data_raw = [[] for _ in range(len(li1) - 1)]
        for li in rdr:
            frame_numbers.append(int(li[0]))
            for i, el in enumerate(li[1:]):
                data_raw[i].append(float(el) if el != '' else math.nan)

    # transform into a dictionary
    for i, (i1, i2) in enumerate(zip(li1[1:], li2[1:])):
        v = data.get(i1, [])
        v.append(data_raw[i])
        data[i1] = v

    # transpose the dictionary
    for k in data.keys():
        data[k] = list(zip(*data[k]))

    return frame_numbers, data
