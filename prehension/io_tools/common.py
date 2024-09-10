#!python3.12
import os
import csv
import re
import numpy as np
import tsm


def import_matrices(filename, equalize_period=-np.inf,
                    rowscols=None, zero_time_start=False, force_positive=True):
    '''[summary]

    Switches between TSM and CSV based on extension.

    Arguments:
        filename {str} -- [description]

    Keyword Arguments:
        equalize_period {float} -- subtract from the whole period maximum element value during the
            period from 0 to this value. (default: {-inf})
        rowscols [(int, int) or int or None] -- number of rows and columns in the matrix. If None,
            estimated from the first line. (default: {None})
        zero_time_start {bool} -- subtract the first value from the time series. (default: {False})
        force_positive {bool} -- makes all values less than 0 equal to 0. (default: {True})

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
        key_cast = lambda x: x.strip()
    if value_cast is None:
        value_cast = lambda x: x.strip()

    dic = {}
    with open(fname, 'r') as f:
        fd = csv.DictReader(f)
        for l in fd:
            dic[key_cast(l[keyword])] = value_cast(l[value])

    return dic


def load_roms(filename, dof_names=None):
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
