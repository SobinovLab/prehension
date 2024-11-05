#!python3
# -*- coding: utf-8 -*-
"""
Loading and saving files in OpenSim formats.

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
import csv
import math
import ntpath

from .logs import ws


def import_mot(fname):
    '''Import OpenSim motion file into a python structure.

    Arguments:
        fname {str} -- motion file.

    Returns a list:
        dof_names {list of str} -- names of DOFs.
        times {list of numbers} -- time series.
        dofs {list} -- each item corresponds to values for that DOF for each frame.
            dofs[iDOF][iTime]
    '''
    with open(fname, 'r') as f:
        rdr = csv.reader(f, dialect='excel-tab')

        li = next(rdr)
        while len(li) < 1 or not li[0].strip().lower() == 'time':
            li = next(rdr)

        dof_names = [i.strip() for i in li[1:]]

        times = []
        dofs = [[] for _ in dof_names]

        for li in rdr:
            times.append(float(li[0]))
            for idof, vdof in enumerate(li[1:]):
                dofs[idof].append(float(vdof))
    return dof_names, times, dofs


def export_mot(fname, dof_names, times, dofs):
    '''Exports python structures into a motion file for OpenSim.

    Arguments:
        fname {str} -- filename of the mot file to output into.
        dof_names {list of str} -- each element is the DOF string name.
        times {list of numbers} -- time series.
        dofs {list} -- each item corresponds to values for that DOF for each frame.
    '''
    with open(fname, 'w', newline='') as f:
        wrr = csv.writer(f, dialect='excel-tab')

        wrr.writerow(['Coordinates'])
        wrr.writerow(['version=1'])
        wrr.writerow(['nRows={}'.format(len(times))])
        wrr.writerow(['nColumns={}'.format(len(dof_names) + 1)])
        wrr.writerow(['inDegrees=yes'])
        wrr.writerow([])
        wrr.writerow(['Units are S.I. units (second, meters, Newtons, ...)'])
        wrr.writerow(['Angles are in degrees.'])
        wrr.writerow([])
        wrr.writerow(['endheader'])
        wrr.writerow(['time'] + dof_names)

        for itime, time in enumerate(times):
            wrr.writerow([time] + [dof_vals[itime] for dof_vals in dofs])


def import_trc(filename):
    '''Import OpenSim trc file into a Python structure format.

    Arguments:
        filename {str} -- trc file name.
    Output:
        bodyparts {list of str} -- names of markers.
        frame_numbers {list of ints} -- Frame # column.
        times {list of floats} -- Time column
        points {array NFrames X NBodyparts X 3} -- [iframe][ibodypart][0:x,1:y,2:z]
        rate {float} -- DataRate.
        units {str} - units of the data.
    '''
    with open(filename, 'r') as fin:
        rdr = csv.reader(fin, delimiter='\t', dialect='excel-tab')

        li = next(rdr)  # flavor text
        li = next(rdr)  # flavor text

        li = next(rdr)
        if not li[0] == li[1] or not li[0] == li[5]:
            ws('DataRate, CameraRate or OrigDataRate do not match. Using DataRate.')
        rate = float(li[0])
        units = li[4]

        li = next(rdr)
        bodyparts = li[slice(2, len(li), 3)]

        li = next(rdr)
        li = next(rdr)

        frame_numbers = []
        times = []
        points = []
        for li in rdr:
            if len(li) == 0:
                continue
            frame_numbers.append(int(li[0]))
            times.append(float(li[1]))
            points.append([])
            for ibp in range(len(bodyparts)):
                points[-1].append([])
                if li[2 + ibp * 3] == '':
                    points[-1][-1].append(math.nan)
                    points[-1][-1].append(math.nan)
                    points[-1][-1].append(math.nan)
                else:
                    points[-1][-1].append(float(li[2 + ibp * 3]))
                    points[-1][-1].append(float(li[2 + ibp * 3 + 1]))
                    points[-1][-1].append(float(li[2 + ibp * 3 + 2]))
    return bodyparts, frame_numbers, times, points, rate, units


def export_trc(filename, bodyparts, points, rate, frame_numbers=None, times=None, units='mm'):
    '''Exports marker data into OpenSim-compatible trc file.

    Arguments:
        filename {str} -- output file name.
        bodyparts {list of str} -- names of markers.
        points {array NFrames X NBodyparts X 3} -- [iframe][ibodypart][0:x,1:y,2:z]
        rate {float} -- DataRate.
    Keyword Arguments:
        frame_numbers {list of ints} -- Frame # column. If None, generated from rate and length of
            points starting at 1.
        times {list of floats} -- Time column. If None, generated from rate and length of
            points starting at 0.
        units {str} - units of the data. {default: 'mm'}
    '''
    if frame_numbers is None:
        frame_numbers = list(range(1, len(points) + 1))
    if times is None:
        period = 1.0 / rate
        times = [i * period for i in range(len(frame_numbers))]

    n_bodyparts = len(bodyparts)
    n_frames = len(frame_numbers)

    with open(filename, 'w', newline='') as fou:
        wrr = csv.writer(fou, delimiter='\t', dialect='excel-tab')

        # header
        wrr.writerow(['PathFileType', '4', '(X/Y/Z)', ntpath.basename(filename)])
        wrr.writerow(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate',
                      'OrigDataStartFrame', 'OrigNumFrames'])
        wrr.writerow([rate, rate, n_frames, n_bodyparts, units, rate, 1, 1])

        # bodyparts
        lo = ['Frame#', 'Time']
        for bp in bodyparts:
            lo += [bp, '', '']
        wrr.writerow(lo)

        # XYZ columns
        lo = ['', '']
        for ibp in range(n_bodyparts):
            lo += ['X{}'.format(ibp+1), 'Y{}'.format(ibp+1), 'Z{}'.format(ibp+1)]
        wrr.writerow(lo)
        wrr.writerow([])  # necessary

        # data
        for frame_number, time, point in zip(frame_numbers, times, points):
            lo = [frame_number, time]
            for ibp in range(n_bodyparts):
                if math.isnan(point[ibp][0]):
                    lo += ['', '', '']
                else:
                    lo += point[ibp]

            # OpenSim4.0 cannot read the line properly when the last value is
            # empty and wants an additional tab:
            if lo[-1] == '':
                lo.append('')

            wrr.writerow(lo)
