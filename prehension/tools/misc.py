#!python3
# -*- coding: utf-8 -*-
"""
Miscellaneous functions.

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
import re
import datetime

import numpy as np


# @https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
def find_first(x):
    idx = x.view(bool).argmax() // x.itemsize
    return idx if x[idx] else -1


def find_last(x):
    ff = find_first(np.flip(x))
    if ff == -1:
        return -1
    return len(x) - ff - 1


def session_to_date(session):
    """Helper function to get the date from the session name and the set number.

    Returns date string, datetime object, and set number.
    """
    # Regex to capture the date part (YYYY_MM_DD) and optional SetN part
    match = re.match(r"(\d{4}_\d{2}_\d{2})(?:_Set(\d+))?", session)

    if match:
        # Extract the date string and convert to datetime
        date_str = match.group(1)
        current_date = datetime.datetime.strptime(date_str, r"%Y_%m_%d")

        # Extract the set number if available, otherwise return None
        if match.group(2):
            set_number = int(match.group(2))
        else:
            set_number = None

        return date_str, current_date, set_number

    else:
        raise ValueError(f'Could not parse date from session name: {session}')


def sessions_to_dates(sessions):
    '''Groups sessions by dates they were performed'''
    s2d = {s: session_to_date(s)[0] for s in sessions}
    days = {d: [] for d in s2d.values()}
    for s, d in s2d.items():
        days[d].append(s)
    return days


def trailing_int(name):
    """Trailing integer of a name ('experiment2' -> 2, 'recording10' -> 10).

    Returns None when the name has no trailing digits.
    """
    digits = ''
    for ch in reversed(name):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    return int(digits) if digits else None


def get_field(ev, possible_names):
    '''First present field of a numpy structured array by candidate names.

    Returns the field array for the first name in ``possible_names`` present in
    ``ev.dtype.names``, or None if none match.
    '''
    names = ev.dtype.names or ()
    for name in possible_names:
        if name in names:
            return ev[name]
    return None


def offset_array(arr, delta):
    '''arr + delta as a float array, or None when arr is None.'''
    if arr is None:
        return None
    return np.asarray(arr, dtype=float) + delta


def concat_arrays(arrays):
    '''Concatenate the non-None arrays as floats; None if there are none.'''
    present = [np.asarray(a, dtype=float) for a in arrays if a is not None]
    if not present:
        return None
    return np.concatenate(present)


def unmatched_mask(values, reference, tol):
    '''Boolean mask over ``values``: True where no ``reference`` entry is within ``tol``.

    Both inputs are 1-D numeric arrays in the same units; a True entry marks a value
    whose nearest reference (by absolute difference) is farther than ``tol`` away.
    '''
    values = np.asarray(values, dtype=float)
    reference = np.sort(np.asarray(reference, dtype=float))
    if values.size == 0:
        return np.zeros(0, dtype=bool)
    if reference.size == 0:
        return np.ones(values.size, dtype=bool)
    idx = np.searchsorted(reference, values)
    idx_hi = np.clip(idx, 0, len(reference) - 1)
    idx_lo = np.clip(idx - 1, 0, len(reference) - 1)
    nearest = np.minimum(np.abs(reference[idx_hi] - values),
                         np.abs(reference[idx_lo] - values))
    return nearest > tol
