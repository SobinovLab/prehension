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
