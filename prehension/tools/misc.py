#!python3
# -*- coding: utf-8 -*-
"""
Miscellaneous functions.

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
