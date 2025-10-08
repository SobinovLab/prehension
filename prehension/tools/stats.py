#!python3
# -*- coding: utf-8 -*-
"""
Stats-related functions.

Copyright (C) 2025 Anton Sobinov
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


def format_p(p):
    '''Formats p-value according to recommendations with additional zero.

    Relevant papers:
    https://journals.physiology.org/doi/full/10.1152/ajpendo.00213.2004
    https://journals.physiology.org/doi/full/10.1152/advan.00022.2007
    https://journals.physiology.org/doi/full/10.1152/advan.00231.2023
    '''
    if p < 0.0001:
        ps = 'p < 0.0001'
    elif p < 0.001:
        ps = f'p = {round(p, 4):.4f}'
    elif p < 0.01:
        ps = f'p = {round(p, 3):.3f}'
    else:
        ps = f'p = {round(p, 2):.2f}'
    return ps
