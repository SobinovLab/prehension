#!python3
# -*- coding: utf-8 -*-
"""
Reusable helpers shared across the neural figure modules.

    behaviour --- prehension behavioural-meta helpers (timepoints, target force).
    pooling   --- cross-session pooling of per-neuron / per-trial neural activity.
    traces    --- pure figure drawers (PC/dPC traces, trajectories, classification).

Copyright (C) 2026 Anton Sobinov
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
__all__ = ['behaviour', 'pooling', 'traces']

from . import behaviour
from . import pooling
from . import traces
