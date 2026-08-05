#!python3
# -*- coding: utf-8 -*-
"""
Reusable neural helpers shared across neural processing and plotting.

Themed modules of functions that are neural-specific but not tied to the current
pipeline's NeuralConfig / behavioural model:
    openephys  --- Open Ephys folder / recording-layout helpers.
    streams    --- SpikeInterface stream / segment resolution, loaders, merge timeline.
    probe      --- probe-type resolution, V-probe geometry, NWB electrodes.
    events     --- event-channel resolution and TTL edge extraction.
    spikes     --- NWB spike reading, per-trial slicing, neuron selection + constants.
    population --- pure aggregation of pooled activity (matrices, tensors, dPCA, pools).

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
__all__ = ['openephys', 'streams', 'probe', 'events', 'spikes', 'population']

from . import openephys
from . import streams
from . import probe
from . import events
from . import spikes
from . import population
