#!python3
# -*- coding: utf-8 -*-
"""
Full TTL sync extraction (loads the recording to map edges to frames).

extract_ttl_events returns per-pulse rising/falling times (seconds, on the
recording timebase) plus origin-corrected candidates and recording frames.  The
event-channel resolution and the edge-time extraction it builds on now live in
neural_processing.common.events; the recording loaders in
neural_processing.common.streams.

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
import numpy as np

from ..tools.logs import rs, ws
from .common import events
from .common import streams


def extract_ttl_events(cfg, recording=None):
    """Extract TTL sync pulses.

    Arguments:
        cfg {NeuralConfig} --- Resolved session configuration.
        recording --- Optional recording used to map times to frames; if None a
            synced raw recording is loaded.

    Returns a dict with rising/falling times (s) and frames, plus the channel and
    segment used.
    """
    rs('Extracting TTL sync events.')
    edges = events.extract_ttl_edge_times(cfg, verbose=True)
    rising_s, falling_s = edges['rising_times_s'], edges['falling_times_s']
    n_rise = 0 if rising_s is None else len(rising_s)
    n_fall = 0 if falling_s is None else len(falling_s)
    print('rising={}  falling={}'.format(n_rise, n_fall))

    if recording is None:
        recording, _, _ = streams.load_recording(cfg, with_sync=True)
    fs = recording.get_sampling_frequency()
    rising_frames = events.event_times_to_recording_frames(recording, rising_s, 0)
    falling_frames = events.event_times_to_recording_frames(recording, falling_s, 0)

    # Origin-corrected candidate times: zero the edges to the sorted segment's
    # first sample so they share an origin with the spike frames (spike_frame/fs).
    # export_nwb picks whichever origin (this or the raw event time) places more
    # spikes inside the pulse windows, which self-corrects for whether the SI
    # event 'time' field is already relative to the segment start.  When merging,
    # extract_ttl_edge_times has already placed edges (times and sample indices) on
    # the concatenated timebase, so no further origin subtraction is needed.
    first_sample = (0 if getattr(cfg, 'is_merged', False)
                    else events.sorted_segment_first_sample(cfg))
    rising_si = edges['rising_event_sample_indices']
    falling_si = edges['falling_event_sample_indices']

    def _from_samples(si_idx):
        if si_idx is None or first_sample is None:
            return None
        return (np.asarray(si_idx, dtype=float) - first_sample) / fs

    rising_times_s_synced = _from_samples(rising_si)
    falling_times_s_synced = _from_samples(falling_si)
    # Epoch/duration edges carry no falling sample index; apply the same constant
    # rising-edge offset to the raw falling times.
    if (falling_times_s_synced is None and rising_times_s_synced is not None
            and falling_s is not None and rising_s is not None
            and len(rising_s) == len(rising_times_s_synced) and len(rising_s) > 0):
        offset = float(np.median(np.asarray(rising_times_s_synced, dtype=float)
                                 - np.asarray(rising_s, dtype=float)))
        falling_times_s_synced = np.asarray(falling_s, dtype=float) + offset

    return dict(channel=edges['channel'], segment=edges['segment'],
                first_sample=first_sample, fs=float(fs),
                rising_times_s=rising_s, falling_times_s=falling_s,
                rising_times_s_synced=rising_times_s_synced,
                falling_times_s_synced=falling_times_s_synced,
                rising_frames=rising_frames, falling_frames=falling_frames,
                rising_event_sample_indices=rising_si,
                falling_event_sample_indices=falling_si)
