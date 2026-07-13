#!python3
# -*- coding: utf-8 -*-
"""
TTL sync extraction (shared between probes).

extract_ttl_events returns per-pulse rising/falling times (seconds, on the
recording timebase).  It handles event-aware channel selection (avoids
'Messages'; prefers ttl/sync/ap), non-empty segment search (TTLs may not be in
segment == recording_index), and both edge-based (state) and epoch-based (time,
duration, label) event formats.  It is consumed by export_nwb.

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

from . import config
from . import io_streams
from ..tools.logs import rs, ws


def get_field(ev, possible_names):
    names = ev.dtype.names or ()
    for name in possible_names:
        if name in names:
            return ev[name]
    return None


def _extract_edges(ev):
    """Return (rising_times, falling_times, rising_sample_idx, falling_sample_idx)."""
    times = get_field(ev, ['time', 'timestamp', 'times'])
    durations = get_field(ev, ['duration', 'durations'])
    sample_indices = get_field(ev, ['sample_index', 'sample_number', 'sample_numbers'])
    states = get_field(ev, ['state', 'event_state', 'ttl_state'])

    if times is None and sample_indices is None:
        raise ValueError('No time or sample-index field in events; inspect '
                         'ev.dtype.names and adapt.')

    if states is not None:
        rising, falling = states == 1, states == 0
        rt = times[rising] if times is not None else None
        ft = times[falling] if times is not None else None
        rsi = sample_indices[rising] if sample_indices is not None else None
        fsi = sample_indices[falling] if sample_indices is not None else None
        return rt, ft, rsi, fsi

    if durations is not None and times is not None:
        # Epoch/pulse events: rising at time, falling at time + duration.
        rsi = np.asarray(sample_indices) if sample_indices is not None else None
        return np.asarray(times), np.asarray(times) + np.asarray(durations), rsi, None

    # No state, no duration: treat every event as a rising edge.
    rsi = np.asarray(sample_indices) if sample_indices is not None else None
    return (np.asarray(times) if times is not None else None,
            np.array([], dtype=float) if times is not None else None, rsi, None)


def event_times_to_recording_frames(recording, event_times_s, segment_index=0):
    """Convert event times (s) to nearest recording sample frame."""
    if event_times_s is None:
        return None
    fs = recording.get_sampling_frequency()
    if not recording.has_time_vector(segment_index=segment_index):
        return np.round(np.asarray(event_times_s) * fs).astype(np.int64)

    rec_times = recording.get_times(segment_index=segment_index)
    frames = np.searchsorted(rec_times, event_times_s)
    frames = np.clip(frames, 0, len(rec_times) - 1)
    prev = np.maximum(frames - 1, 0)
    use_prev = (np.abs(rec_times[prev] - event_times_s)
                < np.abs(rec_times[frames] - event_times_s))
    frames[use_prev] = prev[use_prev]
    return frames.astype(np.int64)


def extract_ttl_edge_times(cfg, verbose=False):
    """Read TTL rising/falling edge times (s) without loading the recording.

    Arguments:
        cfg {NeuralConfig} --- Resolved session configuration.
        verbose {bool} --- Print the channel / per-segment counts.

    Returns a dict with rising/falling times (s) and event sample indices, plus
    the channel and segment used.
    """
    events = config.load_events(cfg)
    ttl_ch = config.resolve_ttl_channel(cfg, events)
    if verbose:
        print('Events per segment:', config.event_segment_counts(events, ttl_ch))
    ttl_segment, _ = config.find_event_segment(cfg, events, ttl_ch)
    if verbose:
        print('TTL channel={}; segment_index={}'.format(ttl_ch, ttl_segment))
        if ttl_segment != cfg.recording_index:
            ws('Event segment {} != recording_index {}; confirm the events belong '
               'to the sorted recording.'.format(ttl_segment, cfg.recording_index))

    ev = events.get_events(channel_id=ttl_ch, segment_index=ttl_segment)
    rising_s, falling_s, rising_si, falling_si = _extract_edges(ev)
    return dict(channel=str(ttl_ch), segment=int(ttl_segment),
                rising_times_s=rising_s, falling_times_s=falling_s,
                rising_event_sample_indices=rising_si,
                falling_event_sample_indices=falling_si)


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
    edges = extract_ttl_edge_times(cfg, verbose=True)
    rising_s, falling_s = edges['rising_times_s'], edges['falling_times_s']
    n_rise = 0 if rising_s is None else len(rising_s)
    n_fall = 0 if falling_s is None else len(falling_s)
    print('rising={}  falling={}'.format(n_rise, n_fall))

    if recording is None:
        recording, _, _ = io_streams.load_recording(cfg, with_sync=True)
    rising_frames = event_times_to_recording_frames(recording, rising_s, 0)
    falling_frames = event_times_to_recording_frames(recording, falling_s, 0)

    return dict(channel=edges['channel'], segment=edges['segment'],
                rising_times_s=rising_s, falling_times_s=falling_s,
                rising_frames=rising_frames, falling_frames=falling_frames,
                rising_event_sample_indices=edges['rising_event_sample_indices'],
                falling_event_sample_indices=edges['falling_event_sample_indices'])
