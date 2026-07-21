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


def _resolve_ttl_segment(cfg, events, ttl_ch, verbose=False):
    """Pick the event segment, enforcing agreement with the sorted segment.

    Prefers an explicit cfg.ttl_event_segment.  Otherwise the TTL events must
    live in segment == cfg.recording_index (the segment that was sorted); if the
    events are only found in a different segment this raises, because reading
    TTLs from another segment would put spikes and pulses on different clocks.
    """
    counts = config.event_segment_counts(events, ttl_ch)
    if verbose:
        print('Events per segment:', counts)

    if cfg.ttl_event_segment is not None:
        return int(cfg.ttl_event_segment)

    found_segment, _ = config.find_event_segment(cfg, events, ttl_ch)
    if found_segment != cfg.recording_index:
        raise ValueError(
            'TTL events for channel {} were found in segment {}, but the sorted '
            'segment is recording_index={} (no events there). Reading TTLs from a '
            'different segment would align spikes and pulses on different clocks. '
            'If this is intended, construct NeuralConfig(..., ttl_event_segment={}).'
            .format(ttl_ch, found_segment, cfg.recording_index, found_segment))
    return int(cfg.recording_index)


def sorted_segment_first_sample(cfg):
    """First continuous sample_number of the sorted segment (recording_index).

    The SpikeInterface event 'time' field is on the acquisition-board clock, but
    sorted spike frames are zeroed to the first sample of the sorted segment.
    Subtracting this offset -- ``(sample_index - first_sample) / fs`` -- puts TTL
    edges on the same origin as the spikes, matching the correction the v_probe
    figure applies (``timestamp - sample_numbers[0] / fs``).  Returns None if it
    cannot be read (the caller then keeps the raw event time field).
    """
    from open_ephys.analysis import Session
    session = Session(str(cfg.oe_folder))
    rec = session.recordnodes[0].recordings[cfg.recording_index]
    return int(rec.continuous[0].sample_numbers[0])


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
    ttl_segment = _resolve_ttl_segment(cfg, events, ttl_ch, verbose=verbose)
    if verbose:
        print('TTL channel={}; segment_index={}'.format(ttl_ch, ttl_segment))

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
    fs = recording.get_sampling_frequency()
    rising_frames = event_times_to_recording_frames(recording, rising_s, 0)
    falling_frames = event_times_to_recording_frames(recording, falling_s, 0)

    # Origin-corrected candidate times: zero the edges to the sorted segment's
    # first sample so they share an origin with the spike frames (spike_frame/fs).
    # export_nwb picks whichever origin (this or the raw event time) places more
    # spikes inside the pulse windows, which self-corrects for whether the SI
    # event 'time' field is already relative to the segment start.
    first_sample = sorted_segment_first_sample(cfg)
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
