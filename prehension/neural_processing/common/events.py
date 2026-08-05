#!python3
# -*- coding: utf-8 -*-
"""
Event and TTL-sync helpers (reusable across neural work).

Read Open Ephys events, pick the TTL/sync channel and segment that actually carry
events, and extract per-pulse rising/falling edge times on the (possibly merged)
recording timebase.  All operate on the duck-typed config object; the full
recording-loading TTL extraction that maps edges to frames lives in ttl_sync.

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

from ...tools.misc import get_field, offset_array, concat_arrays
from . import openephys
from . import streams


# ---------------------------------------------------------------------------
# Event-channel / segment resolution
# ---------------------------------------------------------------------------
def load_events(cfg):
    import spikeinterface.full as si

    return si.read_openephys_event(str(cfg.oe_folder), block_index=cfg.block_index)


def _ttl_name_score(channel_id):
    s = str(channel_id).lower()
    if 'message' in s:
        return -1
    score = 0
    if 'ttl' in s:
        score += 3
    if 'sync' in s:
        score += 2
    if 'ap' in s:
        score += 1
    return score


def event_segment_counts(events, channel_id):
    """Return {segment_index: n_events} for a channel across all event segments."""
    try:
        n_seg = events.get_num_segments()
    except Exception:
        n_seg = 1
    counts = {}
    for s in range(n_seg):
        try:
            ev = events.get_events(channel_id=channel_id, segment_index=s)
            counts[s] = int(len(ev)) if ev is not None else 0
        except Exception:
            counts[s] = None
    return counts


def resolve_ttl_channel(cfg, events):
    """Pick the TTL/sync channel that actually has events (avoids 'Messages')."""
    channel_ids = list(events.channel_ids)
    if cfg.ttl_event_channel is not None:
        if cfg.ttl_event_channel not in channel_ids:
            raise ValueError('ttl_event_channel={!r} not in {}'.format(
                cfg.ttl_event_channel, channel_ids))
        return cfg.ttl_event_channel

    seen, unique_ids = set(), []
    for cid in channel_ids:
        if str(cid) not in seen:
            seen.add(str(cid))
            unique_ids.append(cid)

    def n_events(cid):
        counts = [c for c in event_segment_counts(events, cid).values() if c]
        return max(counts) if counts else 0

    with_events = [cid for cid in unique_ids if n_events(cid) > 0]
    if with_events:
        with_events.sort(key=lambda cid: (_ttl_name_score(cid), n_events(cid)),
                         reverse=True)
        return with_events[0]
    hinted = [cid for cid in unique_ids if _ttl_name_score(cid) > 0]
    return hinted[0] if hinted else channel_ids[0]


def find_event_segment(cfg, events, channel_id, preferred=None):
    """Return (segment_index, ev) for the first segment with non-empty events."""
    if preferred is None:
        preferred = cfg.recording_index
    try:
        n_seg = events.get_num_segments()
    except Exception:
        n_seg = preferred + 1
    order = [preferred] + [s for s in range(n_seg) if s != preferred]

    fallback = None
    for s in order:
        try:
            ev = events.get_events(channel_id=channel_id, segment_index=s)
        except Exception:
            continue
        if fallback is None:
            fallback = (s, ev)
        if ev is not None and len(ev) > 0:
            return s, ev
    return fallback if fallback is not None else (preferred, None)


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------
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
    counts = event_segment_counts(events, ttl_ch)
    if verbose:
        print('Events per segment:', counts)

    if cfg.ttl_event_segment is not None:
        return int(cfg.ttl_event_segment)

    found_segment, _ = find_event_segment(cfg, events, ttl_ch)
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
    rec = openephys.oe_recording(session, cfg.block_index, cfg.recording_index)
    return int(rec.continuous[0].sample_numbers[0])


def _extract_edge_times_source(scfg, verbose=False):
    """Read one source's TTL rising/falling edges (raw acquisition-clock times).

    Returns the same dict shape as extract_ttl_edge_times for a single recording.
    """
    events = load_events(scfg)
    ttl_ch = resolve_ttl_channel(scfg, events)
    ttl_segment = _resolve_ttl_segment(scfg, events, ttl_ch, verbose=verbose)
    if verbose:
        print('TTL channel={}; segment_index={}'.format(ttl_ch, ttl_segment))

    ev = events.get_events(channel_id=ttl_ch, segment_index=ttl_segment)
    rising_s, falling_s, rising_si, falling_si = _extract_edges(ev)
    return dict(channel=str(ttl_ch), segment=int(ttl_segment),
                rising_times_s=rising_s, falling_times_s=falling_s,
                rising_event_sample_indices=rising_si,
                falling_event_sample_indices=falling_si)


def extract_ttl_edge_times(cfg, verbose=False):
    """Read TTL rising/falling edge times (s) without loading the recording.

    Arguments:
        cfg {NeuralConfig} --- Resolved session configuration.
        verbose {bool} --- Print the channel / per-segment counts.

    Returns a dict with rising/falling times (s) and event sample indices, plus
    the channel and segment used.  When cfg spans several merge sources, each
    source's edges are zeroed to its own start and shifted by its cumulative sample
    offset (streams.merge_timeline) onto the concatenated timebase, then appended in
    source order; ``channel``/``segment`` become per-source lists.
    """
    if not getattr(cfg, 'is_merged', False):
        return _extract_edge_times_source(cfg, verbose=verbose)

    sources = streams.resolve_recording_sources(cfg)
    rt, ft, rsi, fsi, channels, segments = [], [], [], [], [], []
    for scfg, entry in zip(sources, streams.merge_timeline(cfg)):
        e = _extract_edge_times_source(scfg, verbose=verbose)
        fs = entry['fs']
        first = entry['first_sample'] or 0
        offset = entry['sample_offset']
        # zero each source to its own start, then place on the concatenated timebase
        t_shift = offset / fs - first / fs
        s_shift = offset - first
        rt.append(offset_array(e['rising_times_s'], t_shift))
        ft.append(offset_array(e['falling_times_s'], t_shift))
        rsi.append(offset_array(e['rising_event_sample_indices'], s_shift))
        fsi.append(offset_array(e['falling_event_sample_indices'], s_shift))
        channels.append(e['channel'])
        segments.append(e['segment'])
    return dict(channel=channels, segment=segments,
                rising_times_s=concat_arrays(rt), falling_times_s=concat_arrays(ft),
                rising_event_sample_indices=concat_arrays(rsi),
                falling_event_sample_indices=concat_arrays(fsi))
