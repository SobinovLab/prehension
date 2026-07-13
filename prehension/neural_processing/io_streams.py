#!python3
# -*- coding: utf-8 -*-
"""
Recording input: stream inspection and the two probe-specific loaders.

Two versions differ between probes:
    load_recording_neuropixels --- AP stream, select recording segment.
    load_recording_vprobe --- 32-ch stream, attach manual probe geometry.
load_recording dispatches on cfg.probe_type.

inspect_streams is diagnostic: it prints streams, segments, channel names and
per-segment event counts, and writes a JSON summary.

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
import os

from . import config
from ..tools.logs import rs, ws


def load_recording_neuropixels(cfg, with_sync=True):
    """Neuropixels: read the AP stream and reduce to the chosen segment."""
    import spikeinterface.full as si

    stream_names, stream_ids = config.get_streams(cfg)
    stream_id, stream_name = config.resolve_stream(cfg, stream_names, stream_ids)

    rec = si.read_openephys(str(cfg.oe_folder), stream_id=str(stream_id),
                            block_index=cfg.block_index, load_sync_timestamps=with_sync)
    if rec.get_num_segments() > 1:
        if cfg.recording_index >= rec.get_num_segments():
            raise ValueError('recording_index={} but recording has {} segment(s).'.format(
                cfg.recording_index, rec.get_num_segments()))
        rec = config.select_one_segment(rec, cfg.recording_index)
    return rec, stream_id, stream_name


def load_recording_vprobe(cfg, with_sync=True, with_probe=True):
    """V-probe: read the 32-ch stream, check count, attach manual probe."""
    import spikeinterface.full as si

    stream_names, stream_ids = config.get_streams(cfg)
    stream_id, stream_name = config.resolve_stream(cfg, stream_names, stream_ids)

    rec = si.read_openephys(str(cfg.oe_folder), stream_id=str(stream_id),
                            block_index=cfg.block_index, load_sync_timestamps=with_sync)
    if rec.get_num_segments() > 1:
        if cfg.recording_index >= rec.get_num_segments():
            raise ValueError('recording_index={} but recording has {} segment(s).'.format(
                cfg.recording_index, rec.get_num_segments()))
        rec = config.select_one_segment(rec, cfg.recording_index)

    if rec.get_num_channels() != cfg.expected_n_channels:
        raise AssertionError('Stream {!r} has {} channels, expected {}. '
                             'Set stream_id/stream_name.'.format(
                                 stream_name, rec.get_num_channels(),
                                 cfg.expected_n_channels))

    if with_probe:
        rec = config.attach_probe(cfg, rec)
    return rec, stream_id, stream_name


def load_recording(cfg, with_sync=True):
    """Dispatch to the probe-specific loader."""
    if cfg.probe_type == 'neuropixels':
        return load_recording_neuropixels(cfg, with_sync=with_sync)
    return load_recording_vprobe(cfg, with_sync=with_sync)


def inspect_streams(cfg):
    """Diagnostic: print streams / segments / channels / events; save JSON."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Inspecting streams, segments and events.')

    stream_names, stream_ids = config.get_streams(cfg)
    print('Continuous streams:')
    for name, sid in zip(stream_names, stream_ids):
        print('  stream_id={} | stream_name={}'.format(sid, name))
    chosen_id, chosen_name = config.resolve_stream(cfg, stream_names, stream_ids)
    print('Selected stream: id={} name={}'.format(chosen_id, chosen_name))

    rec = si.read_openephys(str(cfg.oe_folder), stream_id=str(chosen_id),
                            block_index=cfg.block_index)
    fs = rec.get_sampling_frequency()
    n_seg = rec.get_num_segments()
    print('\nSegments: {}; fs={} Hz; channels={}'.format(n_seg, fs, rec.get_num_channels()))
    seg_info = []
    for seg in range(n_seg):
        n = rec.get_num_samples(segment_index=seg)
        tag = '  <-- default' if seg == cfg.recording_index else ''
        print('  segment {}: {} samples, {:.1f} s{}'.format(seg, n, n / fs, tag))
        seg_info.append(dict(segment_index=seg, n_samples=int(n),
                             duration_s=float(n / fs)))

    channel_names = config.recording_channel_names(rec)
    print('\nChannel names (stored order):')
    for i, nm in enumerate(channel_names):
        print('  {:2d}: {}'.format(i, nm))

    events = config.load_events(cfg)
    print('\nEvent channels:', list(events.channel_ids))
    event_info = []
    for ch in events.channel_ids:
        counts = config.event_segment_counts(events, ch)
        best_seg, ev = config.find_event_segment(cfg, events, ch)
        print('  {}: per-segment {}; non-empty segment {}'.format(ch, counts, best_seg))
        event_info.append(dict(channel_id=str(ch),
                               events_per_segment={str(k): v for k, v in counts.items()},
                               chosen_segment=int(best_seg),
                               n_events=int(len(ev)) if ev is not None else 0))
    ttl_ch = config.resolve_ttl_channel(cfg, events)
    ttl_seg, _ = config.find_event_segment(cfg, events, ttl_ch)
    print('\nTTL channel: {} (events in segment_index={})'.format(ttl_ch, ttl_seg))

    config.save_json(dict(
        probe_type=cfg.probe_type, oe_folder=str(cfg.oe_folder),
        chosen_stream=dict(stream_id=str(chosen_id), stream_name=chosen_name),
        sampling_frequency=float(fs), num_channels=int(rec.get_num_channels()),
        channel_names_stored_order=channel_names, segments=seg_info,
        event_channels=event_info, chosen_ttl_event_channel=str(ttl_ch),
        chosen_ttl_event_segment=int(ttl_seg),
    ), os.path.join(cfg.work_folder, 'inspect_streams.json'))
    rs('Saved inspect_streams.json in {}'.format(cfg.work_folder))
