#!python3
# -*- coding: utf-8 -*-
"""
Stream / event inspection diagnostic.

inspect_streams prints streams, segments, channel names and per-segment event
counts, and writes a JSON summary.  The reusable stream primitives and the two
probe-specific loaders now live in neural_processing.common.streams; this module
keeps only the (pipeline-specific) diagnostic.

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

from ..tools import io
from ..tools.logs import rs, ws
from .common import streams
from .common import events


def inspect_streams(cfg):
    """Diagnostic: print streams / segments / channels / events; save JSON."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Inspecting streams, segments and events.')

    stream_names, stream_ids = streams.get_streams(cfg)
    print('Continuous streams:')
    for name, sid in zip(stream_names, stream_ids):
        print('  stream_id={} | stream_name={}'.format(sid, name))
    chosen_id, chosen_name = streams.resolve_stream(cfg, stream_names, stream_ids)
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

    channel_names = streams.recording_channel_names(rec)
    print('\nChannel names (stored order):')
    for i, nm in enumerate(channel_names):
        print('  {:2d}: {}'.format(i, nm))

    ev = events.load_events(cfg)
    print('\nEvent channels:', list(ev.channel_ids))
    event_info = []
    for ch in ev.channel_ids:
        counts = events.event_segment_counts(ev, ch)
        best_seg, ev_data = events.find_event_segment(cfg, ev, ch)
        print('  {}: per-segment {}; non-empty segment {}'.format(ch, counts, best_seg))
        event_info.append(dict(channel_id=str(ch),
                               events_per_segment={str(k): v for k, v in counts.items()},
                               chosen_segment=int(best_seg),
                               n_events=int(len(ev_data)) if ev_data is not None else 0))
    ttl_ch = events.resolve_ttl_channel(cfg, ev)
    ttl_seg, _ = events.find_event_segment(cfg, ev, ttl_ch)
    print('\nTTL channel: {} (events in segment_index={})'.format(ttl_ch, ttl_seg))

    io.save_json(dict(
        probe_type=cfg.probe_type, oe_folder=str(cfg.oe_folder),
        chosen_stream=dict(stream_id=str(chosen_id), stream_name=chosen_name),
        sampling_frequency=float(fs), num_channels=int(rec.get_num_channels()),
        channel_names_stored_order=channel_names, segments=seg_info,
        event_channels=event_info, chosen_ttl_event_channel=str(ttl_ch),
        chosen_ttl_event_segment=int(ttl_seg),
    ), os.path.join(cfg.work_folder, 'inspect_streams.json'))
    rs('Saved inspect_streams.json in {}'.format(cfg.work_folder))
