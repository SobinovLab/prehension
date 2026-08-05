#!python3
# -*- coding: utf-8 -*-
"""
SpikeInterface stream / segment resolution, recording loaders, and the merge
timeline (reusable across neural work).

Low-level primitives (get_streams, resolve_stream, select_one_segment,
load_si_folder, recording_channel_names) plus the probe-specific recording loaders
and the multi-recording concatenation layout.  Everything operates on a duck-typed
config object (``cfg`` / per-source ``scfg`` carrying oe_folder / block_index /
recording_index / probe_type / expected_n_channels / stream overrides / the
merge_sources helpers), so this module does not depend on the NeuralConfig class.

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

from ...tools.logs import rs, ws
from . import openephys


# ---------------------------------------------------------------------------
# Stream resolution
# ---------------------------------------------------------------------------
def get_streams(cfg):
    """Return (stream_names, stream_ids) for the Open Ephys binary folder."""
    import spikeinterface.full as si

    stream_names, stream_ids = si.get_neo_streams('openephysbinary', str(cfg.oe_folder))
    return list(stream_names), list(stream_ids)


def _is_ap_stream(name):
    n = name.lower()
    return n.endswith('ap') or '-ap' in n or '.ap' in n


def resolve_stream(cfg, stream_names, stream_ids):
    """Pick the continuous neural stream.

    Neuropixels -> first AP-named stream; V-probe -> the stream whose channel
    count equals expected_n_channels.  Priority: stream_id -> stream_name ->
    probe rule -> first stream.  Returns (stream_id, stream_name).
    """
    import spikeinterface.full as si

    pairs = list(zip(stream_names, stream_ids))

    if cfg.stream_id is not None:
        for name, sid in pairs:
            if str(sid) == str(cfg.stream_id):
                return sid, name
        raise ValueError('stream_id={!r} not found in {}'.format(cfg.stream_id, stream_ids))
    if cfg.stream_name is not None:
        for name, sid in pairs:
            if name == cfg.stream_name:
                return sid, name
        raise ValueError('stream_name={!r} not found in {}'.format(cfg.stream_name, stream_names))

    if cfg.probe_type == 'neuropixels':
        ap = [(name, sid) for name, sid in pairs if _is_ap_stream(name)]
        if ap:
            name, sid = ap[0]
            return sid, name
    else:  # vprobe: match channel count
        for name, sid in pairs:
            try:
                r = si.read_openephys(str(cfg.oe_folder), stream_id=str(sid),
                                      block_index=cfg.block_index)
                if r.get_num_channels() == cfg.expected_n_channels:
                    return sid, name
            except Exception:
                continue

    name, sid = pairs[0]
    return sid, name


def select_one_segment(rec, segment_index):
    """Reduce a multi-segment recording to a single mono-segment recording."""
    import spikeinterface.full as si

    if hasattr(rec, 'select_segments'):
        return rec.select_segments([segment_index])
    return si.select_segment_recording(rec, segment_indices=segment_index)


def load_si_folder(folder):
    import spikeinterface.full as si

    if hasattr(si, 'load'):
        try:
            return si.load(str(folder))
        except Exception:
            pass
    return si.load_extractor(str(folder))


def recording_channel_names(rec):
    """Channel names in the recording's stored channel order."""
    names = rec.get_property('channel_name')
    if names is None:
        names = np.array([str(c) for c in rec.get_channel_ids()])
    return [str(n) for n in names]


# ---------------------------------------------------------------------------
# Merge of several recordings into one dataset (concatenation)
# ---------------------------------------------------------------------------
def resolve_recording_sources(cfg):
    """Ordered list of per-source configs to concatenate.

    Returns ``[cfg]`` unchanged for the single-recording path (so existing
    behaviour is byte-identical); otherwise one shallow-copied config per entry of
    cfg.merge_sources, each pinned to its own oe_folder/block/segment via
    cfg.for_source.  The order is the concatenation (and trial) order.
    """
    if not getattr(cfg, 'is_merged', False):
        return [cfg]
    return [cfg.for_source(s) for s in cfg.merge_sources]


def _segment_num_samples(scfg):
    """(n_samples, fs) of the SpikeInterface segment a source contributes.

    This is exactly the sample count concatenate_recordings uses, so the merge
    timeline matches the sorted spike frames.  Reads metadata only (no traces).
    """
    import spikeinterface.full as si

    stream_names, stream_ids = get_streams(scfg)
    stream_id, _ = resolve_stream(scfg, stream_names, stream_ids)
    rec = si.read_openephys(str(scfg.oe_folder), stream_id=str(stream_id),
                            block_index=scfg.block_index)
    if rec.get_num_segments() > 1:
        rec = select_one_segment(rec, scfg.recording_index)
    return rec.get_num_samples(), rec.get_sampling_frequency()


def _source_first_sample(scfg):
    """First continuous sample_number of a source, or None if it cannot be read."""
    try:
        from open_ephys.analysis import Session
        session = Session(str(scfg.oe_folder))
        rec = openephys.oe_recording(session, scfg.block_index, scfg.recording_index)
        return int(rec.continuous[0].sample_numbers[0])
    except Exception:
        return None


def merge_timeline(cfg):
    """Per-source concatenation layout on the joint spike timebase.

    For each source (in concatenation order) returns a dict with oe_folder,
    block_index, recording_index, n_samples (the SpikeInterface segment length),
    first_sample (Open Ephys continuous[0].sample_numbers[0], used to zero that
    source), fs, and sample_offset (cumulative n_samples of all preceding sources).

    Spikes and TTLs share this timeline: for a raw sample within source k,
    ``joint_time = (raw_sample - first_sample_k) / fs + sample_offset_k / fs``.
    """
    layout, offset = [], 0
    for scfg in resolve_recording_sources(cfg):
        n_samples, fs = _segment_num_samples(scfg)
        first_sample = _source_first_sample(scfg)
        layout.append(dict(
            oe_folder=str(scfg.oe_folder),
            block_index=int(scfg.block_index),
            recording_index=int(scfg.recording_index),
            n_samples=int(n_samples),
            first_sample=(None if first_sample is None else int(first_sample)),
            fs=float(fs),
            sample_offset=int(offset)))
        offset += int(n_samples)
    return layout


# ---------------------------------------------------------------------------
# Recording loaders (stream inspection lives in io_streams.inspect_streams)
# ---------------------------------------------------------------------------
def _load_one_source(scfg, with_sync, check_channels):
    """Read one source's stream and reduce it to its chosen segment.

    scfg is a per-source config (see resolve_recording_sources): its
    oe_folder/block_index/recording_index pin a single recording.  Returns
    (mono_segment_recording, stream_id, stream_name).  No probe is attached here;
    for the V-probe that happens once, after any concatenation.
    """
    import spikeinterface.full as si

    stream_names, stream_ids = get_streams(scfg)
    stream_id, stream_name = resolve_stream(scfg, stream_names, stream_ids)

    rec = si.read_openephys(str(scfg.oe_folder), stream_id=str(stream_id),
                            block_index=scfg.block_index, load_sync_timestamps=with_sync)
    if rec.get_num_segments() > 1:
        if scfg.recording_index >= rec.get_num_segments():
            raise ValueError('recording_index={} but recording has {} segment(s).'.format(
                scfg.recording_index, rec.get_num_segments()))
        rec = select_one_segment(rec, scfg.recording_index)

    if check_channels and rec.get_num_channels() != scfg.expected_n_channels:
        raise AssertionError('Stream {!r} has {} channels, expected {}. '
                             'Set stream_id/stream_name.'.format(
                                 stream_name, rec.get_num_channels(),
                                 scfg.expected_n_channels))
    return rec, stream_id, stream_name


def _validate_concatenable(recs):
    """Ensure recordings can be time-concatenated (same fs, channels, order)."""
    fs0 = recs[0].get_sampling_frequency()
    n0 = recs[0].get_num_channels()
    names0 = recording_channel_names(recs[0])
    for i, rec in enumerate(recs[1:], start=1):
        if rec.get_sampling_frequency() != fs0:
            raise ValueError('Cannot merge recordings: source {} sampling rate {} != '
                             'source 0 {}.'.format(i, rec.get_sampling_frequency(), fs0))
        if rec.get_num_channels() != n0:
            raise ValueError('Cannot merge recordings: source {} has {} channels != '
                             'source 0 {}.'.format(i, rec.get_num_channels(), n0))
        if recording_channel_names(rec) != names0:
            raise ValueError('Cannot merge recordings: source {} channel names/order '
                             'differ from source 0.'.format(i))


def _load_and_concatenate(cfg, with_sync, check_channels):
    """Load every merge source and time-concatenate into one mono-segment recording.

    Falls back to a single recording (no concatenation) when there is one source,
    so the common single-recording path is unchanged.  Returns
    (recording, stream_id, stream_name) with stream ids from the last source.
    """
    import spikeinterface.full as si

    recs = []
    stream_id = stream_name = None
    for scfg in resolve_recording_sources(cfg):
        rec, stream_id, stream_name = _load_one_source(
            scfg, with_sync=with_sync, check_channels=check_channels)
        recs.append(rec)

    if len(recs) == 1:
        return recs[0], stream_id, stream_name

    _validate_concatenable(recs)
    rs('Merging {} recordings into one dataset (concatenation).'.format(len(recs)))
    return si.concatenate_recordings(recs), stream_id, stream_name


def load_recording_neuropixels(cfg, with_sync=True):
    """Neuropixels: read the AP stream(s) and reduce to the chosen segment(s).

    When cfg has multiple merge sources the per-source recordings are
    time-concatenated into one continuous recording for a joint sort.
    """
    return _load_and_concatenate(cfg, with_sync=with_sync, check_channels=False)


def load_recording_vprobe(cfg, with_sync=True, with_probe=True):
    """V-probe: read the 32-ch stream(s), check count, attach manual probe.

    When cfg has multiple merge sources the per-source recordings are
    time-concatenated first; the manual probe geometry is attached once to the
    concatenated recording (channels are identical across sources).
    """
    from . import probe  # local import breaks the streams<->probe import cycle

    rec, stream_id, stream_name = _load_and_concatenate(
        cfg, with_sync=with_sync, check_channels=True)
    if with_probe:
        rec = probe.attach_probe(cfg, rec)
    return rec, stream_id, stream_name


def load_recording(cfg, with_sync=True):
    """Dispatch to the probe-specific loader."""
    if cfg.probe_type == 'neuropixels':
        return load_recording_neuropixels(cfg, with_sync=with_sync)
    return load_recording_vprobe(cfg, with_sync=with_sync)
