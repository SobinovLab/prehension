#!python3
# -*- coding: utf-8 -*-
"""
Preprocessing: the two probe-specific versions and a dispatcher.

Neuropixels: highpass(400) -> detect/remove bad channels -> phase_shift -> CMR.
V-probe:     highpass(300) -> detect/remove bad channels -> CMR (no phase_shift).

Both save a binary_folder recording to <work>/preprocessed and write a small
bad-channel JSON (diagnostic, not part of the NWB).

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
import shutil

import numpy as np

from ..tools import io
from ..tools.logs import rs, ws
from .common import streams


def _finish_and_save(cfg, rec_pre, bad_channel_ids, channel_labels, n_before):
    cfg.ensure_work_folder()
    pre_folder = cfg.subfolders()['preprocessed']
    if os.path.exists(pre_folder):
        ws('Removing existing preprocessed folder: {}'.format(pre_folder))
        shutil.rmtree(pre_folder)
    rs('Saving preprocessed recording -> {}'.format(pre_folder))
    rec_saved = rec_pre.save(folder=pre_folder, format='binary', **cfg.job_kwargs)
    print(rec_saved)

    io.save_json(dict(
        probe_type=cfg.probe_type,
        bad_channel_ids=[str(c) for c in bad_channel_ids],
        channel_labels=([str(x) for x in np.asarray(channel_labels).tolist()]
                        if channel_labels is not None else None),
        n_channels_before=int(n_before),
        n_channels_after=int(rec_saved.get_num_channels()),
        highpass_freq_min=cfg.highpass_freq_min,
        phase_shift_applied=bool(cfg.use_phase_shift),
        common_reference=dict(operator=cfg.common_reference_operator,
                              reference=cfg.common_reference_type),
    ), os.path.join(cfg.work_folder, 'preprocess_bad_channels.json'))

    # When several recordings were concatenated, persist the per-source layout so
    # the timing steps (export_nwb, ttl_sync) can reconstruct the joint timebase.
    if getattr(cfg, 'is_merged', False):
        io.save_json(streams.merge_timeline(cfg),
                     os.path.join(cfg.work_folder, 'merge_layout.json'))
    return rec_saved


def preprocess_recording_neuropixels(cfg):
    """Neuropixels preprocessing chain (with phase_shift)."""
    import spikeinterface.full as si

    rs('Preprocessing Neuropixels recording.')
    # Sorting operates on frames; the synced time vector is not needed here.
    rec, _, _ = streams.load_recording_neuropixels(cfg, with_sync=False)

    rec_hp = si.highpass_filter(rec, freq_min=cfg.highpass_freq_min)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec_hp)
    print('Bad channels:', list(bad_channel_ids))
    rec_rm = rec_hp.remove_channels(bad_channel_ids)

    try:
        rec_ps = si.phase_shift(rec_rm)
    except Exception as e:
        ws('phase_shift failed; continuing without it: {}'.format(e))
        rec_ps = rec_rm

    rec_pre = si.common_reference(rec_ps, operator=cfg.common_reference_operator,
                                  reference=cfg.common_reference_type)
    return _finish_and_save(cfg, rec_pre, bad_channel_ids, channel_labels,
                            rec_hp.get_num_channels())


def preprocess_recording_vprobe(cfg):
    """V-probe preprocessing chain (no phase_shift)."""
    import spikeinterface.full as si

    rs('Preprocessing V-probe recording.')
    rec, _, _ = streams.load_recording_vprobe(cfg, with_sync=False, with_probe=True)

    rec_hp = si.highpass_filter(rec, freq_min=cfg.highpass_freq_min)
    bad_channel_ids, channel_labels = si.detect_bad_channels(rec_hp)
    print('Bad channels:', list(bad_channel_ids))
    rec_rm = rec_hp.remove_channels(bad_channel_ids)

    # No phase_shift for a V-probe.
    rec_pre = si.common_reference(rec_rm, operator=cfg.common_reference_operator,
                                  reference=cfg.common_reference_type)
    return _finish_and_save(cfg, rec_pre, bad_channel_ids, channel_labels,
                            rec_hp.get_num_channels())


def preprocess_recording(cfg):
    """Dispatch to the probe-specific preprocessing."""
    if cfg.probe_type == 'neuropixels':
        return preprocess_recording_neuropixels(cfg)
    return preprocess_recording_vprobe(cfg)
