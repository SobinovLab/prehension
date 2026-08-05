#!python3
# -*- coding: utf-8 -*-
"""
Probe helpers (reusable across neural work): resolve a session's probe type from
its meta_structure, build/attach the manual V-probe geometry, and build a
best-effort NWB electrodes table from a recording's channel locations.

Geometry is read from the duck-typed config object (``cfg.geometry`` /
``cfg.contact_pitch_um`` / ``cfg.contact_radius_um`` / ``cfg.contact_channel_names``
/ ``cfg.expected_n_channels``), so this module does not depend on the NeuralConfig
class.

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

import numpy as np

from ...tools import io
from ...tools.logs import rs, ws
from .streams import recording_channel_names


# Mapping from the meta_structure 'neural' field (written by fill_meta_structure) to a
# probe_type key understood by PROBE_DEFAULTS / NeuralConfig.
NEURAL_TO_PROBE_TYPE = {
    'vprobe': 'vprobe',
    'neuropixel': 'neuropixels',
    'neuropixels': 'neuropixels',
}


def probe_type_from_meta(server, processed_server, session):
    """Resolve a session's probe_type from its meta_structure.json 'neural' field.

    The field is written by meta_session.fill_meta_structure during create_meta and is one of
    '' (no neural data), 'vprobe', or 'neuropixel'. Returns a probe_type key of PROBE_DEFAULTS
    ('vprobe' or 'neuropixels'). Raises ValueError if meta is missing or has no neural type.
    """
    meta_path = os.path.join(processed_server, session, 'meta_structure.json')
    if not os.path.exists(meta_path):
        raise ValueError(
            'meta_structure.json not found for session {} at {}. Run create_meta first.'.format(
                session, meta_path))
    mstruct = io.load_json(meta_path)

    neural = (mstruct.get('neural') or '').strip()
    if neural == '':
        raise ValueError(
            'No neural recording type recorded in meta_structure for session {} (the "neural" '
            'field is empty). Re-run create_meta on a session that has a neural/ folder.'.format(
                session))
    if neural not in NEURAL_TO_PROBE_TYPE:
        raise ValueError(
            'Unknown neural recording type {!r} in meta_structure for session {}. Expected one '
            'of {}.'.format(neural, session, list(NEURAL_TO_PROBE_TYPE)))
    return NEURAL_TO_PROBE_TYPE[neural]


def _contact_positions(cfg):
    n = cfg.expected_n_channels
    pos = np.zeros((n, 2), dtype=float)
    if cfg.geometry == 'linear':
        for i in range(n):
            pos[i] = [0.0, i * cfg.contact_pitch_um]
    elif cfg.geometry == 'staggered':
        for i in range(n):
            row, col = divmod(i, 2)
            pos[i] = [col * cfg.horizontal_pitch_um, row * cfg.contact_pitch_um]
    else:
        raise ValueError('Unknown geometry {!r}'.format(cfg.geometry))
    return pos


def build_vprobe(cfg, rec):
    """Build a 32-channel probe and wire contacts to device channels by name."""
    from probeinterface import Probe

    n = cfg.expected_n_channels
    if len(cfg.contact_channel_names) != n:
        raise ValueError('contact_channel_names has {} entries, expected {}.'.format(
            len(cfg.contact_channel_names), n))

    names = recording_channel_names(rec)
    name_to_index = {nm: i for i, nm in enumerate(names)}
    missing = [c for c in cfg.contact_channel_names if c not in name_to_index]
    if missing:
        raise ValueError('Contact channel names not in recording: {}. '
                         'Recording channel names (stored order): {}'.format(missing, names))

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=_contact_positions(cfg), shapes='circle',
                       shape_params={'radius': cfg.contact_radius_um})
    probe.create_auto_shape(probe_type='tip')
    device_indices = np.array([name_to_index[c] for c in cfg.contact_channel_names],
                              dtype='int64')
    probe.set_device_channel_indices(device_indices)
    return probe


def attach_probe(cfg, rec):
    """Attach the manually-built V-probe geometry to a recording."""
    return rec.set_probe(build_vprobe(cfg, rec))


def add_electrodes(nwbfile, cfg, rec):
    """Best-effort NWB electrodes table from the recording probe geometry."""
    try:
        device = nwbfile.create_device(name='probe', description=cfg.probe_type)
        eg = nwbfile.create_electrode_group(
            name='probe0', description='{} probe'.format(cfg.probe_type),
            location='unknown', device=device)
        locs = rec.get_channel_locations()
        for i in range(rec.get_num_channels()):
            x, y = float(locs[i][0]), float(locs[i][1])
            nwbfile.add_electrode(x=x, y=y, z=0.0, location='unknown',
                                  group=eg, group_name='probe0')
        print('Added {} electrodes.'.format(rec.get_num_channels()))
    except Exception as e:
        ws('Skipping electrodes table (schema mismatch): {}'.format(e))
