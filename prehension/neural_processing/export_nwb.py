#!python3
# -*- coding: utf-8 -*-
"""
export_nwb --- the product.

Writes a minimal NWB (work/neural.nwb) holding only the neural structure plus the
TTL sync needed to align spikes to each trial: a Units table (spike_times in
seconds, + original unit_id), electrodes / probe geometry (best-effort), and a
TimeIntervals 'ttl_pulses' (start=rising, stop=falling edge per pulse).

No behavioural/trial metadata (trial numbers, forces, timepoints, success) and no
diagnostics (quality metrics, waveforms) are embedded.  Downstream code pairs
ttl_pulses[i] to trial i and aligns spikes within each pulse window.

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
from uuid import uuid4

import numpy as np

from . import config
from . import ttl_sync
from ..tools.logs import rs, ws


def _select_sorting(cfg):
    """Choose the sorting per cfg.nwb_units ('all' or 'curated')."""
    import spikeinterface.full as si

    if cfg.nwb_units == 'curated':
        phy = cfg.subfolders()['phy']
        if os.path.exists(os.path.join(phy, 'params.py')):
            rs('Units: curated (Phy, noise excluded).')
            return si.read_phy(str(phy), exclude_cluster_groups=['noise'])
        curated = cfg.subfolders()['analyzer_curated']
        if os.path.exists(curated):
            rs('Units: curated (analyzer_curated).')
            return si.load_sorting_analyzer(str(curated)).sorting
        ws("nwb_units='curated' but no phy/analyzer_curated found; "
           'using all sorted units.')
    else:
        rs('Units: all sorted units.')
    return config.load_sorting(cfg)


def _add_electrodes(nwbfile, cfg, rec):
    """Best-effort electrodes table from the recording probe geometry."""
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


def export_nwb(cfg):
    """Write the minimal NWB neural product for one session."""
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.epoch import TimeIntervals

    cfg.ensure_work_folder()
    rs('Exporting neural.nwb.')

    rec_pre = config.load_preprocessed(cfg)
    sorting = _select_sorting(cfg)
    fs = sorting.get_sampling_frequency()

    # TTL sync (times in seconds); the preprocessed rec is enough for the calc.
    ttl = ttl_sync.extract_ttl_events(cfg, recording=rec_pre)

    nwbfile = NWBFile(
        session_description='{} spike sorting, {}'.format(cfg.probe_type, cfg.session),
        identifier=str(uuid4()),
        session_start_time=cfg.session_start_time(),
        session_id=cfg.session)

    _add_electrodes(nwbfile, cfg, rec_pre)

    # Units.
    nwbfile.add_unit_column(name='unit_id', description='original sorter unit id')
    for uid in sorting.get_unit_ids():
        st = np.asarray(sorting.get_unit_spike_train(unit_id=uid), dtype=float) / fs
        nwbfile.add_unit(spike_times=st, unit_id=str(uid))
    print('Added {} units.'.format(len(sorting.get_unit_ids())))

    # TTL sync as TimeIntervals (start=rising, stop=falling).
    rising = ttl['rising_times_s']
    falling = ttl['falling_times_s']
    n = 0 if rising is None else len(rising)
    ttl_ti = TimeIntervals(
        name='ttl_pulses',
        description='TTL sync pulses; start_time=rising edge, stop_time=falling edge')
    for i in range(n):
        start = float(rising[i])
        stop = float(falling[i]) if falling is not None and i < len(falling) else start
        ttl_ti.add_row(start_time=start, stop_time=max(stop, start))
    nwbfile.add_time_intervals(ttl_ti)
    print('Added ttl_pulses with {} pulses (channel {}, segment {}).'.format(
        n, ttl['channel'], ttl['segment']))

    path = cfg.nwb_path
    if os.path.exists(path):
        os.remove(path)
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwbfile)
    rs('Wrote NWB -> {}'.format(path))
    return path
