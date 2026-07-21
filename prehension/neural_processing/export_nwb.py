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


def _sorting_from_phy(cfg, exclude_groups=('noise',)):
    """Load a Phy-curated sorting, dropping clusters whose group is in exclude_groups.

    Reimplements si.read_phy()'s noise-exclusion using the exported numpy arrays
    (spike_times.npy / spike_clusters.npy) plus cluster_group.tsv (falling back to
    the Kilosort auto labels in cluster_KSLabel.tsv).  si.read_phy() itself raises
    "assignment destination is read-only" in this SpikeInterface version: when the
    export contains an 'si_unit_id' column (which export_to_phy writes) it takes
    that column's read-only pandas '.values' and mutates it in place
    (phykilosortextractors.py: ``unit_ids[i] = new_si_id``).  Unit ids here are the
    Phy cluster ids, so they match what was curated in Phy.
    """
    import numpy as np
    import pandas as pd
    import spikeinterface.full as si

    phy = cfg.subfolders()['phy']
    spike_frames = np.load(os.path.join(phy, 'spike_times.npy')).squeeze().astype('int64')
    spike_clusters = np.load(os.path.join(phy, 'spike_clusters.npy')).squeeze().astype('int64')

    # Cluster group labels: manually-curated cluster_group.tsv wins over the
    # Kilosort auto labels in cluster_KSLabel.tsv.
    groups = {}
    for fname, col in (('cluster_group.tsv', 'group'),
                       ('cluster_KSLabel.tsv', 'KSLabel')):
        path = os.path.join(phy, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, sep='\t')
        if 'cluster_id' in df.columns and col in df.columns:
            for cid, grp in zip(df['cluster_id'].values, df[col].values):
                groups.setdefault(int(cid), str(grp))

    exclude = {str(g).lower() for g in exclude_groups}
    drop_ids = [cid for cid, grp in groups.items() if grp.lower() in exclude]

    keep = ~np.isin(spike_clusters, drop_ids)
    spike_frames = spike_frames[keep]
    spike_clusters = spike_clusters[keep]

    fs = config.load_sorting(cfg).get_sampling_frequency()
    sorting = si.NumpySorting.from_samples_and_labels(spike_frames, spike_clusters, float(fs))
    rs('Phy: dropped {} cluster(s) in {}; {} units kept.'.format(
        len(drop_ids), sorted(exclude), len(sorting.get_unit_ids())))
    return sorting


def _select_sorting(cfg):
    """Choose the sorting per cfg.nwb_units.

    'noise_excluded' (default): Phy export with 'noise' clusters dropped when a
        phy export exists; otherwise all sorted units (with a warning).  This
        matches the v_probes plotting flow (curate in Phy -> exclude noise).
    'curated': quality-triaged units (Phy noise-excluded, else analyzer_curated).
    'all': raw sorter output, no exclusion.
    """
    import spikeinterface.full as si

    phy = cfg.subfolders()['phy']
    has_phy = os.path.exists(os.path.join(phy, 'params.py'))

    if cfg.nwb_units == 'all':
        rs('Units: all sorted units (no exclusion).')
        return config.load_sorting(cfg)

    if cfg.nwb_units == 'curated':
        if has_phy:
            rs('Units: curated (Phy, noise excluded).')
            return _sorting_from_phy(cfg, exclude_groups=('noise',))
        curated = cfg.subfolders()['analyzer_curated']
        if os.path.exists(curated):
            rs('Units: curated (analyzer_curated).')
            return si.load_sorting_analyzer(str(curated)).sorting
        ws("nwb_units='curated' but no phy/analyzer_curated found; "
           'using all sorted units.')
        return config.load_sorting(cfg)

    # default: 'noise_excluded'
    if cfg.nwb_units != 'noise_excluded':
        ws("Unknown nwb_units={!r}; treating as 'noise_excluded'.".format(cfg.nwb_units))
    if has_phy:
        rs('Units: noise-excluded (Phy, noise clusters dropped).')
        return _sorting_from_phy(cfg, exclude_groups=('noise',))
    ws('Units: no Phy export found; using all sorted units. Run export_to_phy and '
       'label noise clusters to exclude them from the product.')
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


def _build_windows(rising, falling):
    """Build (starts, stops) arrays (s) from rising/falling edge time arrays.

    stop[i] is falling[i] when available (clamped to >= start), else start[i].
    """
    if rising is None or len(rising) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    starts = np.asarray(rising, dtype=float)
    if falling is None or len(falling) == 0:
        stops = starts.copy()
    else:
        falling = np.asarray(falling, dtype=float)
        stops = np.array([falling[i] if i < len(falling) else starts[i]
                          for i in range(len(starts))], dtype=float)
    return starts, np.maximum(stops, starts)


def _spikes_in_windows(all_spike_times, starts, stops):
    """Count spikes (across all units) falling inside any [start, stop] window."""
    if len(starts) == 0 or len(all_spike_times) == 0:
        return 0
    st = np.sort(np.asarray(all_spike_times, dtype=float))
    lo = np.searchsorted(st, starts, side='left')
    hi = np.searchsorted(st, stops, side='right')
    return int(np.sum(hi - lo))


def _choose_ttl_windows(ttl, all_spike_times):
    """Pick the TTL origin that places the most spikes inside the pulse windows.

    Candidates: 'synced' (edges zeroed to the sorted segment's first sample) and
    'raw' (the SpikeInterface event time field). Returns (starts, stops, origin,
    n_synced, n_raw). Prefers 'synced' on ties when it is available.
    """
    synced_start, synced_stop = _build_windows(
        ttl.get('rising_times_s_synced'), ttl.get('falling_times_s_synced'))
    raw_start, raw_stop = _build_windows(
        ttl.get('rising_times_s'), ttl.get('falling_times_s'))
    n_synced = _spikes_in_windows(all_spike_times, synced_start, synced_stop)
    n_raw = _spikes_in_windows(all_spike_times, raw_start, raw_stop)

    if len(synced_start) and n_synced >= n_raw:
        return synced_start, synced_stop, 'synced', n_synced, n_raw
    return raw_start, raw_stop, 'raw', n_synced, n_raw


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

    # Units (spike times in seconds, zeroed to the sorted segment start).
    unit_spike_times = {
        uid: np.asarray(sorting.get_unit_spike_train(unit_id=uid), dtype=float) / fs
        for uid in sorting.get_unit_ids()}
    nwbfile.add_unit_column(name='unit_id', description='original sorter unit id')
    for uid, st in unit_spike_times.items():
        nwbfile.add_unit(spike_times=st, unit_id=str(uid))
    print('Added {} units.'.format(len(unit_spike_times)))

    all_spikes = (np.concatenate(list(unit_spike_times.values()))
                  if unit_spike_times else np.array([], dtype=float))

    # TTL sync as TimeIntervals (start=rising, stop=falling). Choose the edge
    # origin that best overlaps the spikes so pulses share the spike timebase.
    starts, stops, origin, n_synced, n_raw = _choose_ttl_windows(ttl, all_spikes)
    rs('TTL origin: {} ({} spikes in windows; alternative {}).'.format(
        origin, n_synced if origin == 'synced' else n_raw,
        n_raw if origin == 'synced' else n_synced))
    if len(all_spikes) and max(n_synced, n_raw) == 0:
        ws('No spikes fall inside any TTL pulse window under either origin; check '
           'TTL extraction and the sorted segment before trusting the alignment.')

    ttl_ti = TimeIntervals(
        name='ttl_pulses',
        description='TTL sync pulses; start_time=rising edge, stop_time=falling edge')
    for i in range(len(starts)):
        ttl_ti.add_row(start_time=float(starts[i]), stop_time=float(stops[i]))
    nwbfile.add_time_intervals(ttl_ti)
    print('Added ttl_pulses with {} pulses (channel {}, segment {}, origin {}).'.format(
        len(starts), ttl['channel'], ttl['segment'], origin))

    path = cfg.nwb_path
    if os.path.exists(path):
        os.remove(path)
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwbfile)
    rs('Wrote NWB -> {}'.format(path))
    return path
