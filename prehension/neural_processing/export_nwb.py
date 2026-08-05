#!python3
# -*- coding: utf-8 -*-
"""
export_nwb --- the product.

Writes a minimal NWB (work/neural.nwb) holding only the neural structure plus the
TTL sync needed to align spikes to each trial: a Units table (spike_times in
seconds, + original unit_id), electrodes / probe geometry (best-effort), and a
TimeIntervals 'ttl_pulses' (per-trial [start, stop] windows).

Spikes and TTL windows are produced exactly as neural_plotting.figure_peth2 does:
units come from the sorter output (_select_sorting), and the TTL windows are read
directly from the Open Ephys events and zeroed to the sorted recording start
(_read_oe_ttl_windows == figure_peth2.read_oe_event).  This lets figure_peth3 --
which reads this NWB -- reproduce the figure_peth2 figure.

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
from ..tools.logs import rs, ws
from .common import openephys, streams, probe


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


def _read_oe_ttl_windows_source(scfg, fs, concat_offset_samples=0):
    """TTL [start, stop] windows for one source, on the (joint) spike timebase.

    Consecutive events are paired into [start, stop] windows, zeroed to that
    source's first sample, then shifted by concat_offset_samples/fs to place them
    on the concatenated timebase (0 for the single-recording path).  probe_type
    'neuropixels' reads the 'ProbeA-AP' event stream via sample_number; 'vprobe'
    reads all events via their timestamps.  Returns (starts, stops) lists (s).
    """
    from open_ephys.analysis import Session

    probe = 'np' if scfg.probe_type == 'neuropixels' else 'v'
    session = Session(str(scfg.oe_folder))
    recording = openephys.oe_recording(session, scfg.block_index, scfg.recording_index)
    events = recording.events
    first_sample = recording.continuous[0].sample_numbers[0]
    time_offset = first_sample / fs
    shift = concat_offset_samples / fs

    starts, stops = [], []
    if probe == 'np':
        events_ap = events[events.stream_name == 'ProbeA-AP']
        for i in range(0, len(events_ap) - 1, 2):
            starts.append((events_ap.iloc[i]['sample_number'] - first_sample) / fs + shift)
            stops.append((events_ap.iloc[i + 1]['sample_number'] - first_sample) / fs + shift)
    else:
        for i in range(0, len(events) - 1, 2):
            starts.append(events.iloc[i]['timestamp'] - time_offset + shift)
            stops.append(events.iloc[i + 1]['timestamp'] - time_offset + shift)
    return starts, stops


def _read_oe_ttl_windows(cfg, fs):
    """Per-trial TTL [start, stop] windows read directly from the Open Ephys events.

    Identical to neural_plotting.figure_peth2.read_oe_event for a single recording.
    When cfg spans several merge sources, each source's windows are zeroed to its
    own start and shifted by its cumulative sample offset (streams.merge_timeline),
    so pulses from later recordings continue past the earlier ones on the same
    timebase as the concatenated spikes.  Windows are appended in source order.
    Returns (starts, stops) arrays (s).
    """
    if not getattr(cfg, 'is_merged', False):
        starts, stops = _read_oe_ttl_windows_source(cfg, fs)
    else:
        starts, stops = [], []
        sources = streams.resolve_recording_sources(cfg)
        for scfg, entry in zip(sources, streams.merge_timeline(cfg)):
            s, e = _read_oe_ttl_windows_source(scfg, fs, entry['sample_offset'])
            starts.extend(s)
            stops.extend(e)

    starts = np.asarray(starts, dtype=float)
    stops = np.maximum(np.asarray(stops, dtype=float), starts)
    return starts, stops


def export_nwb(cfg):
    """Write the minimal NWB neural product for one session."""
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.epoch import TimeIntervals

    cfg.ensure_work_folder()
    rs('Exporting neural.nwb.')

    rec_pre = config.load_preprocessed(cfg)
    sorting = _select_sorting(cfg)
    fs = sorting.get_sampling_frequency()

    nwbfile = NWBFile(
        session_description='{} spike sorting, {}'.format(cfg.probe_type, cfg.session),
        identifier=str(uuid4()),
        session_start_time=cfg.session_start_time(),
        session_id=cfg.session)

    probe.add_electrodes(nwbfile, cfg, rec_pre)

    # Units (spike times in seconds, zeroed to the sorted segment start) -- from
    # _select_sorting, exactly as figure_peth2.get_neural_spikes.
    nwbfile.add_unit_column(name='unit_id', description='original sorter unit id')
    n_units = 0
    for uid in sorting.get_unit_ids():
        st = np.asarray(sorting.get_unit_spike_train(unit_id=uid), dtype=float) / fs
        nwbfile.add_unit(spike_times=st, unit_id=str(uid))
        n_units += 1
    print('Added {} units.'.format(n_units))

    # TTL windows read directly from the Open Ephys events, exactly as
    # figure_peth2.read_oe_event -> figure_peth3 reproduces that figure.
    starts, stops = _read_oe_ttl_windows(cfg, fs)
    ttl_ti = TimeIntervals(
        name='ttl_pulses',
        description='Per-trial TTL windows; start_time=window start, stop_time=window end, '
                    'zeroed to the sorted recording start (see figure_peth2.read_oe_event)')
    for i in range(len(starts)):
        ttl_ti.add_row(start_time=float(starts[i]), stop_time=float(stops[i]))
    nwbfile.add_time_intervals(ttl_ti)
    print('Added ttl_pulses with {} pulses.'.format(len(starts)))

    path = cfg.nwb_path
    if os.path.exists(path):
        os.remove(path)
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwbfile)
    rs('Wrote NWB -> {}'.format(path))
    return path
