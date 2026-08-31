#!python3
# -*- coding: utf-8 -*-
"""
Re-export the minimal neural product (neural.nwb) from an already-sorted
`kilosorted.nwb` produced by an external pipeline (the files under a session's
`neural_processed_nwb/` folder), instead of re-running the local Open Ephics ->
SpikeInterface chain.

A `kilosorted.nwb` (Kilosort-sorted, Phy-curated; written by the lab's earlier
export) holds, for one session:
  * a Units table -- row id 0..N-1, ragged `spike_times` (seconds, on the recording
    timebase),
  * a per-unit `extracellular_ephys/electrodes` table (a DynamicTable in older
    exports, an ElectrodesTable in newer ones), one row per unit and positionally
    aligned to the Units table, carrying each unit's original sorter
    id (`unit_id`), Phy curation label (`unit_label`, e.g. 'good' / 'mua'), peak
    channel (`channel_id`) and mean firing rate (`mean_frate`), and
  * a `trials` TimeIntervals (per-trial [start_time, stop_time] + `correct`) on the
    same timebase as the spikes.

This module reads that file and writes the same minimal `neural.nwb` the prehension
neural module consumes (see export_nwb.export_nwb and the reader contract in
neural_processing.common.spikes.read_nwb_spikes_and_ttl):
  * a Units table with `spike_times` (s) and a `unit_id` column (the original sorter
    id), plus the passthrough provenance columns `unit_label` / `channel_id` /
    `mean_frate` (harmless to the reader, which only needs `unit_id`), and
  * a `ttl_pulses` TimeIntervals (per-trial [start_time, stop_time]) taken from the
    source `trials`, so figure_peth3 / the pooling code align spikes to trials
    exactly as they do for a locally-produced neural.nwb.

Only the *selected* units are written: by default the Phy 'good' units (the
`unit_label`); pass a different label set (or None for every unit) to change this.
The kept units are written in ascending probe-depth order, matching
figure_neural.get_phy_data(order_by_depth=1): the depth of each unit is read from the
raw `neural/final_phy/cluster_info.tsv` (keyed by the Phy cluster id, i.e. the unit
`unit_id`) and used only to sort the rows -- the written `unit_id` stays the original
Phy cluster id.  So a downstream `range(n_units)` enumeration of the NWB units walks
them from the probe tip upward, exactly as the figure numbers its neurons.
Spikes and TTL windows share the source timebase, so they are copied through
unchanged -- the downstream code zeroes each trial to its own pulse start, so the
absolute offset does not matter (see pooling.pool_neurons).

A unit `depth` column is written when the depth source (final_phy/cluster_info.tsv)
is available: each unit's Phy/Kilosort template depth (um along the probe), the same
values used to order the rows, so read_nwb_unit_depths() returns them and the
depth-resolved figures work.  This depth is NOT the quantity export_nwb writes (the
template-peak center-of-mass y); the two are comparable in orientation but not
numerically identical, so do not mix depths across the two export paths.  When
cluster_info.tsv is absent, no depth column is written and read_nwb_unit_depths()
returns {} (the source `kilosorted.nwb` itself carries only the peak `channel_id`,
kept as a column).

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

from ..tools.logs import rs, ws, setup_logging


# Layout under a processed session folder (processed_server/<session>/...).  These
# mirror the hard-coded paths in config.NeuralConfig (work_folder='neural_processed',
# nwb_path='neural_processed/neural.nwb'); the source lives alongside them in the
# externally-produced 'neural_processed_nwb' folder.
PROCESSED_NWB_SUBDIR = 'neural_processed_nwb'
KILOSORTED_NAME = 'kilosorted.nwb'
WORK_SUBDIR = 'neural_processed'
NEURAL_NWB_NAME = 'neural.nwb'

# Phy curation output on the raw server (server/<session>/neural/final_phy), holding
# cluster_info.tsv -- the source of per-unit depth used to order the units by depth
# (see read_cluster_depths / figure_neural.get_phy_data).
FINAL_PHY_SUBDIR = os.path.join('neural', 'final_phy')
CLUSTER_INFO_NAME = 'cluster_info.tsv'

# Phy curation labels kept by default ("the selected units in the nwb").  The
# remaining units (typically 'mua') are dropped.  Pass selected_labels=None to keep
# every unit regardless of label.
DEFAULT_SELECTED_LABELS = ('good',)


def kilosorted_path(processed_server, session):
    """Path to a session's source kilosorted.nwb (may not exist)."""
    return os.path.join(processed_server, session, PROCESSED_NWB_SUBDIR, KILOSORTED_NAME)


def neural_nwb_path(processed_server, session):
    """Path to a session's neural.nwb product (same as config.NeuralConfig.nwb_path)."""
    return os.path.join(processed_server, session, WORK_SUBDIR, NEURAL_NWB_NAME)


def final_phy_dir(server, session):
    """Path to a session's Phy-curated final_phy folder on the raw server."""
    return os.path.join(server, session, FINAL_PHY_SUBDIR)


def read_cluster_depths(final_phy_path):
    """Read {cluster_id -> depth_um} from final_phy/cluster_info.tsv (Phy/Kilosort).

    This is the same per-unit depth figure_neural.get_phy_data reads to order units
    by depth: the 'depth' column (um along the probe) keyed by the Phy 'cluster_id'.
    figure_neural indexes the columns positionally (cluster_id=col 0, depth=col 5);
    here they are read by header name, which is equivalent for the standard Phy
    cluster_info.tsv layout and robust to column reordering.

    Returns {} (with a warning) when the file or the needed columns are absent, so
    the export can still run -- it then simply leaves the units in the source order.
    """
    import csv

    path = os.path.join(final_phy_path, CLUSTER_INFO_NAME)
    if not os.path.exists(path):
        ws('cluster_info.tsv not found at {}; units will not be ordered by depth.'.format(path))
        return {}

    depths = {}
    with open(path, newline='') as fd:
        reader = csv.DictReader(fd, delimiter='\t')
        fields = reader.fieldnames or []
        if 'cluster_id' not in fields or 'depth' not in fields:
            ws("cluster_info.tsv at {} lacks 'cluster_id'/'depth' columns (has {}); "
               'no depth ordering.'.format(path, fields))
            return {}
        for row in reader:
            try:
                depths[int(float(row['cluster_id']))] = float(row['depth'])
            except (TypeError, ValueError):
                continue
    rs('cluster_info.tsv: read depth for {} clusters from {}.'.format(len(depths), path))
    return depths


def _cluster_id_of(unit):
    """Integer Phy cluster_id of a unit (its unit_id), or None if not parseable.

    The join key into read_cluster_depths' {cluster_id -> depth} map, used both for
    depth ordering and for the written depth column.
    """
    try:
        return int(round(float(unit['unit_id'])))
    except (TypeError, ValueError):
        return None


def _order_units_by_depth(units, depths):
    """Order kept units by ascending probe depth, as figure_neural does (order_by_depth=1).

    Each unit's depth is looked up by its integer unit_id (the Phy cluster_id) in
    ``depths``.  figure_neural.get_phy_data sorts by ``int(depth.split('.')[0])`` --
    the truncated integer micron depth -- so the same truncation is used here; the
    sort is stable, so units at the same integer depth keep their source order.
    Units without a depth (id not in ``depths``) are placed last, in source order.

    ``depths`` empty -> the units are returned unchanged (no ordering applied).
    """
    if not depths:
        return list(units)

    def _key(u):
        d = depths.get(_cluster_id_of(u))
        return (0, int(d)) if d is not None else (1, 0)

    missing = [u for u in units if _key(u)[0] == 1]
    if missing:
        ws('{} selected unit(s) had no depth in cluster_info.tsv; placed last: {}'.format(
            len(missing), [u['unit_id'] for u in missing]))
    return sorted(units, key=_key)


def _norm_label(value):
    """Normalise a Phy unit label to a lower-case, unpadded string.

    The source stores labels as fixed-width, whitespace-padded text (e.g.
    'good    '); they may arrive as ``bytes`` or ``str`` depending on the reader.
    """
    if isinstance(value, bytes):
        value = value.decode('utf-8', 'replace')
    return str(value).strip().lower()


def _fmt_unit_id(value):
    """Format an original sorter unit id as a string (integral floats -> '12').

    Mirrors export_nwb's ``str(uid)`` so the written `unit_id` values match the ids
    the prehension code (and meta_neural 'good_neurons') already use.
    """
    try:
        f = float(value)
        if np.isfinite(f) and float(int(round(f))) == f:
            return str(int(round(f)))
    except (TypeError, ValueError):
        pass
    return str(value)


ELECTRODES_GROUP = 'general/extracellular_ephys/electrodes'
TRIALS_GROUP = 'intervals/trials'


def _h5_scalar_str(f, key, default=None):
    """Read a scalar / 1-element string dataset from an h5py file, decoded to str.

    NWB stores these as fixed-width bytes, sometimes shape () and sometimes (1,);
    returns ``default`` when the dataset is absent.
    """
    if key not in f:
        return default
    val = f[key][()]
    if isinstance(val, np.ndarray):
        val = val.reshape(-1)[0] if val.size else default
    if isinstance(val, bytes):
        val = val.decode('utf-8', 'replace')
    return default if val is None else str(val)


def _h5_col(group, name, n_expected):
    """Read one electrode/trial column as a numpy array, or None if it does not fit.

    Returns None when the column is absent or its length does not match n_expected,
    so a malformed / partial annotation column is dropped rather than misaligned.
    """
    if group is None or name not in group:
        return None
    arr = np.asarray(group[name][()])
    if arr.ndim != 1 or arr.shape[0] != n_expected:
        ws("column '{}' shape {} != ({},); ignoring it.".format(
            name, arr.shape, n_expected))
        return None
    return arr


def _parse_start_time(value):
    """Parse the NWB session_start_time string to a timezone-aware datetime.

    Falls back to the Unix epoch (UTC) with a warning when absent / unparseable, so
    the NWBFile (which requires a session_start_time) can still be written.
    """
    import datetime

    if value:
        try:
            dt = datetime.datetime.fromisoformat(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except ValueError:
            ws('Could not parse session_start_time {!r}; using the Unix epoch.'.format(value))
    else:
        ws('kilosorted.nwb has no session_start_time; using the Unix epoch.')
    return datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)


def read_kilosorted(source_path):
    """Read the units, per-unit annotations and trials from a kilosorted.nwb.

    Read directly with h5py rather than pynwb: some kilosorted.nwb files type their
    electrodes as the newer ``ElectrodesTable`` neurodata type, whose strict pynwb
    construction requires columns the file does not carry and aborts the whole read
    (ConstructError: "column must have the same number of rows as 'id'").  Reading
    the raw HDF5 datasets sidesteps that schema validation and works uniformly across
    the older (DynamicTable) and newer (ElectrodesTable) exports; the data we need
    (spike times, unit ids/labels, trials) is the same in both.

    All array data is materialised before the file is closed, so the returned dict is
    safe to use afterwards.

    Returns a dict with:
        session_description {str}
        session_start_time  {datetime}
        session_id          {str | None}
        units {list[dict]} --- one entry per source unit, in Units-table row order:
            {'spike_times': ndarray(s), 'unit_id': str, 'unit_label': str,
             'channel_id': float|nan, 'mean_frate': float|nan}
        trials {list[dict]} --- one entry per source trial:
            {'start_time': float, 'stop_time': float, 'correct': float|nan}
    """
    import h5py

    with h5py.File(str(source_path), 'r') as f:
        if 'units/spike_times' not in f or 'units/spike_times_index' not in f:
            raise ValueError('kilosorted.nwb has no Units spike_times: {}'.format(source_path))

        # Ragged spike_times: spike_times_index[i] is the exclusive stop offset of
        # unit i into the flat spike_times; unit i-1's stop is unit i's start (0 for i=0).
        spike_flat = f['units/spike_times'][()].astype(float)
        stops = f['units/spike_times_index'][()].astype('int64')
        n_units = len(stops)
        starts = np.empty_like(stops)
        starts[0] = 0
        starts[1:] = stops[:-1]
        spike_times = [spike_flat[starts[i]:stops[i]] for i in range(n_units)]

        row_ids = (f['units/id'][()] if 'units/id' in f
                   else np.arange(n_units, dtype='int64'))

        # Per-unit annotations (one row per unit, positionally aligned to Units).
        el = f[ELECTRODES_GROUP] if ELECTRODES_GROUP in f else None
        unit_id_col = _h5_col(el, 'unit_id', n_units)
        label_col = _h5_col(el, 'unit_label', n_units)
        channel_col = _h5_col(el, 'channel_id', n_units)
        frate_col = _h5_col(el, 'mean_frate', n_units)

        units = []
        for i in range(n_units):
            uid = unit_id_col[i] if unit_id_col is not None else row_ids[i]
            units.append({
                'spike_times': np.asarray(spike_times[i], dtype=float),
                'unit_id': _fmt_unit_id(uid),
                'unit_label': _norm_label(label_col[i]) if label_col is not None else '',
                'channel_id': float(channel_col[i]) if channel_col is not None else np.nan,
                'mean_frate': float(frate_col[i]) if frate_col is not None else np.nan,
            })

        trials = []
        tr = f[TRIALS_GROUP] if TRIALS_GROUP in f else None
        if tr is not None and 'start_time' in tr and 'stop_time' in tr:
            start_t = tr['start_time'][()].astype(float)
            stop_t = tr['stop_time'][()].astype(float)
            correct_t = tr['correct'][()].astype(float) if 'correct' in tr else None
            for i in range(len(start_t)):
                trials.append({
                    'start_time': float(start_t[i]),
                    'stop_time': float(stop_t[i]),
                    'correct': float(correct_t[i]) if correct_t is not None else np.nan,
                })
        else:
            ws('kilosorted.nwb has no trials table; ttl_pulses will be empty: {}'.format(
                source_path))

        meta = {
            'session_description': _h5_scalar_str(
                f, 'session_description', 'kilosorted re-export'),
            'session_start_time': _parse_start_time(
                _h5_scalar_str(f, 'session_start_time')),
            'session_id': _h5_scalar_str(f, 'general/session_id'),
        }

    rs('kilosorted.nwb: {} units, {} trials read from {}'.format(
        len(units), len(trials), source_path))
    meta['units'] = units
    meta['trials'] = trials
    return meta


def _select_units(units, selected_labels):
    """Keep only units whose (normalised) label is in selected_labels; None -> all.

    Returns (kept_units, dropped_count).  Raises ValueError if a label filter is
    requested but the source carries no labels, or if nothing survives it.
    """
    if selected_labels is None:
        return list(units), 0

    wanted = {_norm_label(l) for l in selected_labels}
    have_labels = any(u['unit_label'] for u in units)
    if not have_labels:
        raise ValueError(
            'selected_labels={} requested but the source has no unit labels; pass '
            'selected_labels=None to keep all units.'.format(sorted(wanted)))

    kept = [u for u in units if u['unit_label'] in wanted]
    dropped = len(units) - len(kept)
    if not kept:
        present = sorted({u['unit_label'] for u in units})
        raise ValueError('No units matched labels {} (labels present: {}).'.format(
            sorted(wanted), present))
    return kept, dropped


def export_neural_nwb_from_kilosorted(source_path, out_path,
                                      selected_labels=DEFAULT_SELECTED_LABELS,
                                      overwrite=True, final_phy_path=None):
    """Write a minimal neural.nwb from one kilosorted.nwb, keeping the selected units.

    The kept units are written in ascending probe-depth order, matching
    figure_neural.get_phy_data(order_by_depth=1): the depth of each unit is read from
    ``final_phy_path``/cluster_info.tsv (keyed by the Phy cluster_id, i.e. the unit
    unit_id) and the units are sorted by it, so a downstream `range(n_units)`
    enumeration of the NWB units walks them from the probe tip upward exactly as the
    figure does.  The written unit_id stays the original Phy cluster id (the label);
    only the row order changes.  When ``final_phy_path`` is None or has no
    cluster_info.tsv, the units keep their source order (with a warning).

    When cluster_info.tsv is available, each unit also gets a `depth` column (its
    Phy/Kilosort template depth in um, the same values used for the ordering) so
    read_nwb_unit_depths() returns them; this depth differs from export_nwb's
    template-peak center-of-mass y (see the module docstring).

    Arguments:
        source_path {str} --- path to the source kilosorted.nwb.
        out_path {str} --- path to write neural.nwb (its folder is created).
        selected_labels {tuple[str] | None} --- Phy labels to keep (default the
            'good' units); None keeps every unit.
        overwrite {bool} --- overwrite an existing out_path (default True; the local
            pipeline always overwrites the product).
        final_phy_path {str | None} --- path to the session's final_phy folder
            (cluster_info.tsv) used to order the units by depth; None -> no depth
            ordering (source order preserved).

    Returns out_path.
    """
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.epoch import TimeIntervals

    if not os.path.exists(source_path):
        raise FileNotFoundError('kilosorted.nwb not found: {}'.format(source_path))
    if os.path.exists(out_path) and not overwrite:
        rs('neural.nwb already exists ({}); skipping (overwrite=False).'.format(out_path))
        return out_path

    data = read_kilosorted(source_path)
    kept, dropped = _select_units(data['units'], selected_labels)
    rs('Selected {} unit(s){}; {} dropped.'.format(
        len(kept),
        '' if selected_labels is None else ' with label(s) {}'.format(
            sorted({_norm_label(l) for l in selected_labels})),
        dropped))

    # Order units by depth (figure_neural.get_phy_data order_by_depth=1): row index i
    # in the NWB then walks the probe from tip upward, so a range(n) enumeration of
    # the units matches the figure's neuron ordering.
    depths = read_cluster_depths(final_phy_path) if final_phy_path else {}
    if depths:
        kept = _order_units_by_depth(kept, depths)
        rs('Ordered {} unit(s) by ascending probe depth.'.format(len(kept)))
    else:
        ws('No depth source ({}); writing units in kilosorted.nwb source order.'.format(
            'final_phy_path not given' if not final_phy_path
            else os.path.join(final_phy_path, CLUSTER_INFO_NAME)))

    nwbfile = NWBFile(
        session_description=data['session_description'],
        identifier=str(uuid4()),
        session_start_time=data['session_start_time'],
        session_id=data['session_id'])

    # Units: spike_times (s) + the original sorter unit id, matching export_nwb.  The
    # source label / peak channel / mean rate ride along as extra columns for
    # provenance (the reader only needs unit_id).
    #
    # depth column: written only when cluster_info.tsv was available (`depths`), from
    # its per-unit 'depth' (um along the probe, keyed by the Phy cluster_id) -- the
    # same values used to order the units above, so read_nwb_unit_depths() now returns
    # them and the depth-resolved figures work.  NOTE this depth is the Phy/Kilosort
    # template depth (larger = farther from the probe tip); it is NOT the same
    # quantity export_nwb writes, which is the center-of-mass y of the template peak
    # amplitude over the channel positions.  The two are comparable in orientation but
    # not numerically identical, so do not mix depths from the two export paths.
    nwbfile.add_unit_column(name='unit_id', description='original sorter unit id')
    nwbfile.add_unit_column(name='unit_label',
                            description='Phy curation label from kilosorted.nwb (e.g. good/mua)')
    nwbfile.add_unit_column(name='channel_id',
                            description='peak channel id of the unit in kilosorted.nwb')
    nwbfile.add_unit_column(name='mean_frate',
                            description='mean firing rate (Hz) of the unit in kilosorted.nwb')
    if depths:
        nwbfile.add_unit_column(
            name='depth',
            description='unit depth along the probe (um): the Phy/Kilosort template '
                        "depth from final_phy/cluster_info.tsv 'depth' (larger = farther "
                        'from the probe tip). Differs from export_nwb, which writes the '
                        'template peak center-of-mass y.')
    for u in kept:
        kwargs = dict(spike_times=np.asarray(u['spike_times'], dtype=float),
                      unit_id=u['unit_id'],
                      unit_label=u['unit_label'],
                      channel_id=float(u['channel_id']),
                      mean_frate=float(u['mean_frate']))
        if depths:
            kwargs['depth'] = float(depths.get(_cluster_id_of(u), np.nan))
        nwbfile.add_unit(**kwargs)
    print('Added {} units{}.'.format(
        len(kept), ' with depth' if depths else ' (no depth column)'))

    # TTL windows: the source per-trial [start, stop], copied through on the same
    # timebase as the spikes (see module docstring).  'correct' rides along.
    ttl_ti = TimeIntervals(
        name='ttl_pulses',
        description='Per-trial windows copied from kilosorted.nwb trials; '
                    'start_time/stop_time on the same timebase as the spikes.')
    ttl_ti.add_column(name='correct',
                      description='trial correctness flag from kilosorted.nwb trials')
    for t in data['trials']:
        ttl_ti.add_row(start_time=t['start_time'], stop_time=t['stop_time'],
                       correct=float(t['correct']))
    nwbfile.add_time_intervals(ttl_ti)
    print('Added ttl_pulses with {} pulses.'.format(len(data['trials'])))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    with NWBHDF5IO(str(out_path), 'w') as fio:
        fio.write(nwbfile)
    rs('Wrote NWB -> {}'.format(out_path))
    return out_path


def reexport_kilosorted(server, processed_server, sessions, temp,
                        selected_labels=DEFAULT_SELECTED_LABELS, overwrite=False,
                        order_by_depth=True):
    """Re-export neural.nwb from kilosorted.nwb for one or more sessions.

    For each session, reads processed_server/<session>/neural_processed_nwb/
    kilosorted.nwb and writes processed_server/<session>/neural_processed/neural.nwb
    (the path the prehension neural module reads).  The units are written in
    ascending probe-depth order (figure_neural.get_phy_data order_by_depth=1) using
    the depth in server/<session>/neural/final_phy/cluster_info.tsv, unless
    order_by_depth is False or that file is missing.  A session without a
    kilosorted.nwb is skipped with a warning; a session whose neural.nwb already
    exists is skipped unless overwrite is True.  A failure in one session is reported
    and the remaining sessions still run.

    Arguments:
        server {str} --- Raw server folder (holds <session>/neural/final_phy with
            cluster_info.tsv, the per-unit depth source for depth ordering).
        processed_server {str} --- Folder holding the processed sessions.
        sessions {list[str]} --- Session directory names.  Empty -> every session
            under processed_server that has a kilosorted.nwb.
        temp {str} --- Folder for local temporary storage (logging).
        selected_labels {tuple[str] | None} --- Phy labels to keep (default 'good');
            None keeps every unit.
        overwrite {bool} --- Overwrite sessions that already have neural.nwb.
        order_by_depth {bool} --- Order the written units by ascending probe depth
            (default True); False writes them in kilosorted.nwb source order.
    """
    setup_logging(temp, sessions_dir=processed_server)

    if sessions:
        found = list(sessions)
    else:
        found = sorted(
            d for d in os.listdir(processed_server)
            if os.path.exists(kilosorted_path(processed_server, d)))
    rs('reexport_kilosorted: {} session(s) to consider: {}'.format(
        len(found), ', '.join(found)))

    failed = []
    for session in found:
        source = kilosorted_path(processed_server, session)
        out = neural_nwb_path(processed_server, session)
        phy = final_phy_dir(server, session) if order_by_depth else None
        if not os.path.exists(source):
            ws('Skipping session {}: no kilosorted.nwb at {}.'.format(session, source))
            continue
        if os.path.exists(out) and not overwrite:
            rs('  {}: neural.nwb already exists ({}); skipping. '
               'Use --overwrite to redo.'.format(session, out))
            continue
        try:
            rs('Session {} -> {}'.format(session, out))
            export_neural_nwb_from_kilosorted(
                source, out, selected_labels=selected_labels, overwrite=True,
                final_phy_path=phy)
            rs('Re-export finished for session {}.'.format(session))
        except Exception as e:  # noqa: BLE001
            import traceback
            ws('Re-export failed for session {}: {}'.format(session, repr(e)))
            ws(traceback.format_exc())
            failed.append(session)

    if failed:
        ws('Re-export failed for {} of {} session(s): {}'.format(
            len(failed), len(found), ', '.join(failed)))
