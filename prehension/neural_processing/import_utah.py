#!python3
# -*- coding: utf-8 -*-
"""
Re-export the minimal neural product(s) for the Utah-array datasets (mojito / pimms
right hemisphere) from the externally-produced Blackrock/Plexon NWBs.

Those M1 sessions were recorded on a 10x10 Blackrock Utah Electrode Array (Cerebus),
threshold-crossed in Central, and sorted in Plexon Offline Sorter (and, for a few
sessions, Kilosort/Phy).  The earlier lab pipeline exported them, under a session's
``neural_processed_nwb/`` folder (usually on the raw server), to several NWBs of the
same family the kilosort re-export already reads (a Units table with ragged
``spike_times`` + ``spike_times_index``, an ``intervals/trials`` TimeIntervals, and a
per-unit ``extracellular_ephys/electrodes`` table), but with varying column names and
Utah row/col geometry instead of a probe depth:
  * sorted units:        manually_sorted.nwb / plexonsorted.nwb / kilosorted.nwb
  * threshold crossings: raw_threshold_crossings.nwb / unsorted.nwb (one row per channel)

This module writes, for one session, the same minimal NWBs the prehension neural
module consumes (see export_nwb / import_kilosorted and the reader contract in
neural_processing.common.spikes.read_nwb_spikes_and_ttl):
  * ``neural_processed/neural.nwb``                     from the best sorted source, and
  * ``neural_processed/neural_threshold_crossings.nwb`` from the threshold-crossing source,

each a Units table (``spike_times`` (s) + a ``unit_id``, plus the passthrough columns
``unit_label`` / ``channel_id`` / ``channel_label`` / ``mean_frate`` and, so the Utah
coordinates are preserved, per-unit ``rel_x`` / ``rel_y`` on the array grid) and a
``ttl_pulses`` TimeIntervals copied from the source trials.  A best-effort electrodes
table (one row per channel with the in-plane Utah coordinates x=col*400um, y=row*400um)
records the channel geometry.  The h5py-based reader (import_kilosorted.read_kilosorted)
tolerates the source column-name variants and missing geometry.

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
from .common.probe import UTAH_PITCH_UM
from .import_kilosorted import (
    PROCESSED_NWB_SUBDIR, read_kilosorted, neural_nwb_path,
    threshold_crossings_nwb_path)


# Utah 10x10 electrode array: UTAH_PITCH_UM contact pitch (from common.probe; the source
# stores geometry as the 0-based channel row/col on that grid, converted to in-plane um).
UTAH_DEVICE_DESC = '10x10 Utah Electrode Array, 400 um spacing'

# Source filename precedence.  Sorted: manual Plexon sort (the manuscript's sorter) is
# preferred, then its corrected-columns re-export, then Kilosort/Phy.  Threshold
# crossings: the explicit crossings file, then the unsorted (unit 255 dropped) file.
SORTED_SOURCES = ('manually_sorted.nwb', 'plexonsorted.nwb', 'kilosorted.nwb')
TX_SOURCES = ('raw_threshold_crossings.nwb', 'unsorted.nwb')

# The Utah-array presets, by canonical name and every alias, so reexport_kilosorted_nwb
# can pick this re-export path from the preset name regardless of which alias was passed.
UTAH_PRESETS = frozenset({
    'mojito_right_hemisphere', 'mrhem', 'mojito_right_hem',
    'pimms_right_hemisphere', 'prhem', 'pimms', 'pimms_right_hem'})


def is_utah_preset(name):
    """True when the preset name/alias is one of the Utah-array datasets."""
    return name in UTAH_PRESETS


def _readable_units_nwb(path):
    """True if `path` opens with h5py and carries a Units spike_times (else warn/False)."""
    import h5py

    try:
        with h5py.File(str(path), 'r') as f:
            return 'units/spike_times' in f and 'units/spike_times_index' in f
    except Exception as e:  # noqa: BLE001 --- corrupt / partial write, try the next candidate
        ws('Unreadable NWB {} ({}); skipping.'.format(path, repr(e)))
        return False


def find_utah_source(server, processed_server, session, candidates):
    """First readable source NWB for a session among `candidates` (raw then processed).

    Searches <server>/<session>/neural_processed_nwb and the processed mirror in the
    filename precedence order, returning the first file that opens and carries a Units
    spike_times; corrupt/partial files are skipped with a warning.  None when none match.
    """
    dirs = [os.path.join(server, session, PROCESSED_NWB_SUBDIR),
            os.path.join(processed_server, session, PROCESSED_NWB_SUBDIR)]
    for name in candidates:
        for d in dirs:
            path = os.path.join(d, name)
            if os.path.exists(path) and _readable_units_nwb(path):
                return path
    return None


def _utah_xy(u):
    """In-plane Utah coordinates (x=col*pitch, y=row*pitch) in um, or (nan, nan)."""
    row, col = u.get('channel_row', np.nan), u.get('channel_col', np.nan)
    if np.isfinite(row) and np.isfinite(col):
        return col * UTAH_PITCH_UM, row * UTAH_PITCH_UM
    return np.nan, np.nan


def _unit_id_for(u, one_per_channel):
    """A unique unit id for one source row, tolerant of the source kind.

    Kilosort/Phy rows carry a globally-unique cluster id in `unit_id` and a `unit_label`,
    so that id is used directly.  Threshold-crossing files have one row per channel (and a
    meaningless `unit_id` of 0), so the channel label (e.g. 'elec78') is the id.  Plexon
    files number units per channel, so the id disambiguates with the channel label
    ('elec78_1').  Falls back to the source `unit_id` when there is no channel label.
    """
    if u['unit_label']:
        return u['unit_id']
    if u['channel_label']:
        return u['channel_label'] if one_per_channel \
            else '{}_{}'.format(u['channel_label'], u['unit_id'])
    return u['unit_id']


def _unique_unit_ids(ids):
    """Disambiguate any duplicate unit ids by suffixing repeats (id, id_1, id_2, ...).

    A safety net for degenerate sources (e.g. a threshold-crossing file without channel
    labels, where every row would otherwise share the same id); normal sources are
    already unique and pass through unchanged.
    """
    seen, out = {}, []
    for uid in ids:
        if uid in seen:
            seen[uid] += 1
            out.append('{}_{}'.format(uid, seen[uid]))
        else:
            seen[uid] = 0
            out.append(uid)
    return out


def _add_utah_electrodes(nwbfile, eg, units):
    """Best-effort electrodes table: one row per unique channel with Utah coordinates.

    Adds x=col*pitch, y=row*pitch (um), z=0 plus channel_id/label and the 0-based grid
    row/col for every channel that carries geometry.  A schema problem degrades to no
    table (units still carry their own rel_x/rel_y), matching export_nwb.add_electrodes.
    Returns the number of electrodes written.
    """
    try:
        nwbfile.add_electrode_column(name='channel_id', description='Blackrock channel number')
        nwbfile.add_electrode_column(name='channel_label', description="channel label (e.g. 'elec78')")
        nwbfile.add_electrode_column(name='rel_row', description='0-based Utah grid row')
        nwbfile.add_electrode_column(name='rel_col', description='0-based Utah grid column')
        seen = set()
        for u in units:
            cid = u['channel_id']
            x, y = _utah_xy(u)
            if not np.isfinite(x) or not np.isfinite(cid) or cid in seen:
                continue
            seen.add(cid)
            nwbfile.add_electrode(
                x=float(x), y=float(y), z=0.0, location='M1', group=eg, group_name='array0',
                channel_id=float(cid), channel_label=u['channel_label'],
                rel_row=float(u['channel_row']), rel_col=float(u['channel_col']))
        print('Added {} Utah electrodes.'.format(len(seen)))
        return len(seen)
    except Exception as e:  # noqa: BLE001
        ws('Skipping electrodes table (schema mismatch): {}'.format(e))
        return 0


def write_utah_nwb(units, trials, meta, out_path, threshold_crossings=False):
    """Write one minimal Utah neural NWB (sorted units or threshold crossings).

    Arguments:
        units {list[dict]} --- units from read_kilosorted (kept as-is; no label filtering,
            since the Plexon / threshold sources have no curation labels).
        trials {list[dict]} --- source trials -> ttl_pulses.
        meta {dict} --- read_kilosorted meta (session_description/start_time/id).
        out_path {str} --- path to write (its folder is created).
        threshold_crossings {bool} --- only annotates the units-table description.

    Returns out_path.
    """
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.epoch import TimeIntervals

    kind = 'threshold crossings' if threshold_crossings else 'sorted units'
    nwbfile = NWBFile(
        session_description=meta['session_description'],
        identifier=str(uuid4()),
        session_start_time=meta['session_start_time'],
        session_id=meta['session_id'])

    device = nwbfile.create_device(name='array', description=UTAH_DEVICE_DESC,
                                   manufacturer='Blackrock')
    eg = nwbfile.create_electrode_group(name='array0', description='Utah electrode array',
                                        location='M1', device=device)
    n_electrodes = _add_utah_electrodes(nwbfile, eg, units)

    # One row per channel (unique channel labels == rows) -> the channel label is the
    # unit id; otherwise a per-channel unit number is disambiguated with the label.
    have_labels = any(u['unit_label'] for u in units)
    ch_labels = [u['channel_label'] for u in units if u['channel_label']]
    one_per_channel = bool(ch_labels) and len(set(ch_labels)) == len(units)

    nwbfile.add_unit_column(name='unit_id', description='original sorter / channel unit id')
    if have_labels:
        nwbfile.add_unit_column(name='unit_label', description='curation label (e.g. good/mua)')
    nwbfile.add_unit_column(name='channel_id', description='Blackrock channel number of the unit')
    nwbfile.add_unit_column(name='channel_label', description="channel label (e.g. 'elec78')")
    nwbfile.add_unit_column(name='mean_frate', description='mean firing rate (Hz) from the source')
    nwbfile.add_unit_column(name='rel_x', description='unit x on the Utah grid (um, col*pitch)')
    nwbfile.add_unit_column(name='rel_y', description='unit y on the Utah grid (um, row*pitch)')
    uids = _unique_unit_ids([_unit_id_for(u, one_per_channel) for u in units])
    n_coord = 0
    for uid, u in zip(uids, units):
        x, y = _utah_xy(u)
        n_coord += int(np.isfinite(x))
        kwargs = dict(spike_times=np.asarray(u['spike_times'], dtype=float),
                      unit_id=uid,
                      channel_id=float(u['channel_id']),
                      channel_label=u['channel_label'],
                      mean_frate=float(u['mean_frate']),
                      rel_x=float(x), rel_y=float(y))
        if have_labels:
            kwargs['unit_label'] = u['unit_label']
        nwbfile.add_unit(**kwargs)
    print('Added {} {} ({} with coordinates, {} electrodes).'.format(
        len(units), kind, n_coord, n_electrodes))

    ttl_ti = TimeIntervals(
        name='ttl_pulses',
        description='Per-trial windows copied from the source trials; start_time/stop_time '
                    'on the same timebase as the spikes.')
    ttl_ti.add_column(name='correct', description='trial correctness flag from the source trials')
    for t in trials:
        ttl_ti.add_row(start_time=t['start_time'], stop_time=t['stop_time'],
                       correct=float(t['correct']))
    nwbfile.add_time_intervals(ttl_ti)
    print('Added ttl_pulses with {} pulses.'.format(len(trials)))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
    with NWBHDF5IO(str(out_path), 'w') as fio:
        fio.write(nwbfile)
    rs('Wrote NWB -> {}'.format(out_path))
    return out_path


def export_utah_nwb(source_path, out_path, threshold_crossings=False, overwrite=True):
    """Write one Utah neural NWB (sorted or threshold crossings) from a source file."""
    if not os.path.exists(source_path):
        raise FileNotFoundError('source not found: {}'.format(source_path))
    if os.path.exists(out_path) and not overwrite:
        rs('{} already exists; skipping (overwrite=False).'.format(out_path))
        return out_path
    data = read_kilosorted(source_path)
    return write_utah_nwb(data['units'], data['trials'], data, out_path,
                          threshold_crossings=threshold_crossings)


def reexport_utah(server, processed_server, sessions, temp,
                  include_threshold_crossings=True, overwrite=False):
    """Re-export neural.nwb (+ threshold crossings) for one or more Utah sessions.

    For each session, writes processed_server/<session>/neural_processed/neural.nwb from
    the best sorted source and, when include_threshold_crossings, .../neural_threshold_
    crossings.nwb from the threshold-crossing source (see SORTED_SOURCES / TX_SOURCES;
    sources are searched on the raw server first, then the processed mirror).  A session
    with only a threshold-crossing source still gets that file (with a warning that no
    sorted source was found); one whose product already exists is skipped unless
    overwrite.  A failure in one session is reported and the rest still run.

    Arguments:
        server {str} --- Raw server folder (holds <session>/neural_processed_nwb sources).
        processed_server {str} --- Folder holding the processed sessions (outputs).
        sessions {list[str]} --- Session directory names.  Empty -> every session under
            server that has a neural_processed_nwb folder.
        temp {str} --- Folder for local temporary storage (logging).
        include_threshold_crossings {bool} --- also write the threshold-crossing NWB.
        overwrite {bool} --- overwrite products that already exist.
    """
    setup_logging(temp, sessions_dir=processed_server)

    if sessions:
        found = list(sessions)
    else:
        found = sorted(
            d for d in os.listdir(server)
            if os.path.isdir(os.path.join(server, d, PROCESSED_NWB_SUBDIR)))
    rs('reexport_utah: {} session(s) to consider: {}'.format(len(found), ', '.join(found)))

    failed = []
    for session in found:
        try:
            wrote_any = False

            sorted_src = find_utah_source(server, processed_server, session, SORTED_SOURCES)
            out_sorted = neural_nwb_path(processed_server, session)
            if sorted_src is None:
                ws('  {}: no sorted source among {}.'.format(session, SORTED_SOURCES))
            elif os.path.exists(out_sorted) and not overwrite:
                rs('  {}: neural.nwb already exists; skipping (use --overwrite).'.format(session))
            else:
                rs('Session {} sorted -> {}'.format(session, out_sorted))
                export_utah_nwb(sorted_src, out_sorted, threshold_crossings=False,
                                overwrite=True)
                wrote_any = True

            if include_threshold_crossings:
                tx_src = find_utah_source(server, processed_server, session, TX_SOURCES)
                out_tx = threshold_crossings_nwb_path(processed_server, session)
                if tx_src is None:
                    ws('  {}: no threshold-crossing source among {}.'.format(session, TX_SOURCES))
                elif os.path.exists(out_tx) and not overwrite:
                    rs('  {}: threshold crossings already exist; skipping.'.format(session))
                else:
                    rs('Session {} threshold crossings -> {}'.format(session, out_tx))
                    export_utah_nwb(tx_src, out_tx, threshold_crossings=True, overwrite=True)
                    wrote_any = True

            if not wrote_any:
                ws('Session {}: nothing re-exported.'.format(session))
            else:
                rs('Re-export finished for session {}.'.format(session))
        except Exception as e:  # noqa: BLE001
            import traceback
            ws('Re-export failed for session {}: {}'.format(session, repr(e)))
            ws(traceback.format_exc())
            failed.append(session)

    if failed:
        ws('Re-export failed for {} of {} session(s): {}'.format(
            len(failed), len(found), ', '.join(failed)))
