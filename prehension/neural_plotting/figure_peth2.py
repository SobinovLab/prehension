#!python3
# -*- coding: utf-8 -*-
"""
Peri-event time histogram (PETH) figure -- sorter-output variant.

Same PETH as figure_peth.py, but reads its inputs WITHOUT the NWB:
  * spikes come straight from the sorter output (the Phy export with 'noise'
    clusters dropped, else the raw SpikeInterface sorter folder),
  * per-trial TTL windows come straight from the Open Ephys events,

matching the known-good v_probes figure_neural_vprobe_new.py data path.  The
behavioural meta, dup-aware timepoint loading, unit selection and the plotting
itself are reused unchanged from figure_peth.py.

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

from open_ephys.analysis import Session

from .. import meta_session
from ..tools.logs import rs
from ..neural_processing import config as npconfig
from ..neural_processing import export_nwb
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA,
    load_timepoints_into_msession, get_trial_data_spike,
    resolve_neuron_selection, _plot_peth)


# ---------------------------------------------------------------------------
# Neural data (from the sorter output, no NWB)
# ---------------------------------------------------------------------------
def get_neural_spikes(cfg):
    """Load per-unit spike trains (seconds) directly from the sorter output.

    Prefers the Phy export (noise clusters dropped) when one exists, else the raw
    SpikeInterface sorter folder (all units).  Returns
    (spike_per_unit, unit_ids, fs) where spike_per_unit[i] is a sorted array of
    spike times in seconds for unit unit_ids[i].
    """
    phy = cfg.subfolders()['phy']
    if os.path.exists(os.path.join(phy, 'params.py')):
        rs('Spikes: Phy export (noise clusters dropped).')
        sorting = export_nwb._sorting_from_phy(cfg, exclude_groups=('noise',))
    else:
        rs('Spikes: raw sorter folder (all units).')
        sorting = npconfig.load_sorting(cfg)

    fs = sorting.get_sampling_frequency()
    unit_ids = sorted(sorting.get_unit_ids().tolist())
    spike_per_unit = [
        np.sort(np.asarray(sorting.get_unit_spike_train(unit_id=uid), dtype=float) / fs)
        for uid in unit_ids]
    rs('Loaded {} units from the sorter output.'.format(len(unit_ids)))
    return spike_per_unit, unit_ids, fs


def read_oe_event(cfg, fs):
    """Per-trial TTL windows read directly from the Open Ephys events (no NWB).

    Ported from figure_neural_vprobe_new.read_oe_event: consecutive events are
    paired into [start, stop] windows and zeroed to the sorted recording's first
    sample (sample_offset) so they share the spike timebase.  probe_type
    'neuropixels' reads the 'ProbeA-AP' event stream; 'vprobe' reads all events.
    Returns events_time (list of [start, stop] arrays, seconds).
    """
    probe = 'np' if cfg.probe_type == 'neuropixels' else 'v'
    session = Session(str(cfg.oe_folder))
    recording = session.recordnodes[0].recordings[cfg.recording_index]
    events = recording.events
    sample_offset = recording.continuous[0].sample_numbers[0]
    time_offset = sample_offset / fs

    events_time = []
    if probe == 'np':
        events_ap = events[events.stream_name == 'ProbeA-AP']
        for i in range(0, len(events_ap) - 1, 2):
            events_time.append(np.array([
                (events_ap.iloc[i]['sample_number'] - sample_offset) / fs,
                (events_ap.iloc[i + 1]['sample_number'] - sample_offset) / fs]))
    else:
        for i in range(0, len(events) - 1, 2):
            events_time.append(np.array([
                events.iloc[i]['timestamp'] - time_offset,
                events.iloc[i + 1]['timestamp'] - time_offset]))
    return events_time


# ---------------------------------------------------------------------------
# Entry point (same signature as figure_peth.plot_perievent_histograms)
# ---------------------------------------------------------------------------
def plot_perievent_histograms(server, processed_server, session, probe_type,
                              neuron_ids=None, align_timepoint=ALIGN_TIMEPOINT,
                              group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                              bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA):
    """Plot PETH traces for one session directly from the sorter output.

    Same arguments and figures as figure_peth.plot_perievent_histograms, but the
    spikes and TTL windows are read from the sorter output / Open Ephys events
    instead of the NWB.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        neuron_ids {list} --- Unit ids to plot; None/empty -> all units.
        align_timepoint {str} --- Trial timepoint to align to.
        group_column {str} --- Object property to colour-code by.
        before, after {float} --- Window (s) around the alignment timepoint.
        bin_width, filter_sigma {float} --- Binning and smoothing (s).
    """
    cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type)

    # behavioural meta (dup-aware timepoint loading reused from figure_peth)
    mstruct, _, mobject, msession = meta_session.load_meta_information(
        cfg.rserv, cfg.pserv)
    load_timepoints_into_msession(msession, mstruct)

    # neural spikes straight from the sorter output (no NWB)
    spikes, unit_ids, fs = get_neural_spikes(cfg)

    # per-trial TTL windows straight from the Open Ephys events (no NWB).
    # Pulses carry no trial ids: pulse i is paired to trial i strictly by
    # position, so msession MUST be in recording (chronological) order.
    events_time = read_oe_event(cfg, fs)
    if len(events_time) != len(msession):
        raise ValueError(
            '{} TTL pulses but {} behavioural trials. Pulses are matched to trials strictly by '
            'position, so a count mismatch misaligns every trial after the discrepancy and '
            'flattens the PETH. Inspect with figure_ttl_alignment and fix the pulse<->trial '
            'correspondence before plotting.'.format(len(events_time), len(msession)))

    session_spikes = get_trial_data_spike(spikes, events_time)
    num_neurons = len(unit_ids)
    rs('Total {} trials with {} units.'.format(len(msession), num_neurons))
    for trial_spikes, trial_events in zip(session_spikes, events_time):
        ttl_start = trial_events[0]
        for i_n in range(num_neurons):
            trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ttl_start

    for i_trial, trial in zip(range(len(session_spikes)), msession):
        trial.spikes = session_spikes[i_trial]
    msession = [t for t in msession[:len(session_spikes)]
                if t.success and hasattr(t, 'spikes')]

    neuron_selection, neuron_labels = resolve_neuron_selection(unit_ids, neuron_ids)
    rs('Plotting {} unit(s): {}'.format(len(neuron_selection), neuron_labels))

    _plot_peth(msession, mobject, neuron_selection, neuron_labels, align_timepoint,
               group_column, before, after, bin_width, filter_sigma, cfg.work_folder)
