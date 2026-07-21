#!python3
# -*- coding: utf-8 -*-
"""
Peri-event time histogram (PETH) figure -- NWB variant of figure_peth2.

Reads spikes + per-trial TTL windows from the NWB written by
neural_processing.export_nwb, then plots exactly like figure_peth / figure_peth2.
Because export_nwb now produces its Units and ttl_pulses the same way
figure_peth2 does (sorter output + Open Ephys TTL windows, no origin guessing),
reading them back here reproduces the figure_peth2 figure -- but from the single,
portable neural.nwb rather than the raw sorter/Open Ephys folders.

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

from .. import meta_session
from ..tools.logs import rs
from ..neural_processing import config as npconfig
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH, FILTER_SIGMA,
    read_nwb_spikes_and_ttl, load_timepoints_into_msession, get_trial_data_spike,
    resolve_neuron_selection, _plot_peth)


def plot_perievent_histograms(server, processed_server, session, probe_type,
                              neuron_ids=None, align_timepoint=ALIGN_TIMEPOINT,
                              group_column=GROUP_COLUMN, before=BEFORE, after=AFTER,
                              bin_width=BIN_WIDTH, filter_sigma=FILTER_SIGMA):
    """Plot PETH traces for one session from its NWB (figure_peth2 data path).

    Same arguments and figures as figure_peth/figure_peth2; the neural inputs are
    read from neural.nwb (Units + ttl_pulses) written by export_nwb.

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

    # spikes + per-trial TTL windows from the NWB (figure_peth2-style contents).
    # Pulses carry no trial ids: pulse i is paired to trial i strictly by
    # position, so msession MUST be in recording (chronological) order.
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(cfg.nwb_path)
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
