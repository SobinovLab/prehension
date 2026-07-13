#!python3
# -*- coding: utf-8 -*-
"""
Top-level neural processing pipelines for a single session.

run_necessary --- the minimum chain to produce the NWB product:
    load_recording -> preprocess_recording -> run_spike_sorting -> export_nwb
    (export_nwb extracts the TTL sync internally).
run_diagnostics --- optional diagnostics + curation, run after run_necessary.

Both build a NeuralConfig for the session and dispatch the probe-specific
versioned steps by probe_type.

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
from .. import tools
from ..tools.logs import rs
from . import config
from . import io_streams
from . import preprocessing
from . import spike_sorting
from . import postprocessing
from . import export_phy
from . import export_nwb


def run_necessary(server, processed_server, session, probe_type, temp,
                  nwb_units='all', sorter=config.SORTER_NAME, processes=config.N_JOBS):
    """Run the necessary chain to produce neural.nwb for one session.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        temp {str} --- Folder for local temporary storage (logging).
        nwb_units {str} --- Units written to the NWB: 'all' or 'curated'.
        sorter {str} --- SpikeInterface sorter name.
        processes {int} --- Number of parallel jobs for SpikeInterface.
    """
    tools.logs.setup_logging(temp, sessions_dir=server)
    cfg = config.NeuralConfig(server, processed_server, session, probe_type,
                              nwb_units=nwb_units, sorter=sorter, n_jobs=processes)
    cfg.ensure_work_folder()
    rs('Session {} ({}) -> {}'.format(session, probe_type, cfg.work_folder))

    preprocessing.preprocess_recording(cfg)   # load + preprocess (probe-specific)
    spike_sorting.run_spike_sorting(cfg)
    export_nwb.export_nwb(cfg)

    rs('Necessary pipeline finished for session {}.'.format(session))


def run_diagnostics(server, processed_server, session, probe_type, temp,
                    all_units=False, curate=False, inspect=False,
                    sorter=config.SORTER_NAME, processes=config.N_JOBS):
    """Run optional diagnostics + curation for one session.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        temp {str} --- Folder for local temporary storage (logging).
        all_units {bool} --- Report on all units (full analyzer).
        curate {bool} --- Run the quality triage -> analyzer_curated.
        inspect {bool} --- Also run stream/event inspection first.
        sorter {str} --- SpikeInterface sorter name.
        processes {int} --- Number of parallel jobs for SpikeInterface.
    """
    tools.logs.setup_logging(temp, sessions_dir=server)
    cfg = config.NeuralConfig(server, processed_server, session, probe_type,
                              sorter=sorter, n_jobs=processes)
    cfg.ensure_work_folder()
    rs('Diagnostics for session {} ({}).'.format(session, probe_type))

    if inspect:
        io_streams.inspect_streams(cfg)

    postprocessing.build_sorting_analyzer(cfg)
    postprocessing.compute_quality_metrics(cfg)

    if curate:
        postprocessing.curate_by_quality(cfg)

    export_phy.export_to_phy(cfg)
    export_phy.export_report(cfg, all_units=all_units)

    rs('Diagnostics finished for session {}.'.format(session))
