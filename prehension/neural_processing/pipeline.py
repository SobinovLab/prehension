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
import os

from .. import tools
from .. import meta_session
from ..tools.logs import rs, ws
from . import config
from . import io_streams
from . import preprocessing
from . import spike_sorting
from . import postprocessing
from . import export_phy
from . import export_nwb


def run_necessary(server, processed_server, sessions, temp,
                  nwb_units='all', sorter=config.SORTER_NAME, processes=config.N_JOBS,
                  overwrite=False):
    """Run the necessary chain to produce neural.nwb for one or more sessions.

    Sessions that already have processed neural data (a neural.nwb exists) are skipped unless
    overwrite is True. The probe type is read per session from its meta_structure ('neural'
    field), so mixed-probe session lists are fine. A failure in one session is reported and the
    remaining sessions still run.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        sessions {list[str]} --- Session directory names to process. If empty, all sessions
            found on the server are processed.
        temp {str} --- Folder for local temporary storage (logging).
        nwb_units {str} --- Units written to the NWB: 'all' or 'curated'.
        sorter {str} --- SpikeInterface sorter name.
        processes {int} --- Number of parallel jobs for SpikeInterface.
        overwrite {bool} --- Reprocess and overwrite sessions that already have neural.nwb.
    """
    tools.logs.setup_logging(temp, sessions_dir=server)

    if sessions:
        found_sessions = [s for s in sessions if os.path.isdir(os.path.join(server, s))]
    else:
        found_sessions = meta_session.find_session_dirs(server)
    rs('Found {} session(s) to consider: {}'.format(
        len(found_sessions), ', '.join(found_sessions)))

    failed_sessions = []
    for session in found_sessions:
        # probe type comes from the session meta_structure; sessions without neural data are skipped
        try:
            probe_type = config.probe_type_from_meta(server, processed_server, session)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        cfg = config.NeuralConfig(server, processed_server, session, probe_type,
                                  nwb_units=nwb_units, sorter=sorter, n_jobs=processes)

        # skip sessions whose neural data has already been processed
        if os.path.exists(cfg.nwb_path) and not overwrite:
            rs('Session {} already processed ({} exists); skipping. Use --overwrite to '
               'reprocess.'.format(session, cfg.nwb_path))
            continue

        cfg.ensure_work_folder()
        rs('Session {} ({}) -> {}'.format(session, probe_type, cfg.work_folder))

        try:
            preprocessing.preprocess_recording(cfg)   # load + preprocess (probe-specific)
            spike_sorting.run_spike_sorting(cfg)
            export_nwb.export_nwb(cfg)
        except Exception as e:
            ws('Neural processing failed for session {}: {}'.format(session, repr(e)))
            failed_sessions.append(session)
            continue

        rs('Necessary pipeline finished for session {}.'.format(session))

    if failed_sessions:
        ws('Neural processing failed for {} of {} session(s): {}'.format(
            len(failed_sessions), len(found_sessions), ', '.join(failed_sessions)))


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
