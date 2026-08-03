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
import traceback

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


# Ordered keys of the steps run by run_necessary; used to validate/select --steps.
NECESSARY_STEPS = ('preprocessing', 'spike_sorting', 'export_nwb')


def create_meta_neural(server, processed_server, sessions, temp):
    """Create per-session meta_neural.json (run before run_necessary).

    Writes processed_server/<session>/meta_neural.json with defaults for the
    session's probe type (resolved from meta_structure).  Created once: an
    existing file is left unchanged (there is no overwrite).  Sessions without
    neural data are skipped.  Edit region/burr_hole/depth_um/recording/etc. in the
    file before running run_necessary; the neural steps read their defaults from it.

    The neural/ folder is scanned for Open Ephys recordings: when more than one is
    found, merge_recordings is pre-populated with all of them (concatenated for a
    joint sort by run_necessary); when exactly one is found, merge_recordings is
    left empty and the single recording is printed.  Review/reorder the populated
    list before running run_necessary -- list order is the concatenation order.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        sessions {list[str]} --- Session directory names; empty -> all sessions.
        temp {str} --- Folder for local temporary storage (logging).
    """
    tools.logs.setup_logging(temp, sessions_dir=processed_server)

    if sessions:
        found_sessions = [s for s in sessions if os.path.isdir(os.path.join(server, s))]
    else:
        found_sessions = meta_session.find_session_dirs(server)
    rs('create_meta_neural: {} session(s) to consider: {}'.format(
        len(found_sessions), ', '.join(found_sessions)))

    for session in found_sessions:
        # only sessions whose raw folder actually contains a neural/ folder
        raw_neural_dir = os.path.join(server, session, 'neural')
        if not os.path.isdir(raw_neural_dir):
            ws('Skipping session {}: no raw neural folder at {}.'.format(
                session, raw_neural_dir))
            continue

        # probe type comes from meta_structure; sessions without neural data are skipped
        try:
            probe_type = config.probe_type_from_meta(server, processed_server, session)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        path = config.meta_neural_path(processed_server, session)
        if os.path.exists(path):
            rs('  {}: meta_neural.json already exists; leaving it unchanged.'.format(session))
            continue

        meta = config.default_meta_neural(probe_type)

        # Enumerate every recording under neural/.  More than one -> populate
        # merge_recordings so they are concatenated for a joint sort; exactly one ->
        # leave merge_recordings empty (single-recording path) but report it.
        recordings = config.enumerate_recordings(raw_neural_dir)
        if len(recordings) > 1:
            meta['merge_recordings'] = recordings
            rs('  {}: found {} recordings; populated merge_recordings:'.format(
                session, len(recordings)))
            for r in recordings:
                rs('      {}'.format(config.describe_recording(r)))
        elif len(recordings) == 1:
            rs('  {}: single recording found ({}); merge_recordings left empty.'.format(
                session, config.describe_recording(recordings[0])))
        else:
            ws('  {}: no recordings enumerated under {}; merge_recordings left empty.'.format(
                session, raw_neural_dir))

        config.save_json(meta, path)
        rs('  {}: wrote {}'.format(session, path))


def run_necessary(server, processed_server, sessions, temp,
                  nwb_units=None, sorter=None,
                  processes=config.N_JOBS, overwrite=False, recording=None,
                  steps=None):
    """Run the necessary chain to produce neural.nwb for one or more sessions.

    Each step (preprocessing, spike sorting, NWB export) is skipped when its output already
    exists, unless overwrite is True; this lets a partially-processed session resume from where
    it stopped. The probe type is read per session from its meta_structure ('neural' field), so
    mixed-probe session lists are fine. A failure in one session is reported and the remaining
    sessions still run.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        sessions {list[str]} --- Session directory names to process. If empty, all sessions
            found on the server are processed.
        temp {str} --- Folder for local temporary storage (logging).
        nwb_units {str} --- Units written to the NWB ('noise_excluded'/'curated'/'all');
            None -> meta_neural.json then 'noise_excluded'.
        sorter {str} --- SpikeInterface sorter name; None -> meta_neural.json then kilosort4.
        processes {int} --- Number of parallel jobs for SpikeInterface.
        overwrite {bool} --- Reprocess and overwrite sessions that already have neural.nwb.
        recording {int|str} --- Open Ephys recording within experiment1 to process,
            1-based (Recording1, Recording2, ...); None -> the probe default. Applies
            to every session in this run.
        steps {list[str]} --- Subset of NECESSARY_STEPS ('preprocessing', 'spike_sorting',
            'export_nwb') to run; None or empty runs all of them. Steps always execute in
            pipeline order regardless of the order given here. Note that later steps read
            prior steps' outputs from disk, so running one in isolation requires the earlier
            steps' outputs to already exist.
    """
    if steps:
        unknown = [s for s in steps if s not in NECESSARY_STEPS]
        if unknown:
            raise ValueError(
                'Unknown step(s) {}; valid steps are {}.'.format(
                    ', '.join(unknown), ', '.join(NECESSARY_STEPS)))
        selected_steps = [s for s in NECESSARY_STEPS if s in steps]
    else:
        selected_steps = list(NECESSARY_STEPS)

    tools.logs.setup_logging(temp, sessions_dir=processed_server)

    if sessions:
        found_sessions = [s for s in sessions if os.path.isdir(os.path.join(server, s))]
    else:
        found_sessions = meta_session.find_session_dirs(server)
    rs('Found {} session(s) to consider: {}'.format(
        len(found_sessions), ', '.join(found_sessions)))

    failed_sessions = []
    for session in found_sessions:
        # probe type from meta_structure; the config also loads meta_neural.json.
        # A missing meta_structure OR meta_neural raises ValueError -> skip session.
        try:
            probe_type = config.probe_type_from_meta(server, processed_server, session)
            cfg = config.NeuralConfig(server, processed_server, session, probe_type,
                                      nwb_units=nwb_units, sorter=sorter, n_jobs=processes,
                                      recording=recording)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        cfg.ensure_work_folder()
        rs('Session {} ({}) -> {}'.format(session, probe_type, cfg.work_folder))

        # each step is skipped if its output already exists, unless overwrite is set. This lets a
        # partially-processed session resume from where it stopped. Later steps read prior outputs
        # from disk (see config.load_preprocessed / load_sorting), so reusing them is safe.
        sub = cfg.subfolders()
        step_defs = {
            'preprocessing': ('preprocessing', sub['preprocessed'],
                              lambda: preprocessing.preprocess_recording(cfg)),   # load + preprocess (probe-specific)
            'spike_sorting': ('spike sorting', sub['sorter'],
                              lambda: spike_sorting.run_spike_sorting(cfg)),
            'export_nwb': ('NWB export', cfg.nwb_path,
                           lambda: export_nwb.export_nwb(cfg)),
        }

        try:
            for step_key in selected_steps:
                step_name, output_path, step_fn = step_defs[step_key]
                # when the previous step is not run in this invocation, its output must
                # already exist on disk (this step reads it); fail early if it is missing.
                step_idx = NECESSARY_STEPS.index(step_key)
                if step_idx > 0:
                    prev_key = NECESSARY_STEPS[step_idx - 1]
                    if prev_key not in selected_steps:
                        prev_name, prev_output, _ = step_defs[prev_key]
                        if not os.path.exists(prev_output):
                            raise RuntimeError(
                                '{} requires the output of the previous step, {} ({}), '
                                'which does not exist; run that step first.'.format(
                                    step_name, prev_name, prev_output))
                if not overwrite and os.path.exists(output_path):
                    rs('  {} already done ({}); skipping. Use --overwrite to redo.'.format(
                        step_name, output_path))
                    continue
                step_fn()
        except Exception as e:
            ws('Neural processing failed for session {}: {}'.format(session, repr(e)))
            ws(traceback.format_exc())
            failed_sessions.append(session)
            continue

        rs('Necessary pipeline finished for session {}.'.format(session))

    if failed_sessions:
        ws('Neural processing failed for {} of {} session(s): {}'.format(
            len(failed_sessions), len(found_sessions), ', '.join(failed_sessions)))


def run_diagnostics(server, processed_server, session, probe_type, temp,
                    all_units=False, curate=False, inspect=False,
                    sorter=None, processes=config.N_JOBS,
                    recording=None):
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
        recording {int|str} --- Open Ephys recording within experiment1 to process,
            1-based (Recording1, Recording2, ...); None -> the probe default.
    """
    tools.logs.setup_logging(temp, sessions_dir=processed_server)
    cfg = config.NeuralConfig(server, processed_server, session, probe_type,
                              sorter=sorter, n_jobs=processes, recording=recording)
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
