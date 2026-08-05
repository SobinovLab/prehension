#!python3
# -*- coding: utf-8 -*-
"""
Per-session configuration for neural spike sorting (Open Ephys -> minimal NWB), for
both Neuropixels and 32-channel V-probe recordings.

Per-session paths are resolved from a server/processed_server/session triple (as
elsewhere in prehension): raw neural = server/session/neural, processed output =
processed_server/session/neural_processed.  The preset and session are supplied
on the command line by the calling scripts.

Only two processing steps differ between probes (recording load and
preprocessing).  Probe-scoped parameters live in PROBE_DEFAULTS below; the larger
per-dataset V-probe wiring lives in this local config file.  A NeuralConfig
instance carries the resolved values for one session and is passed to the step
functions.

The reusable stream / probe / event / Open-Ephys helpers this config used to hold
now live in neural_processing.common (openephys, streams, probe, events); this
module keeps the NeuralConfig class, the meta_neural defaults and the saved-folder
loaders.

Setup recommendations:
# Conda packages:
conda create -n prehension_si -c conda-forge python=3.11 pynwb hdmf numpy scipy pandas matplotlib pyarrow h5py
conda activate prehension_si

# Pip packages
# sorter(s) and the behavioural module (pip; Kilosort4 needs a CUDA GPU):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade kilosort spikeinterface probeinterface

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
import copy

from ..tools import io
from ..tools.cmd_args import resolve_meta_arg
from .common import openephys, streams


# ---------------------------------------------------------------------------
# Shared defaults
# ---------------------------------------------------------------------------
BLOCK_INDEX = 0
SORTER_NAME = 'kilosort4'
N_JOBS = 14

COMMON_REFERENCE_OPERATOR = 'median'
COMMON_REFERENCE_TYPE = 'global'

# Quality-metric triage (used only by curate_by_quality / --nwb_units curated).
CURATION_QUERY = (
    '(amplitude_cutoff < 0.2) & '
    '(isi_violations_ratio < 1.0) & '
    '(presence_ratio > 0.8) & '
    '(snr > 4.0)')

# Probe-scoped parameters read by the two versioned step functions.
PROBE_DEFAULTS = {
    'neuropixels': {
        'recording_index': 0,       # recording1
        'highpass_freq_min': 400.0,
        'use_phase_shift': True,
        'sparse': True,             # sparse waveforms for high channel counts
        'expected_n_channels': None,  # do not enforce a channel count
    },
    'vprobe': {
        'recording_index': 0,       # recording1
        'highpass_freq_min': 300.0,
        'use_phase_shift': False,   # phase_shift is Neuropixels-specific
        'sparse': False,            # full waveforms fine for 32 channels
        'expected_n_channels': 32,
    },
}


# ---------------------------------------------------------------------------
# V-probe geometry and wiring (only used when probe_type == 'vprobe')
# ---------------------------------------------------------------------------
# A Plexon-style V-Probe is a single-shank linear array; default is a single
# column of contacts at CONTACT_PITCH_UM spacing.
GEOMETRY = 'linear'              # 'linear' | 'staggered'
CONTACT_PITCH_UM = 100.0
HORIZONTAL_PITCH_UM = 30.0
CONTACT_RADIUS_UM = 7.5
# Channel name ('CHn') wired to each contact, tip(0) -> top(N-1).  Replace with
# the real V-probe + headstage wiring; this is too large / dataset-specific for
# the command line.
CONTACT_CHANNEL_NAMES = [
    'CH24', 'CH23', 'CH22', 'CH21', 'CH20', 'CH19', 'CH18', 'CH17',
    'CH16', 'CH15', 'CH14', 'CH13', 'CH12', 'CH11', 'CH10', 'CH9',
    'CH25', 'CH26', 'CH27', 'CH28', 'CH29', 'CH30', 'CH31', 'CH32',
    'CH1', 'CH2', 'CH3', 'CH4', 'CH5', 'CH6', 'CH7', 'CH8']


META_NEURAL_NAME = 'meta_neural.json'


def meta_neural_path(processed_server, session):
    return os.path.join(processed_server, session, META_NEURAL_NAME)


def default_meta_neural(probe_type):
    """Default meta_neural dict for a probe type, from the module constants.

    Captures the full per-session neural config so a session's meta_neural.json is
    a complete, editable snapshot: annotations (region/burr_hole/depth_um/notes),
    the CLI-backed args (recording/skip_ttl/sorter/nwb_units), the shared
    processing constants, and the resolved PROBE_DEFAULTS.  The V-probe geometry /
    wiring block is added only for probe_type == 'vprobe'.
    """
    if probe_type not in PROBE_DEFAULTS:
        raise ValueError('Unknown probe_type {!r} (expected {}).'.format(
            probe_type, list(PROBE_DEFAULTS)))
    d = PROBE_DEFAULTS[probe_type]
    meta = {
        'probe_type': probe_type,
        'region': '',
        'burr_hole': '',
        'depth_um': '',
        'notes': '',
        'good_neurons': [],   # unit ids to plot when figure_peth is run with --only_good
        'recording': d['recording_index'] + 1,   # 1-based Open Ephys recording number
        # Optional list of recordings to concatenate into one dataset for a joint
        # spike sort. Empty or a single entry -> the single-recording path above
        # (uses 'recording'). Each entry may set any of:
        #   folder     -- timestamped Open Ephys folder under neural/ (default: auto)
        #   experiment -- 1-based experimentN -> block_index (default: 1)
        #   recording  -- 1-based RecordingN -> segment    (default: probe default)
        # List order is the concatenation (and trial) order. See streams.merge_timeline.
        'merge_recordings': [],
        'skip_ttl': 0,
        'skip_ttl_last': 0,
        'sorter': SORTER_NAME,
        'nwb_units': 'noise_excluded',
        'block_index': BLOCK_INDEX,
        'common_reference_operator': COMMON_REFERENCE_OPERATOR,
        'common_reference_type': COMMON_REFERENCE_TYPE,
        'curation_query': CURATION_QUERY,
        'highpass_freq_min': d['highpass_freq_min'],
        'use_phase_shift': d['use_phase_shift'],
        'sparse': d['sparse'],
        'expected_n_channels': d['expected_n_channels'],
    }
    if probe_type == 'vprobe':
        meta.update({
            'geometry': GEOMETRY,
            'contact_pitch_um': CONTACT_PITCH_UM,
            'horizontal_pitch_um': HORIZONTAL_PITCH_UM,
            'contact_radius_um': CONTACT_RADIUS_UM,
            'contact_channel_names': list(CONTACT_CHANNEL_NAMES),
        })
    return meta


def load_meta_neural(processed_server, session):
    """Read a session's meta_neural.json (required; created by create_meta_neural).

    Raises ValueError if the file is missing so the neural steps refuse to run
    before create_meta_neural has produced it.
    """
    path = meta_neural_path(processed_server, session)
    if not os.path.exists(path):
        raise ValueError(
            'meta_neural.json not found for session {} at {}. Run create_meta_neural '
            'first.'.format(session, path))
    return io.load_json(path)


class NeuralConfig():
    """Resolved configuration for processing a single session's neural data.

    Arguments:
        server {str} --- Folder where the raw sessions are located.
        processed_server {str} --- Folder where the processed data is located.
        session {str} --- Session directory name.
        probe_type {str} --- 'neuropixels' or 'vprobe'.
        nwb_units {str} --- Units written to the NWB: 'noise_excluded' (default;
            drops Phy 'noise' clusters when a phy export exists, else all units),
            'curated' (quality-triaged), or 'all' (raw sorter output).
        sorter {str} --- SpikeInterface sorter name.
        n_jobs {int} --- Number of parallel jobs for SpikeInterface.
        stream_id {str} --- Optional continuous stream id override.
        stream_name {str} --- Optional continuous stream name override.
        ttl_event_channel {str} --- Optional TTL event channel override.
        ttl_event_segment {int} --- Optional TTL event segment override.  Leave
            None to require the TTL events to come from the sorted segment
            (recording_index); set explicitly only when the TTLs legitimately
            live in a different segment than the one being sorted.
        recording {int|str} --- Which Open Ephys recording within experiment1
            (block 0) to process, 1-based (Recording1/Recording2/...; accepts 2 or
            'Recording2').  None -> meta_neural.json 'recording', then the probe
            default.
        meta_neural {dict} --- Pre-loaded meta_neural dict; None loads
            meta_neural.json from the processed session folder (REQUIRED -- run
            create_meta_neural first).  Every config field (block_index,
            common_reference_*, curation_query, the PROBE_DEFAULTS params and the
            V-probe geometry) is sourced from it, falling back to module defaults;
            nwb_units/sorter/recording additionally honour the CLI kwarg when given.
    """
    def __init__(self, server, processed_server, session, probe_type,
                 nwb_units=None, sorter=None, n_jobs=N_JOBS,
                 stream_id=None, stream_name=None, ttl_event_channel=None,
                 ttl_event_segment=None, recording=None, meta_neural=None):
        if probe_type not in PROBE_DEFAULTS:
            raise ValueError('Unknown probe_type {!r} (expected {}).'.format(
                probe_type, list(PROBE_DEFAULTS)))

        # Per-session neural config (required; created by create_meta_neural).
        meta = (meta_neural if meta_neural is not None
                else load_meta_neural(processed_server, session))
        self.meta_neural = meta
        d = PROBE_DEFAULTS[probe_type]

        self.server = server
        self.processed_server = processed_server
        self.session = session
        self.probe_type = probe_type
        self.n_jobs = n_jobs
        self.job_kwargs = dict(n_jobs=n_jobs, chunk_duration='1s', progress_bar=True)
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.ttl_event_channel = ttl_event_channel
        self.ttl_event_segment = ttl_event_segment

        # CLI-backed args: CLI kwarg > meta_neural > default.
        self.nwb_units = resolve_meta_arg(nwb_units, meta, 'nwb_units', 'noise_excluded')
        self.sorter_name = resolve_meta_arg(sorter, meta, 'sorter', SORTER_NAME)
        _rec = openephys.parse_recording_index(
            resolve_meta_arg(recording, meta, 'recording', None))
        self.recording_index = d['recording_index'] if _rec is None else _rec

        # Shared processing config: meta_neural > module default.
        self.block_index = meta.get('block_index', BLOCK_INDEX)
        self.common_reference_operator = meta.get('common_reference_operator',
                                                  COMMON_REFERENCE_OPERATOR)
        self.common_reference_type = meta.get('common_reference_type',
                                              COMMON_REFERENCE_TYPE)
        self.curation_query = meta.get('curation_query', CURATION_QUERY)

        # probe-scoped parameters: meta_neural > PROBE_DEFAULTS.
        self.highpass_freq_min = meta.get('highpass_freq_min', d['highpass_freq_min'])
        self.use_phase_shift = meta.get('use_phase_shift', d['use_phase_shift'])
        self.sparse = meta.get('sparse', d['sparse'])
        self.expected_n_channels = meta.get('expected_n_channels', d['expected_n_channels'])

        # v-probe geometry / wiring: meta_neural > module default.
        self.geometry = meta.get('geometry', GEOMETRY)
        self.contact_pitch_um = meta.get('contact_pitch_um', CONTACT_PITCH_UM)
        self.horizontal_pitch_um = meta.get('horizontal_pitch_um', HORIZONTAL_PITCH_UM)
        self.contact_radius_um = meta.get('contact_radius_um', CONTACT_RADIUS_UM)
        self.contact_channel_names = meta.get('contact_channel_names', CONTACT_CHANNEL_NAMES)

        # paths
        self.rserv = os.path.join(server, session)
        self.pserv = os.path.join(processed_server, session)
        self.raw_neural_dir = os.path.join(self.rserv, 'neural')
        self.oe_folder = openephys.find_oe_folder(self.raw_neural_dir)
        self.work_folder = os.path.join(self.pserv, 'neural_processed')
        self.nwb_path = os.path.join(self.work_folder, 'neural.nwb')

        # Optional merge of several recordings into one dataset (see the
        # 'merge_recordings' field in default_meta_neural).  Resolve each entry to
        # a concrete source {oe_folder, block_index, recording_index}; an empty or
        # single-entry list keeps the single-recording behaviour above.  When
        # merging, the primary attributes (oe_folder/block_index/recording_index)
        # are taken from source 0 so session_start_time / electrodes reflect the
        # earliest source.
        self.merge_recordings = meta.get('merge_recordings', []) or []
        if self.merge_recordings:
            self.merge_sources = [self._resolve_source_entry(e)
                                  for e in self.merge_recordings]
        else:
            self.merge_sources = [dict(oe_folder=self.oe_folder,
                                       block_index=self.block_index,
                                       recording_index=self.recording_index)]
        self.is_merged = len(self.merge_sources) > 1
        primary = self.merge_sources[0]
        self.oe_folder = primary['oe_folder']
        self.block_index = primary['block_index']
        self.recording_index = primary['recording_index']

    def _resolve_source_entry(self, entry):
        """Resolve one merge_recordings entry to {oe_folder, block_index, recording_index}.

        Unset keys fall back to this config's primary values: 'folder' -> the
        auto-resolved oe_folder, 'experiment' -> block_index, 'recording' -> the
        resolved recording_index.  'experiment'/'recording' are 1-based Open Ephys
        numbers converted to 0-based block/segment indices.
        """
        folder = entry.get('folder')
        oe = (openephys.find_oe_folder(os.path.join(self.raw_neural_dir, folder))
              if folder else self.oe_folder)
        experiment = entry.get('experiment')
        block = (self.block_index if experiment in (None, '')
                 else int(experiment) - 1)
        rec = entry.get('recording')
        rec_index = (self.recording_index if rec in (None, '')
                     else openephys.parse_recording_index(rec))
        return dict(oe_folder=oe, block_index=block, recording_index=rec_index)

    def for_source(self, source):
        """A shallow copy pinned to one merge source (oe_folder/block/segment).

        Lets the existing single-recording helpers (get_streams, resolve_stream,
        load_events, the loaders, the TTL readers) run unchanged against one source
        of a merge.  Only the three location scalars are overridden; meta_neural and
        every other resolved field are shared.
        """
        scfg = copy.copy(self)
        scfg.oe_folder = source['oe_folder']
        scfg.block_index = source['block_index']
        scfg.recording_index = source['recording_index']
        return scfg

    def subfolders(self):
        return {
            'preprocessed': os.path.join(self.work_folder, 'preprocessed'),
            'sorter': os.path.join(self.work_folder, self.sorter_name),
            'analyzer': os.path.join(self.work_folder, 'analyzer'),
            'analyzer_curated': os.path.join(self.work_folder, 'analyzer_curated'),
            'report': os.path.join(self.work_folder, 'reports', 'curated_report'),
            'phy': os.path.join(self.work_folder, 'phy'),
        }

    def ensure_work_folder(self):
        os.makedirs(self.work_folder, exist_ok=True)
        return self.work_folder

    def session_start_time(self):
        """Timezone-aware session start parsed from the OE timestamp folder name."""
        return openephys.session_start_time(self.oe_folder)


# ---------------------------------------------------------------------------
# Loaders for saved intermediates
# ---------------------------------------------------------------------------
def load_preprocessed(cfg):
    folder = cfg.subfolders()['preprocessed']
    if not os.path.exists(folder):
        raise FileNotFoundError('Preprocessed recording not found at {}. '
                                'Run preprocess_recording first.'.format(folder))
    return streams.load_si_folder(folder)


def load_sorting(cfg):
    import spikeinterface.full as si

    folder = cfg.subfolders()['sorter']
    if not os.path.exists(folder):
        raise FileNotFoundError('Sorter output not found at {}. '
                                'Run run_spike_sorting first.'.format(folder))
    return si.read_sorter_folder(str(folder))


def load_analyzer(cfg):
    import spikeinterface.full as si

    folder = cfg.subfolders()['analyzer']
    if not os.path.exists(folder):
        raise FileNotFoundError('Sorting analyzer not found at {}. '
                                'Run build_sorting_analyzer first.'.format(folder))
    return si.load_sorting_analyzer(str(folder))
