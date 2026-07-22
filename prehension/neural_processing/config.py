#!python3
# -*- coding: utf-8 -*-
"""
Configuration and shared helpers for neural spike sorting (Open Ephys ->
minimal NWB), for both Neuropixels and 32-channel V-probe recordings.

Per-session paths are resolved from a server/processed_server/session triple (as
elsewhere in prehension): raw neural = server/session/neural, processed output =
processed_server/session/neural_processed.  The preset and session are supplied
on the command line by the calling scripts.

Only two processing steps differ between probes (recording load and
preprocessing).  Probe-scoped parameters live in PROBE_DEFAULTS below; the larger
per-dataset V-probe wiring lives in this local config file.  A NeuralConfig
instance carries the resolved values for one session and is passed to the step
functions.

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
import json
import datetime


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
        'recording_index': 1,       # recording2
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

# Mapping from the meta_structure 'neural' field (written by fill_meta_structure) to a probe_type
# key understood by PROBE_DEFAULTS / NeuralConfig.
NEURAL_TO_PROBE_TYPE = {
    'vprobe': 'vprobe',
    'neuropixel': 'neuropixels',
    'neuropixels': 'neuropixels',
}


def probe_type_from_meta(server, processed_server, session):
    """Resolve a session's probe_type from its meta_structure.json 'neural' field.

    The field is written by meta_session.fill_meta_structure during create_meta and is one of
    '' (no neural data), 'vprobe', or 'neuropixel'. Returns a probe_type key of PROBE_DEFAULTS
    ('vprobe' or 'neuropixels'). Raises ValueError if meta is missing or has no neural type.
    """
    meta_path = os.path.join(processed_server, session, 'meta_structure.json')
    if not os.path.exists(meta_path):
        raise ValueError(
            'meta_structure.json not found for session {} at {}. Run create_meta first.'.format(
                session, meta_path))
    with open(meta_path, 'r') as f:
        mstruct = json.load(f)

    neural = (mstruct.get('neural') or '').strip()
    if neural == '':
        raise ValueError(
            'No neural recording type recorded in meta_structure for session {} (the "neural" '
            'field is empty). Re-run create_meta on a session that has a neural/ folder.'.format(
                session))
    if neural not in NEURAL_TO_PROBE_TYPE:
        raise ValueError(
            'Unknown neural recording type {!r} in meta_structure for session {}. Expected one '
            'of {}.'.format(neural, session, list(NEURAL_TO_PROBE_TYPE)))
    return NEURAL_TO_PROBE_TYPE[neural]


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


# ---------------------------------------------------------------------------
# Per-session configuration
# ---------------------------------------------------------------------------
def find_oe_folder(neural_dir):
    """Locate the Open Ephys folder (containing 'Record Node ###') in a neural dir.

    Handles neural/ directly containing Record Node dirs, or
    neural/<timestamped session>/Record Node ###.
    """
    try:
        entries = [os.path.join(neural_dir, e) for e in os.listdir(neural_dir)]
        entries = [e for e in entries if os.path.isdir(e)]
    except (FileNotFoundError, NotADirectoryError):
        return neural_dir

    if any(os.path.basename(e).startswith('Record Node') for e in entries):
        return neural_dir

    def has_record_node(d):
        try:
            return any(os.path.isdir(os.path.join(d, c)) and c.startswith('Record Node')
                       for c in os.listdir(d))
        except OSError:
            return False

    candidates = sorted((e for e in entries if has_record_node(e)),
                        key=os.path.basename)
    return candidates[-1] if candidates else neural_dir


def parse_recording_index(recording):
    """Convert a user recording spec to a 0-based segment index.

    Open Ephys names the recordings within an experiment Recording1, Recording2,
    ... (1-based).  Accepts that number as an int or string ('2', 'Recording2')
    and returns the 0-based SpikeInterface segment index (Recording2 -> 1).
    Returns None when recording is None so the caller keeps the probe default.
    """
    if recording is None:
        return None
    s = str(recording).strip().lower()
    if s.startswith('recording'):
        s = s[len('recording'):].strip()
    try:
        number = int(s)
    except (TypeError, ValueError):
        raise ValueError(
            "Could not parse recording {!r}; use e.g. 1, 2 or 'Recording2'.".format(recording))
    if number < 1:
        raise ValueError(
            'recording must be >= 1 (Open Ephys Recording1 is the first), got {}.'.format(number))
    return number - 1


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
            (block 0) to process, 1-based to match the Recording1/Recording2/...
            folder names (accepts 2 or 'Recording2').  None -> the probe default
            (vprobe Recording1, neuropixels Recording2).
    """
    def __init__(self, server, processed_server, session, probe_type,
                 nwb_units='noise_excluded', sorter=SORTER_NAME, n_jobs=N_JOBS,
                 stream_id=None, stream_name=None, ttl_event_channel=None,
                 ttl_event_segment=None, recording=None):
        if probe_type not in PROBE_DEFAULTS:
            raise ValueError('Unknown probe_type {!r} (expected {}).'.format(
                probe_type, list(PROBE_DEFAULTS)))

        self.server = server
        self.processed_server = processed_server
        self.session = session
        self.probe_type = probe_type
        self.nwb_units = nwb_units
        self.sorter_name = sorter
        self.n_jobs = n_jobs
        self.job_kwargs = dict(n_jobs=n_jobs, chunk_duration='1s', progress_bar=True)
        self.block_index = BLOCK_INDEX
        self.stream_id = stream_id
        self.stream_name = stream_name
        self.ttl_event_channel = ttl_event_channel
        self.ttl_event_segment = ttl_event_segment
        self.common_reference_operator = COMMON_REFERENCE_OPERATOR
        self.common_reference_type = COMMON_REFERENCE_TYPE
        self.curation_query = CURATION_QUERY

        # probe-scoped parameters
        d = PROBE_DEFAULTS[probe_type]
        _rec_override = parse_recording_index(recording)
        self.recording_index = (d['recording_index'] if _rec_override is None
                                else _rec_override)
        self.highpass_freq_min = d['highpass_freq_min']
        self.use_phase_shift = d['use_phase_shift']
        self.sparse = d['sparse']
        self.expected_n_channels = d['expected_n_channels']

        # v-probe geometry / wiring
        self.geometry = GEOMETRY
        self.contact_pitch_um = CONTACT_PITCH_UM
        self.horizontal_pitch_um = HORIZONTAL_PITCH_UM
        self.contact_radius_um = CONTACT_RADIUS_UM
        self.contact_channel_names = CONTACT_CHANNEL_NAMES

        # paths
        self.rserv = os.path.join(server, session)
        self.pserv = os.path.join(processed_server, session)
        self.raw_neural_dir = os.path.join(self.rserv, 'neural')
        self.oe_folder = find_oe_folder(self.raw_neural_dir)
        self.work_folder = os.path.join(self.pserv, 'neural_processed')
        self.nwb_path = os.path.join(self.work_folder, 'neural.nwb')

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
        name = os.path.basename(self.oe_folder)  # e.g. 2026-06-24_15-02-48
        local_tz = datetime.datetime.now().astimezone().tzinfo
        try:
            return datetime.datetime.strptime(
                name, '%Y-%m-%d_%H-%M-%S').replace(tzinfo=local_tz)
        except ValueError:
            return datetime.datetime(1970, 1, 1, tzinfo=local_tz)


# ---------------------------------------------------------------------------
# Stream resolution
# ---------------------------------------------------------------------------
def get_streams(cfg):
    """Return (stream_names, stream_ids) for the Open Ephys binary folder."""
    import spikeinterface.full as si

    stream_names, stream_ids = si.get_neo_streams('openephysbinary', str(cfg.oe_folder))
    return list(stream_names), list(stream_ids)


def _is_ap_stream(name):
    n = name.lower()
    return n.endswith('ap') or '-ap' in n or '.ap' in n


def resolve_stream(cfg, stream_names, stream_ids):
    """Pick the continuous neural stream.

    Neuropixels -> first AP-named stream; V-probe -> the stream whose channel
    count equals expected_n_channels.  Priority: stream_id -> stream_name ->
    probe rule -> first stream.  Returns (stream_id, stream_name).
    """
    import spikeinterface.full as si

    pairs = list(zip(stream_names, stream_ids))

    if cfg.stream_id is not None:
        for name, sid in pairs:
            if str(sid) == str(cfg.stream_id):
                return sid, name
        raise ValueError('stream_id={!r} not found in {}'.format(cfg.stream_id, stream_ids))
    if cfg.stream_name is not None:
        for name, sid in pairs:
            if name == cfg.stream_name:
                return sid, name
        raise ValueError('stream_name={!r} not found in {}'.format(cfg.stream_name, stream_names))

    if cfg.probe_type == 'neuropixels':
        ap = [(name, sid) for name, sid in pairs if _is_ap_stream(name)]
        if ap:
            name, sid = ap[0]
            return sid, name
    else:  # vprobe: match channel count
        for name, sid in pairs:
            try:
                r = si.read_openephys(str(cfg.oe_folder), stream_id=str(sid),
                                      block_index=cfg.block_index)
                if r.get_num_channels() == cfg.expected_n_channels:
                    return sid, name
            except Exception:
                continue

    name, sid = pairs[0]
    return sid, name


def select_one_segment(rec, segment_index):
    """Reduce a multi-segment recording to a single mono-segment recording."""
    import spikeinterface.full as si

    if hasattr(rec, 'select_segments'):
        return rec.select_segments([segment_index])
    return si.select_segment_recording(rec, segment_indices=segment_index)


# ---------------------------------------------------------------------------
# Probe geometry / wiring (V-probe)
# ---------------------------------------------------------------------------
def _contact_positions(cfg):
    import numpy as np

    n = cfg.expected_n_channels
    pos = np.zeros((n, 2), dtype=float)
    if cfg.geometry == 'linear':
        for i in range(n):
            pos[i] = [0.0, i * cfg.contact_pitch_um]
    elif cfg.geometry == 'staggered':
        for i in range(n):
            row, col = divmod(i, 2)
            pos[i] = [col * cfg.horizontal_pitch_um, row * cfg.contact_pitch_um]
    else:
        raise ValueError('Unknown geometry {!r}'.format(cfg.geometry))
    return pos


def recording_channel_names(rec):
    """Channel names in the recording's stored channel order."""
    import numpy as np

    names = rec.get_property('channel_name')
    if names is None:
        names = np.array([str(c) for c in rec.get_channel_ids()])
    return [str(n) for n in names]


def build_vprobe(cfg, rec):
    """Build a 32-channel probe and wire contacts to device channels by name."""
    import numpy as np
    from probeinterface import Probe

    n = cfg.expected_n_channels
    if len(cfg.contact_channel_names) != n:
        raise ValueError('contact_channel_names has {} entries, expected {}.'.format(
            len(cfg.contact_channel_names), n))

    names = recording_channel_names(rec)
    name_to_index = {nm: i for i, nm in enumerate(names)}
    missing = [c for c in cfg.contact_channel_names if c not in name_to_index]
    if missing:
        raise ValueError('Contact channel names not in recording: {}. '
                         'Recording channel names (stored order): {}'.format(missing, names))

    probe = Probe(ndim=2, si_units='um')
    probe.set_contacts(positions=_contact_positions(cfg), shapes='circle',
                       shape_params={'radius': cfg.contact_radius_um})
    probe.create_auto_shape(probe_type='tip')
    device_indices = np.array([name_to_index[c] for c in cfg.contact_channel_names],
                              dtype='int64')
    probe.set_device_channel_indices(device_indices)
    return probe


def attach_probe(cfg, rec):
    """Attach the manually-built V-probe geometry to a recording."""
    return rec.set_probe(build_vprobe(cfg, rec))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------
def load_events(cfg):
    import spikeinterface.full as si

    return si.read_openephys_event(str(cfg.oe_folder), block_index=cfg.block_index)


def _ttl_name_score(channel_id):
    s = str(channel_id).lower()
    if 'message' in s:
        return -1
    score = 0
    if 'ttl' in s:
        score += 3
    if 'sync' in s:
        score += 2
    if 'ap' in s:
        score += 1
    return score


def event_segment_counts(events, channel_id):
    """Return {segment_index: n_events} for a channel across all event segments."""
    try:
        n_seg = events.get_num_segments()
    except Exception:
        n_seg = 1
    counts = {}
    for s in range(n_seg):
        try:
            ev = events.get_events(channel_id=channel_id, segment_index=s)
            counts[s] = int(len(ev)) if ev is not None else 0
        except Exception:
            counts[s] = None
    return counts


def resolve_ttl_channel(cfg, events):
    """Pick the TTL/sync channel that actually has events (avoids 'Messages')."""
    channel_ids = list(events.channel_ids)
    if cfg.ttl_event_channel is not None:
        if cfg.ttl_event_channel not in channel_ids:
            raise ValueError('ttl_event_channel={!r} not in {}'.format(
                cfg.ttl_event_channel, channel_ids))
        return cfg.ttl_event_channel

    seen, unique_ids = set(), []
    for cid in channel_ids:
        if str(cid) not in seen:
            seen.add(str(cid))
            unique_ids.append(cid)

    def n_events(cid):
        counts = [c for c in event_segment_counts(events, cid).values() if c]
        return max(counts) if counts else 0

    with_events = [cid for cid in unique_ids if n_events(cid) > 0]
    if with_events:
        with_events.sort(key=lambda cid: (_ttl_name_score(cid), n_events(cid)),
                         reverse=True)
        return with_events[0]
    hinted = [cid for cid in unique_ids if _ttl_name_score(cid) > 0]
    return hinted[0] if hinted else channel_ids[0]


def find_event_segment(cfg, events, channel_id, preferred=None):
    """Return (segment_index, ev) for the first segment with non-empty events."""
    if preferred is None:
        preferred = cfg.recording_index
    try:
        n_seg = events.get_num_segments()
    except Exception:
        n_seg = preferred + 1
    order = [preferred] + [s for s in range(n_seg) if s != preferred]

    fallback = None
    for s in order:
        try:
            ev = events.get_events(channel_id=channel_id, segment_index=s)
        except Exception:
            continue
        if fallback is None:
            fallback = (s, ev)
        if ev is not None and len(ev) > 0:
            return s, ev
    return fallback if fallback is not None else (preferred, None)


# ---------------------------------------------------------------------------
# Loaders for saved intermediates
# ---------------------------------------------------------------------------
def load_si_folder(folder):
    import spikeinterface.full as si

    if hasattr(si, 'load'):
        try:
            return si.load(str(folder))
        except Exception:
            pass
    return si.load_extractor(str(folder))


def load_preprocessed(cfg):
    folder = cfg.subfolders()['preprocessed']
    if not os.path.exists(folder):
        raise FileNotFoundError('Preprocessed recording not found at {}. '
                                'Run preprocess_recording first.'.format(folder))
    return load_si_folder(folder)


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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=str)
