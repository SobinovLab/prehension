#!python3
# -*- coding: utf-8 -*-
"""
Open Ephys folder / recording-layout helpers (reusable across neural work).

Locate the Open Ephys folder in a session's neural/ directory, enumerate the
recordings it contains, map a 1-based Open Ephys recording spec to a 0-based
segment index, fetch an Open Ephys Recording object, and parse the session start
time from the timestamped folder name.  None of these need a NeuralConfig class;
they take primitive paths / indices or a duck-typed source config.

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
import glob
import datetime

from ...tools.misc import trailing_int


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


def describe_recording(entry):
    """One-line human description of a merge_recordings entry."""
    return 'folder={} experiment={} recording={}'.format(
        entry.get('folder', '<default>'),
        entry.get('experiment', 1), entry.get('recording', 1))


def enumerate_recordings(raw_neural_dir):
    """List every Open Ephys recording under a session's neural/ folder.

    Walks neural/[<timestamp>/]Record Node */experiment*/recording*/continuous
    (the same layout meta_session uses to detect neural data) and returns one
    merge_recordings entry per recording: {folder, experiment, recording}.
    'folder' is the path (relative to neural/) of the Open Ephys folder holding the
    Record Node, omitted when the Record Node sits directly in neural/ so the
    default oe_folder resolution applies.  'experiment'/'recording' are the 1-based
    experimentN/recordingM numbers.  Ordered by (folder, experiment, recording) --
    chronologically for timestamp-named folders.
    """
    found = {}
    # timestamped layout first, then Record Node directly under neural/; the two
    # patterns cannot match the same recording (they differ in nesting depth).
    for pattern in ('*/Record Node */experiment*/recording*/continuous',
                    'Record Node */experiment*/recording*/continuous'):
        for cont in glob.glob(os.path.join(raw_neural_dir, pattern)):
            rec_dir = os.path.dirname(cont)          # .../recordingM
            exp_dir = os.path.dirname(rec_dir)       # .../experimentN
            rn_dir = os.path.dirname(exp_dir)        # .../Record Node ###
            parent = os.path.dirname(rn_dir)         # neural/ or neural/<timestamp>
            experiment = trailing_int(os.path.basename(exp_dir))
            recording = trailing_int(os.path.basename(rec_dir))
            if experiment is None or recording is None:
                continue
            rel = os.path.relpath(parent, raw_neural_dir)
            folder = None if rel == '.' else rel
            entry = {'experiment': experiment, 'recording': recording}
            if folder is not None:
                entry['folder'] = folder
            found[(folder or '', experiment, recording)] = entry

    return [found[k] for k in sorted(found)]


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


def oe_recording(session, block_index, recording_index):
    """Open Ephys Recording for (experiment=block_index, recording=recording_index).

    Prefers matching the recording's own experiment/recording indices when the
    installed open-ephys-python exposes them; otherwise falls back to flat indexing
    of recordnodes[0].recordings, which reproduces the previous behaviour for a
    single experiment (block_index 0).
    """
    recs = session.recordnodes[0].recordings
    matches = [r for r in recs
               if getattr(r, 'experiment_index', None) == block_index
               and getattr(r, 'recording_index', None) == recording_index]
    if matches:
        return matches[0]
    return recs[recording_index]


def session_start_time(oe_folder):
    """Timezone-aware session start parsed from the OE timestamp folder name.

    ``oe_folder`` basename is expected like '2026-06-24_15-02-48'; unparseable
    names fall back to the Unix epoch (still tz-aware).
    """
    name = os.path.basename(oe_folder)  # e.g. 2026-06-24_15-02-48
    local_tz = datetime.datetime.now().astimezone().tzinfo
    try:
        return datetime.datetime.strptime(
            name, '%Y-%m-%d_%H-%M-%S').replace(tzinfo=local_tz)
    except ValueError:
        return datetime.datetime(1970, 1, 1, tzinfo=local_tz)
