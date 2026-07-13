#!python3
# -*- coding: utf-8 -*-
"""
Reports which sessions have duplicate trial IDs in their behavioural (auto) log.

Read-only utility: it does not create, overwrite, or modify any files. Use it to find the sessions
affected by duplicate-trial-ID recordings (whose camera data was stored as ``trial<N>_1`` and
pressure data as ``<serial>_<N>_1``) before regenerating their meta information with create_meta.

Usage:
    python -m prehension_scripts.check_duplicate_trials <preset> [--sessions S1 S2 ...]
                                                         [--server DIR]

Copyright (C) 2019-2024 Anton Sobinov
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
import argparse
import os

from prehension import preset
from prehension import meta_session
from prehension.tools import cmd_args
from prehension.create_meta import find_duplicate_trials


def check_session_duplicates(raw_ss, session):
    """Return ``(duplicates, n_total)`` for a session's behavioural log.

    Builds an in-memory meta structure (no meta_structure.json is written) just far enough to
    locate the auto log, then delegates to create_meta.find_duplicate_trials. Read-only.
    """
    mstruct = meta_session.get_default_meta_structure()
    meta_session.fill_meta_structure(mstruct, raw_ss, session)
    return find_duplicate_trials(mstruct)


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description='Reports which sessions have duplicate trial IDs in the behavioural log. '
                    'Read-only: it creates and modifies nothing.')
    cmd_args.add_default_kwarguments(parser, {'server': current_preset['default_server']})
    cmd_args.add_default_arguments(parser, ('sessions',))
    args = parser.parse_args(args=argv)

    server = args.server
    if len(args.sessions) > 0:
        sessions = [s for s in args.sessions if os.path.exists(os.path.join(server, s))]
    else:
        sessions = meta_session.find_session_dirs(server)

    print('Checking {} session(s) on {} for duplicate behavioural-log trial IDs.'.format(
        len(sessions), server))

    n_affected = 0
    for session in sessions:
        raw_ss = os.path.join(server, session)
        try:
            duplicates, n_total = check_session_duplicates(raw_ss, session)
        except Exception as e:
            print('  {}: could not check ({})'.format(session, repr(e)))
            continue

        if duplicates:
            n_affected += 1
            # show which recordings each duplicated ID maps to (later occurrences get '_1', ...)
            detail = '; '.join(
                'trial {} x{} -> {}'.format(
                    t, c,
                    ', '.join([str(t)] + ['{}_{}'.format(t, i) for i in range(1, c)]))
                for t, c in sorted(duplicates.items()))
            print('  {}: DUPLICATES ({} log rows): {}'.format(session, n_total, detail))
        else:
            print('  {}: no duplicates ({} log rows)'.format(session, n_total))

    print('Done. {} of {} session(s) have duplicate trial IDs.'.format(n_affected, len(sessions)))
