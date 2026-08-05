#!python3
# -*- coding: utf-8 -*-
"""
Functions for parsing command line arguments.

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
import os


def resolve_meta_arg(cli_value, meta, key, default=None):
    '''CLI kwarg if provided (not None), else the meta value, else default.

    A generic precedence resolver used to layer a command-line override over a
    per-session config dict: returns ``cli_value`` when it is not None, otherwise
    ``meta[key]`` when present and not empty, otherwise ``default``.
    '''
    if cli_value is not None:
        return cli_value
    v = meta.get(key, None) if meta else None
    return default if v is None or v == '' else v


def add_default_arguments(parser, arguments):
    '''Possible arguments:
        sessions
        sessions2 -- second, independent session list
        session -- singular
        trials
        trial -- singular required argument
        temp
        processes
        overwrite
        dry_run
    '''
    if isinstance(arguments, str):
        arguments = (arguments, )
    if 'sessions' in arguments:
        parser.add_argument(
            '--sessions',
            type=str, default=[], nargs='*', metavar='SESSION',
            help='List of session directories to process. If empty, find all unprocessed '
            'directories (empty by default). A token may also be a selector '
            '"region:<value>" or "burr_hole:<value>" (case-insensitive) that expands, via '
            'resolve_sessions(), to every session whose meta_neural.json matches.')

    if 'sessions2' in arguments:
        parser.add_argument(
            '--sessions2',
            type=str, default=[], nargs='*', metavar='SESSION',
            help='A second, independent list of session directories, processed separately '
            'from --sessions (e.g. pooled on its own and overlaid on the same plot for '
            'comparison). Same token syntax as --sessions (literal names or '
            '"region:<value>"/"burr_hole:<value>" selectors expanded via '
            'resolve_sessions()). Empty by default (no second set).')

    if 'trials' in arguments:
        parser.add_argument(
            '--trials',
            type=int, default=[], nargs='*', metavar='TRIAL_NUMBER',
            help='List of trials for processing. If empty, find all unprocessed trials. '
            'Empty by default.')

    temp_folder = os.path.join('C:\\', 'tmp')
    if 'temp' in arguments:
        parser.add_argument(
            '--temp',
            type=str, default=temp_folder,
            help='Folder for local temporary storage. Default: {}'.format(temp_folder))

    processes = int(round(os.cpu_count()*1.4))
    if 'processes' in arguments:
        parser.add_argument(
            '--processes',
            type=int, default=processes,
            help='Number of parallel processes in the pool. Defaults to {}.'.format(processes))

    # affecting data
    if 'overwrite' in arguments:
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrites the created files if they exist.')

    if 'dry_run' in arguments:
        parser.add_argument(
            '--dry_run', '--check',
            action='store_true',
            help='Check which trials have not been converted. Does not create data.')

    if 'session' in arguments:
        parser.add_argument(
            'session',
            type=str,
            help='Session directory to use. Required argument.')

    if 'trial' in arguments:
        parser.add_argument(
            'trial',
            type=int,
            help='Trial to do adjustment on. Required argument.')

    if 'make_plots' in arguments:
        parser.add_argument(
            '--make_plots',
            action='store_true',
            help='Makes some inspection figures. Run with --processes 1.')

    if 'debug' in arguments:
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Run script in debug mode'
        )


def add_default_kwargument(parser, k, v):
    if k == 'server':
        parser.add_argument(
            '--server',
            type=str, default=v,
            help='Folder where the sessions are located. Default: {}'.format(v))
        return

    if k == 'processed_server':
        parser.add_argument(
            '--processed_server',
            type=str, default=v,
            help='Folder where the processed data from sessions are located. Default: {}'.format(v))
        return

    raise ValueError('Unknown keyword argument for parser: {}'.format(k))


def add_default_kwarguments(parser, kwarguments):
    '''Possible kwargs:
        server
    '''
    for k, v in kwarguments.items():
        add_default_kwargument(parser, k, v)


# Selector keys accepted in a --sessions token ('key:value'); the value is matched
# case-insensitively against the given field of the session's meta_neural.json.
_SESSION_SELECTOR_KEYS = {
    'region': 'region',
    'burr_hole': 'burr_hole',
    'burrhole': 'burr_hole',
}


def resolve_sessions(sessions, processed_server):
    '''Expand 'region:'/'burr_hole:' selectors in a --sessions list.

    Each token in ``sessions`` is either a literal session directory name or a
    selector 'region:<value>' or 'burr_hole:<value>'.  A selector expands to every
    session that has a meta_neural.json whose matching field equals <value>
    (capitalization-invariant).  Literal names and all selector matches are merged,
    order-preserving and de-duplicated; multiple selectors are OR'd (union).  An
    empty list is returned unchanged (downstream treats empty as "all sessions").

    Needed modules are imported lazily so importing cmd_args stays cheap and free of
    heavy / circular dependencies.
    '''
    if not sessions:
        return sessions

    selectors, literals = [], []
    for tok in sessions:
        if ':' in tok:
            key, _, val = tok.partition(':')
            mapped = _SESSION_SELECTOR_KEYS.get(key.strip().lower())
            if mapped is not None:
                selectors.append((mapped, val.strip()))
                continue
        literals.append(tok)

    if not selectors:
        return sessions

    # lazy imports (avoid heavy / circular deps at module import time)
    from .. import meta_session
    from ..neural_processing import config as npconfig
    from .logs import rs, ws

    matched = []
    for session in meta_session.find_session_dirs(processed_server):
        try:
            meta = npconfig.load_meta_neural(processed_server, session)
        except ValueError:
            continue  # no meta_neural.json for this session
        for field, val in selectors:
            mv = meta.get(field, None)
            if mv is not None and str(mv).strip().lower() == val.lower():
                matched.append(session)
                break

    label = ['{}:{}'.format(k, v) for k, v in selectors]
    if not matched:
        ws('No sessions with a meta_neural.json matched {}.'.format(label))

    out, seen = [], set()
    for s in literals + matched:
        if s not in seen:
            seen.add(s)
            out.append(s)
    if not out:
        # selectors were given but nothing matched: do NOT fall back to "all
        # sessions" (empty list), which would silently process everything.
        raise ValueError(
            'No sessions matched selectors {} (and none given literally); refusing '
            'to fall back to all sessions.'.format(label))
    rs('Resolved --sessions selectors {} -> {} session(s): {}'.format(
        label, len(out), ', '.join(out)))
    return out
