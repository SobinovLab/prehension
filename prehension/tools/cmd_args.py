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
import re
import argparse


def sessions_name_stub(sessions, sessions2=None):
    '''Filesystem-safe filename stub from the raw --sessions (+ --sessions2) tokens.

    Joins the tokens supplied on the command line with '_' and replaces any
    filename-unsafe characters (path separators, the ':' of region:/burr_hole:
    selectors, whitespace) with '-'.  An empty token list -> 'all_sessions'.  When
    ``sessions2`` is a non-empty list, its stub is appended as '<stub>__vs__<stub2>'
    so a figure comparing two session sets is named after both.  Used to name pooled
    figures after the session strings that produced them.
    '''
    def _stub(tokens):
        if not tokens:
            return 'all_sessions'
        joined = '_'.join(str(t) for t in tokens)
        return re.sub(r'[^A-Za-z0-9._-]+', '-', joined).strip('-_') or 'all_sessions'

    stub = _stub(sessions)
    if sessions2:
        stub = '{}__vs__{}'.format(stub, _stub(sessions2))
    return stub


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


class _ResolveSessionsAction(argparse.Action):
    '''argparse action that expands session selectors at parse time.

    Applied to --sessions / --sessions2 so every script gets flexible selection
    (sel:/region:/burr_hole:) for free.  The token list is passed through
    resolve_sessions using the active preset's processed_server and session_selections
    (set by preset.get_preset before the parser runs), so sessions without a
    meta_neural.json are simply skipped by the region:/burr_hole: selectors rather than
    erroring.  A bad selector (e.g. an unknown sel:name) is reported cleanly via
    parser.error.  processed_server is taken from the preset, not a --processed_server
    override, which only matters for the rare region:/burr_hole: + override combination.
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        from .. import preset
        try:
            resolved = resolve_sessions(values, preset.active_processed_server(),
                                        session_selections=preset.session_selections())
        except ValueError as e:
            parser.error(str(e))
        setattr(namespace, self.dest, resolved)


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
            action=_ResolveSessionsAction,
            help='List of session directories to process. If empty, find all unprocessed '
            'directories (empty by default). A token may also be a selector that expands '
            'via resolve_sessions(): "sel:<name>" -> the named list in the preset\'s '
            '"session_selections"; "region:<value>" or "burr_hole:<value>" '
            '(case-insensitive) -> every session whose meta_neural.json matches (sessions '
            'without a meta_neural.json are skipped).')

    if 'sessions2' in arguments:
        parser.add_argument(
            '--sessions2',
            type=str, default=[], nargs='*', metavar='SESSION',
            action=_ResolveSessionsAction,
            help='A second, independent list of session directories, processed separately '
            'from --sessions (e.g. pooled on its own and overlaid on the same plot for '
            'comparison). Same token syntax as --sessions (literal names or '
            '"sel:<name>"/"region:<value>"/"burr_hole:<value>" selectors expanded via '
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

    if 'drift_correct' in arguments:
        parser.add_argument(
            '--no_drift_correction', dest='drift_correct', action='store_false',
            help='Do not subtract each unit\'s linear session drift (rate trend across '
                 'the session) before analysis/plotting. Drift correction is ON by '
                 'default.')


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


def resolve_sessions(sessions, processed_server, session_selections=None):
    '''Expand 'sel:'/'region:'/'burr_hole:' selectors in a --sessions list.

    Each token in ``sessions`` is either a literal session directory name or a
    selector:
      * 'sel:<name>' expands to the named list under the preset's 'session_selections'
        (in the order given in the preset), e.g. 'sel:good_neuropixel'.  A 'sel:' that
        is undefined or maps to an empty list raises (rather than silently expanding to
        nothing / falling back to "all sessions").
      * 'region:<value>' / 'burr_hole:<value>' expands to every session whose
        meta_neural.json field equals <value> (capitalization-invariant).
    Literal names and all selector expansions are merged, order-preserving and
    de-duplicated; multiple selectors are OR'd (union).  An empty list is returned
    unchanged (downstream treats empty as "all sessions").

    ``session_selections`` is the {name: [session, ...]} map backing 'sel:'; None
    reads it from the active preset (preset.session_selections()).

    Needed modules are imported lazily so importing cmd_args stays cheap and free of
    heavy / circular dependencies.
    '''
    if not sessions:
        return sessions

    sel_names, meta_selectors, literals = [], [], []
    for tok in sessions:
        if ':' in tok:
            key, _, val = tok.partition(':')
            key = key.strip().lower()
            if key == 'sel':
                sel_names.append(val.strip())
                continue
            mapped = _SESSION_SELECTOR_KEYS.get(key)
            if mapped is not None:
                meta_selectors.append((mapped, val.strip()))
                continue
        literals.append(tok)

    if not sel_names and not meta_selectors:
        return sessions

    # lazy imports (avoid heavy / circular deps at module import time)
    from .logs import rs, ws

    # 'sel:' named lists from the preset (order preserved as written there).
    if session_selections is None:
        from .. import preset
        session_selections = preset.session_selections()
    named = []
    for nm in sel_names:
        lst = session_selections.get(nm)
        if not lst:  # undefined (None) or empty list -> the selector yields nothing
            raise ValueError(
                "Session selection 'sel:{}' is undefined or empty; refusing to run on "
                "any sessions (a failed 'sel:' must not fall back to all sessions). "
                "Defined selections in the preset's 'session_selections': {}.".format(
                    nm, sorted(session_selections)))
        named.extend(lst)

    # 'region:'/'burr_hole:' selectors matched against each session's meta_neural.json.
    matched = []
    if meta_selectors:
        from .. import meta_session
        from ..neural_processing import config as npconfig
        for session in meta_session.find_session_dirs(processed_server):
            try:
                meta = npconfig.load_meta_neural(processed_server, session)
            except Exception:  # noqa: BLE001 - missing/unreadable neural config -> skip
                continue  # session has no usable meta_neural.json; not added to the list
            for field, val in meta_selectors:
                mv = meta.get(field, None)
                if mv is not None and str(mv).strip().lower() == val.lower():
                    matched.append(session)
                    break
        if not matched:
            ws('No sessions with a meta_neural.json matched {}.'.format(
                ['{}:{}'.format(k, v) for k, v in meta_selectors]))

    label = (['sel:{}'.format(n) for n in sel_names] +
             ['{}:{}'.format(k, v) for k, v in meta_selectors])
    out, seen = [], set()
    for s in literals + named + matched:
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
