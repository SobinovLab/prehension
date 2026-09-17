#!python3
# -*- coding: utf-8 -*-
"""
Two-panel encoding summary across sessions: the per-unit optimal lags (top) and the
per-unit adjusted pseudo-R2 (bottom), grouped by predictor modality on a shared x-axis.

Reads what the encoding_lag and encoding scripts saved for the requested sessions --
encoding/<predictor>_lag[_tx].json (per-unit best_lag_s + session mode_lag_s) and
encoding/encoding_<predictor>[_tx].json (per-unit adj_pr2) -- pools the units across
sessions per predictor, and draws a jittered swarm for each: optimal lag (ms) on top with
the pooled mode marked, and the adjusted pseudo-R2 distribution beneath with its median
marked.  Predictors with either file present are shown; missing panels are simply empty.

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

import numpy as np
import matplotlib.pyplot as plt

from .. import meta_session
from ..tools import io, plotting
from ..tools.logs import rs, ws
from ..neural_plotting.common.traces import resolve_pooled_save_dir
from .encoding import (
    encoding_json_path, lag_json_path, resolve_period, ALL_PREDICTORS, MIN_ADJ_R2, _suffix,
    _period_suffix, _joint_group_suffix, DEFAULT_JOINT_GROUP)


LAG_EDGE_S = 1.0   # per-unit lags at +/- this (the sweep edges) are excluded from the median


def _central_lag(lags_s):
    """Median of the per-unit lags (s) with the +/-LAG_EDGE_S sweep edges removed (NaN if none).

    Optimal lags pinned at the +/- window edge are clipped, unreliable optima, so they are
    dropped before the central lag is computed.
    """
    a = np.asarray(lags_s, dtype=float)
    a = a[np.abs(a) < LAG_EDGE_S - 1e-9]
    return float(np.median(a)) if a.size else np.nan


def _read_summary(processed_server, sessions, predictors, use_threshold_crossings, period,
                  joint_group):
    """Pool per-unit best lags (s) and adjusted pR2 across sessions, per predictor.

    Each predictor is read for its effective period (resolve_period(predictor, period); None
    -> the predictor's default) and the requested `joint_group`.  Returns (lags, adj) dicts
    keyed by predictor: lags[p] the list of per-unit best_lag_s and adj[p] the list of per-unit
    adj_pr2, over the sessions that have the files.  When a predictor has one side (lag or
    encoding) for its period but not the other, a warning names the missing side / period and the
    script that generates it, so a half-empty panel is explained rather than silent.
    """
    lags, adj = {}, {}
    for predictor in predictors:
        eff = resolve_period(predictor, period)
        pl, pa, n_lag, n_enc = [], [], 0, 0
        for session in sessions:
            lp = lag_json_path(processed_server, session, predictor, use_threshold_crossings, eff,
                               joint_group)
            if os.path.exists(lp):
                n_lag += 1
                pl += [u['best_lag_s'] for u in io.load_json(lp).get('units', [])
                       if u.get('best_lag_s') is not None]
            ep = encoding_json_path(processed_server, session, predictor,
                                    use_threshold_crossings, eff, joint_group)
            if os.path.exists(ep):
                n_enc += 1
                pa += [u['adj_pr2'] for u in io.load_json(ep).get('units', [])
                       if u.get('adj_pr2') is not None]
        if pl:
            lags[predictor] = pl
        if pa:
            adj[predictor] = pa
        # a one-sided period (lag but no encoding, or the reverse) leaves a panel empty -- say why
        if n_lag and not n_enc:
            ws("{} [{}]: found lag file(s) but no encoding r2 file for this period; the r2 panel "
               "will be empty. Generate it with encoding.py for period '{}'.".format(
                   predictor, eff, eff))
        elif n_enc and not n_lag:
            ws("{} [{}]: found encoding r2 file(s) but no lag file for this period; the lag panel "
               "will be empty. Generate it with encoding_lag.py for period '{}'.".format(
                   predictor, eff, eff))
    return lags, adj


def encoding_summary(processed_server, sessions, predictors=None, min_adj_r2=MIN_ADJ_R2,
                     use_threshold_crossings=False, name=None, save=True, save_dir=None,
                     period=None, joint_group=DEFAULT_JOINT_GROUP):
    """Plot per-unit optimal lags (top) and adjusted pR2 (bottom) per predictor, across sessions.

    Pools the units saved for `sessions` (empty -> all sessions under processed_server) from
    the encoding/ folder and draws two stacked, x-aligned seaborn panels per predictor: a
    swarm of the optimal lag in ms (top, with the median of the non-edge lags -- excluding
    those at the +/-1 s sweep edges -- marked in red) and a matching swarm of the per-unit
    adjusted pseudo-R2 (bottom, with the median marked and a dashed cutoff line at
    `min_adj_r2`).  `min_adj_r2` only positions that reference line -- it does not
    re-filter the points; the lag panel already reflects whatever threshold produced the lag
    files.  `predictors` restricts the modalities shown (default: any with saved data).
    `period` selects which trial sub-period's files to read (see resolve_period; None -> each
    predictor's default, so columns may differ in period -- shown in the x tick labels).  Saved
    by default into <processed_server>/pooled_figures/encoding_summary.  Returns the figure.
    """
    import seaborn as sns

    found = sessions if sessions else meta_session.find_session_dirs(processed_server)
    preds = list(predictors) if predictors else list(ALL_PREDICTORS)

    lags, adj = _read_summary(processed_server, found, preds, use_threshold_crossings, period,
                              joint_group)
    shown = [p for p in preds if p in lags or p in adj]
    if not shown:
        want = sorted({resolve_period(p, period) for p in preds})
        raise ValueError(
            'No encoding lag / adjusted-R2 data found for sessions {} at period(s) {}. Generate '
            'them for these periods (encoding_lag.py / encoding.py --period ...).'.format(
                sessions, want))

    positions = np.arange(len(shown))
    save_dir = resolve_pooled_save_dir(processed_server, 'encoding_summary', save, save_dir)

    # long-form (category, value) vectors for seaborn (lags in ms); predictors with no data
    # still get a reserved column via order=shown, keeping the two panels x-aligned.
    lag_cat = [p for p in shown for _ in lags.get(p, [])]
    lag_val = [v * 1e3 for p in shown for v in lags.get(p, [])]
    adj_cat = [p for p in shown for _ in adj.get(p, [])]
    adj_val = [v for p in shown for v in adj.get(p, [])]

    fig, (ax_lag, ax_adj) = plt.subplots(
        2, 1, sharex=True, figsize=(max(8, 2.0 * len(shown)), 9))

    # top: swarm of per-unit optimal lags, with the median (of non-edge lags) marked in red
    if lag_val:
        sns.swarmplot(x=lag_cat, y=lag_val, order=shown, ax=ax_lag, color='k', size=3, alpha=0.7)
    for i, p in enumerate(shown):
        m = _central_lag(lags.get(p, []))
        if np.isfinite(m):
            ax_lag.plot([i - 0.4, i + 0.4], [m * 1e3] * 2, color='tab:red', lw=2.0)
            ax_lag.annotate('med {:+.0f} ms'.format(m * 1e3), (i, m * 1e3),
                            textcoords='offset points', xytext=(6, 2), fontsize=7,
                            color='tab:red')
    ax_lag.axhline(0.0, color='0.6', lw=0.8, ls=':')
    ax_lag.set_ylabel('Optimal lag, ms (>0: input follows firing)')
    ax_lag.set_title('Per-unit optimal lags and encoding performance across predictors '
                     '({} session(s))'.format(len(found)))

    # bottom: per-unit adjusted pseudo-R2 as a swarm with the median marked -- same style as
    # the top panel -- plus the dashed cutoff line at min_adj_r2
    if adj_val:
        sns.swarmplot(x=adj_cat, y=adj_val, order=shown, ax=ax_adj, color='k', size=3, alpha=0.7)
    for i, p in enumerate(shown):
        vals = adj.get(p, [])
        if vals:
            med = float(np.median(vals))
            ax_adj.plot([i - 0.4, i + 0.4], [med] * 2, color='tab:blue', lw=2.0)
            ax_adj.annotate('med {:.3f}'.format(med), (i, med), textcoords='offset points',
                            xytext=(6, 2), fontsize=7, color='tab:blue')
    ax_adj.axhline(0.0, color='0.6', lw=0.8, ls=':')
    ax_adj.axhline(min_adj_r2, color='tab:red', lw=1.2, ls='--',
                   label='cutoff {:.3f}'.format(min_adj_r2))
    ax_adj.legend(loc='upper right', fontsize=7)
    ax_adj.set_ylabel('Adjusted pseudo-R$^2$')
    ax_adj.set_xticks(positions)
    ax_adj.set_xticklabels(
        ['{} [{}]\nlag n={}, r$^2$ n={}'.format(
            p, resolve_period(p, period), len(lags.get(p, [])), len(adj.get(p, [])))
         for p in shown], fontsize=8)
    ax_adj.set_xlim(-0.6, len(shown) - 0.4)
    sns.despine(fig=fig)

    fig.tight_layout()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = name or 'encoding_summary{}{}{}'.format(
            _period_suffix(period), _joint_group_suffix(joint_group),
            _suffix(use_threshold_crossings))
        plotting.savefig(save_dir, name, fig=fig)
        rs('Saved encoding summary ({} predictor(s), {} session(s)).'.format(
            len(shown), len(found)))
    return fig
