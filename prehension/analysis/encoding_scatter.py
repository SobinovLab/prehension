#!python3
# -*- coding: utf-8 -*-
"""
Compare the encoding-model performance (adjusted pseudo-R2) of predictors against each
other, for whichever models a session has.

Discovers the encoding/encoding_<predictor>.json files a session (or set of sessions)
has and aligns each unit's adjusted pseudo-R2 across predictors by unit id.  Two figures
with the same one-panel-per-predictor-pair arrangement are produced: encoding_scatter
draws the pairwise scatter with the identity line (points above the diagonal favour the
y-axis predictor, as in the manuscript's model-comparison figure); encoding_difference_hist
draws, for each pair, the histogram of the paired per-unit difference (y - x).  Both annotate
every panel with the paired Wilcoxon signed-rank test and its matched-pairs rank-biserial
correlation effect size (tools.stats.nonparam_signedrank_rbc).

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

import numpy as np
import matplotlib.pyplot as plt

from .. import meta_session
from ..tools import io, plotting, stats
from ..tools.logs import rs, ws
from ..neural_plotting.common.traces import resolve_pooled_save_dir
from .encoding import (
    encoding_dir, resolve_period, ALL_PREDICTORS, PERIOD_ALL, _suffix, _period_suffix,
    _joint_group_suffix, DEFAULT_JOINT_GROUP, MIN_ADJ_R2)


def _load_available(processed_server, session, predictors, use_threshold_crossings, min_adj_r2,
                    period, joint_group):
    """{predictor: {'<session>:<unit_id>': adj_pr2}} for the saved encoding of one session.

    Reads the encoding_<predictor>[...] .json files matching the requested neural source,
    keeping the adjusted pseudo-R2 of each unit that is not null and is at or above
    `min_adj_r2` (poorly-encoded units are dropped).  Filtered to `predictors` when given, to
    each predictor's effective period (resolve_period(predictor, period); files written before
    the period feature have no 'period' key and count as PERIOD_ALL), and to `joint_group`
    (files written before the joint-group feature have no 'joint_group' key and count as 'all').
    """
    out = {}
    directory = encoding_dir(processed_server, session)
    if not os.path.isdir(directory):
        return out
    skipped_periods = set()   # periods present on disk but not the one requested (for a warning)
    for path in sorted(glob.glob(os.path.join(directory, 'encoding_*.json'))):
        if os.path.basename(path).endswith('_tx.json') != bool(use_threshold_crossings):
            continue
        data = io.load_json(path)
        predictor = data.get('predictor')
        if predictors and predictor not in predictors:
            continue
        if data.get('joint_group', 'all') != joint_group:
            continue
        file_period = data.get('period', PERIOD_ALL)
        if file_period != resolve_period(predictor, period):
            skipped_periods.add(file_period)
            continue
        out[predictor] = {
            '{}:{}'.format(session, u['unit_id']): u['adj_pr2']
            for u in data.get('units', [])
            if u.get('adj_pr2') is not None and u['adj_pr2'] >= min_adj_r2}
    # files exist for this session but none at the requested period -- explain the empty result
    if not out and skipped_periods:
        ws('{}: no encoding r2 files match the requested period; found period(s) {} instead. '
           'Pass a matching --period or generate the encoding for the requested period.'.format(
               session, sorted(skipped_periods)))
    return out


def _pooled_pairs(processed_server, sessions, predictors, use_threshold_crossings, min_adj_r2,
                  period, joint_group):
    """(preds, pairs, pair_xy): per-predictor unit adj-pR2 pooled across sessions, paired.

    Pools each unit's adjusted pseudo-R2 per predictor across `sessions` (empty -> all under
    processed_server), forms every predictor pair, and for each pair aligns by unit id the
    units present for both.  pair_xy is a list of (a, b, x, y) with x predictor a's adj-pR2 and
    y predictor b's over the shared units, so y - x is the paired difference (favouring b).
    Shared by encoding_scatter and encoding_difference_hist.  Raises when fewer than two
    predictors have saved encoding for the requested period and joint group.
    """
    found = sessions if sessions else meta_session.find_session_dirs(processed_server)

    per_pred = {}
    for session in found:
        for predictor, unit_map in _load_available(
                processed_server, session, predictors, use_threshold_crossings,
                min_adj_r2, period, joint_group).items():
            per_pred.setdefault(predictor, {}).update(unit_map)

    preds = sorted(per_pred)
    if len(preds) < 2:
        want = sorted({resolve_period(p, period) for p in (predictors or ALL_PREDICTORS)})
        raise ValueError(
            'Need at least two predictors with saved encoding to compare; found {} at period(s) '
            '{} for joint group {!r}. Generate the encoding for these (encoding.py --period ... '
            '--joint_group {}) or pass a matching --period / --joint_group.'.format(
                preds, want, joint_group, joint_group))

    pairs = [(a, b) for i, a in enumerate(preds) for b in preds[i + 1:]]
    pair_xy = []
    for a, b in pairs:
        common = sorted(set(per_pred[a]) & set(per_pred[b]))
        x = np.array([per_pred[a][u] for u in common], dtype=float)
        y = np.array([per_pred[b][u] for u in common], dtype=float)
        pair_xy.append((a, b, x, y))
    return preds, pairs, pair_xy


def _pair_stats_text(x, y):
    """Panel annotation for a paired predictor comparison: median diff, Wilcoxon p, rank-biserial.

    x, y are the paired per-unit adj-pR2 of predictors a, b; the difference y - x favours b, so a
    positive median / rank-biserial means b encodes better.  Uses the paired Wilcoxon signed-rank
    test and its matched-pairs rank-biserial effect size (tools.stats.nonparam_signedrank_rbc).
    Returns a short multi-line string, or None when there are no paired units; falls back to a
    "test n/a" line when the test cannot run (e.g. too few or all-tied differences).
    """
    if x.size == 0:
        return None
    median = float(np.median(y - x))
    try:
        res = stats.nonparam_signedrank_rbc(y, x)
        return 'median {:+.3f}\n{}\n$r_{{rb}}$ = {:+.2f}'.format(
            median, stats.format_p(res['p_value']), res['rank_biserial'])
    except Exception as e:  # noqa: BLE001
        return 'median {:+.3f}\ntest n/a ({})'.format(median, e)


def _annotate_stats(ax, x, y):
    """Draw _pair_stats_text in the top-left of `ax` (semi-transparent box), if there is any."""
    text = _pair_stats_text(x, y)
    if text is not None:
        ax.text(0.03, 0.97, text, transform=ax.transAxes, va='top', ha='left', fontsize=6.5,
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.7))


def encoding_scatter(processed_server, sessions, predictors=None, min_adj_r2=MIN_ADJ_R2,
                     use_threshold_crossings=False, name=None, save=True, save_dir=None,
                     period=None, joint_group=DEFAULT_JOINT_GROUP):
    """Scatter adjusted pseudo-R2 across predictors for the available encoding models.

    Pools each unit's adjusted pseudo-R2 per predictor across `sessions` (empty -> all
    sessions under processed_server), dropping units whose adjusted pseudo-R2 is below
    `min_adj_r2`, and draws one scatter panel per predictor pair (units kept for both), with
    the identity line.  `period` selects which trial sub-period's files to read (see
    resolve_period; None -> each predictor's default, so paired predictors may have been fit
    on different periods -- pass a single period for a like-for-like comparison).  `joint_group`
    selects which joint-group files to read ('hand' default; must match the encoding run).  Saved
    by default into <processed_server>/pooled_figures/encoding_scatter.  Returns the figure (or
    None when fewer than two predictors are available).
    """
    import seaborn as sns
    from matplotlib.ticker import MaxNLocator

    preds, pairs, pair_xy = _pooled_pairs(
        processed_server, sessions, predictors, use_threshold_crossings, min_adj_r2, period,
        joint_group)
    save_dir = resolve_pooled_save_dir(processed_server, 'encoding_scatter', save, save_dir)

    xn, yn = plotting.xy_numsubplots(len(pairs))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    # A shared square range across panels (equal x/y limits) puts the identity line at 45
    # degrees and makes the panels directly comparable.
    lo, hi = np.inf, -np.inf
    for _, _, x, y in pair_xy:
        if x.size:
            lo = min(lo, float(x.min()), float(y.min()))
            hi = max(hi, float(x.max()), float(y.max()))
    if not np.isfinite(lo):
        lo, hi = 0.0, 1.0
    margin = 0.05 * (hi - lo) if hi > lo else 0.05
    lo, hi = lo - margin, hi + margin

    # one shared set of "nice" ticks (0, 0.25, 0.5, ... family via the 2.5 step), applied to both
    # x and y so the two axes carry the identical ticks (the identity line then lands on matching
    # gridlines); computed once so every panel shares them too.
    shared_ticks = [t for t in MaxNLocator(nbins=4, steps=[1, 2.5, 5, 10]).tick_values(lo, hi)
                    if lo - 1e-9 <= t <= hi + 1e-9]

    for ax, (a, b, x, y) in zip(axs, pair_xy):
        sns.scatterplot(x=x, y=y, ax=ax, s=18, color='k', alpha=0.6, edgecolor='none',
                        legend=False)
        ax.plot([lo, hi], [lo, hi], color='0.6', linewidth=0.8, linestyle='--')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)
        ax.set_aspect('equal', adjustable='box')   # square subplot, identical x/y scale
        ax.set_xlabel('{} adj. pR$^2$'.format(a), fontsize=8)
        ax.set_ylabel('{} adj. pR$^2$'.format(b), fontsize=8)
        ax.set_title('{} vs {} ({} units)'.format(b, a, x.size), fontsize=8)
        ax.tick_params(labelsize=6)
        _annotate_stats(ax, x, y)   # paired Wilcoxon signed-rank + rank-biserial (favours b)
    for ax in axs[len(pairs):]:
        ax.axis('off')
    fig.suptitle('Encoding-model performance (adjusted pseudo-R$^2$) across predictors')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = name or 'encoding_scatter{}{}{}'.format(
            _period_suffix(period), _joint_group_suffix(joint_group),
            _suffix(use_threshold_crossings))
        plotting.savefig(save_dir, name, fig=fig)
        rs('Saved encoding scatter ({} predictors, {} pairs).'.format(len(preds), len(pairs)))
    return fig


def encoding_difference_hist(processed_server, sessions, predictors=None, min_adj_r2=MIN_ADJ_R2,
                             use_threshold_crossings=False, name=None, save=True, save_dir=None,
                             period=None, bins=30, joint_group=DEFAULT_JOINT_GROUP):
    """Histogram the paired adj-pR2 difference per predictor pair, with a Wilcoxon test.

    Same one-panel-per-predictor-pair arrangement as encoding_scatter (over the units present
    for both predictors of the pair), but each panel is the histogram of the paired per-unit
    difference d = <b> - <a> (b the y-axis predictor of the matching scatter, so positive d
    favours b).  A paired Wilcoxon signed-rank test and its matched-pairs rank-biserial
    correlation effect size (tools.stats.nonparam_signedrank_rbc; positive favours b) are
    computed per pair and annotated on the panel, with a dashed line at zero and the median
    difference marked.  The x-range is shared and symmetric across panels.  `period` selects the
    trial sub-period as in encoding_scatter.  Saved by default into
    <processed_server>/pooled_figures/encoding_difference_hist.  Returns the figure.
    """
    preds, pairs, pair_xy = _pooled_pairs(
        processed_server, sessions, predictors, use_threshold_crossings, min_adj_r2, period,
        joint_group)
    save_dir = resolve_pooled_save_dir(
        processed_server, 'encoding_difference_hist', save, save_dir)

    # shared symmetric difference range across panels so the zero line / skew are comparable
    max_abs = 0.0
    for _, _, x, y in pair_xy:
        if x.size:
            max_abs = max(max_abs, float(np.max(np.abs(y - x))))
    max_abs = max_abs or 1.0
    edges = np.linspace(-max_abs, max_abs, bins + 1)

    xn, yn = plotting.xy_numsubplots(len(pairs))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, figsize=(16, 9), squeeze=False)
    axs = axs.flatten()
    for ax, (a, b, x, y) in zip(axs, pair_xy):
        d = y - x
        ax.hist(d, bins=edges, color='0.5', edgecolor='white', linewidth=0.3)
        ax.axvline(0.0, color='0.3', linewidth=0.8, linestyle='--')
        ax.set_xlim(-max_abs, max_abs)
        ax.set_xlabel('{} - {} adj. pR$^2$'.format(b, a), fontsize=8)
        ax.set_ylabel('units', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title('{} vs {} ({} units)'.format(b, a, d.size), fontsize=8)
        if d.size == 0:
            continue
        ax.axvline(float(np.median(d)), color='tab:red', linewidth=1.2)
        _annotate_stats(ax, x, y)   # median diff, paired Wilcoxon signed-rank + rank-biserial
    for ax in axs[len(pairs):]:
        ax.axis('off')
    fig.suptitle('Encoding-model performance: paired adj. pseudo-R$^2$ differences '
                 '(Wilcoxon signed-rank)')
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        name = name or 'encoding_difference_hist{}{}{}'.format(
            _period_suffix(period), _joint_group_suffix(joint_group),
            _suffix(use_threshold_crossings))
        plotting.savefig(save_dir, name, fig=fig)
        rs('Saved encoding difference histograms ({} predictors, {} pairs).'.format(
            len(preds), len(pairs)))
    return fig
