#!python3
# -*- coding: utf-8 -*-
"""
Neural <-> kinematic/dynamic space alignment: how much structure the neural population shares
with several biomechanical descriptions of the same movement, per session.

For a session this assembles a continuous, shared-bin design across the trials -- every unit's
binned, smoothed firing rate as the neural space X (T, N), and one or more behavioural blocks
resampled onto the same bins as the behavioural space Y (T, M) -- and quantifies the shared
structure with three complementary methods through a single entry point:

    compare_spaces(X, Y, method="rrr", ...) -> dict

  * "rrr"  reduced-rank regression: cross-validated R2 as a function of rank, the selected
           rank, per-target R2, and the B = A @ C factorization, run in BOTH directions
           (X->Y and Y->X) with the asymmetry reported explicitly;
  * "cca"  canonical correlation analysis: canonical correlations, the number of significant
           components, weights and loadings, and a cross-validated (held-out) version, with
           ridge regularization for the N >> M / N ~ T regime;
  * "dsa"  Dynamical Similarity Analysis (Ostrow et al. 2023): delay embedding -> HAVOK/DMD
           operator per system -> Procrustes-over-vector-fields distance, wrapping the ``DSA``
           package when importable and falling back to an explicit implementation.

Every method returns the same schema -- a scalar headline 'similarity', a 'diagnostics' dict,
the fitted 'fit' objects, and a 'null' field (a surrogate null distribution with an effect
size, since all three metrics read comfortably non-zero on unrelated data, so the null is the
actual result).  The numerical kernels live in tools.space_similarity; this module is the
prehension-specific assembly, the three named comparisons, the file output and the plotting,
mirroring analysis.encoding.

Results are written per session to ``space_similarity/`` (a compact, human-readable JSON with
the headline / diagnostics / null summary, plus a companion .npz holding the heavy arrays --
fitted operators, per-target R2, loadings and the raw null values -- for inspection), in the
same style as ``encoding/``.  When the outputs are already present and --overwrite was not
given, the plotting reads straight from them instead of recomputing.

The three comparisons (thin configs over the same machinery; see COMPARISONS):
  A. neural activity  vs  joint angles
  B. neural activity  vs  joint angles + per-segment forces
  C. neural activity  vs  joint angles + joint torques
For B and C the behavioural space Y concatenates blocks with different units and very
different variances.  This is handled explicitly: per-block standardization by default, an
option to whiten each block and/or weight blocks (block_weights='equal' -> 1/sqrt(channels))
so a wide block (e.g. 30 position channels) does not swamp a narrow one (e.g. 3 force
channels), and a block-wise variance-explained breakdown (RRR per-block R2 / CCA per-block
loadings) so you can see whether the neural space tracks kinematics, dynamics, or both.
Block structure changes the interpretation of every headline number -- see compare_spaces.


================================  PARAMETER RECOMMENDATIONS  ================================

These matter more than the defaults suggest; the defaults below are conservative starting
points for T ~ 1e3-1e4 bins (a session of trials at 1/fps) and N ~ 50-300 units.  A companion
write-up is in documentation/neural_kinematic_alignment.md.

Shared preprocessing (matters a lot):
  * Bin width: the session kinematic/video period 1/fps (as in encoding/decoding).  Wider bins
    -> fewer, less autocorrelated samples; narrower -> more samples but heavier smoothing
    dependence.  This is the single biggest lever on every method's numbers.
  * n_pcs (reduce X to its top PCs before CCA/DSA): STRONGLY recommended when N approaches T.
    20-40 neural PCs usually retain the population structure while removing the directions that
    let CCA/DSA overfit.  For RRR it matters less (the rank constraint already regularizes).

RRR (reduced-rank regression):
  * Rank range: scan 1..min(N, M).  The selected rank is where held-out R2 plateaus; report
    the whole curve, not just the argmax (the plateau location is the interpretable quantity).
    Rank barely matters past the plateau -- extra ranks neither help nor hurt much with ridge.
  * Regularization alpha: matters a lot when T is small relative to N.  Start at alpha=1.0 on
    standardized inputs; increase (10-100) if the rank-1 held-out R2 is already negative
    (overfitting) and decrease (1e-2) if even full rank underfits.  Choose it by the held-out
    R2, never in-sample.
  * CV folds: 5 is fine; use fold splits that respect trials (whole trials held out) when
    trials are strongly autocorrelated -- otherwise adjacent bins leak across folds and inflate
    R2.  Fold count barely matters (5 vs 10).
  * T small relative to N: raise alpha, cap the rank scan (e.g. 1..20), and/or pre-reduce X
    with n_pcs.  With T < ~5N, trust the rank-1/2 numbers and the null far more than full rank.

CCA (canonical correlation analysis):
  * Ridge (alpha_x, alpha_y): the parameter that matters most.  Unregularized CCA with N ~ T
    reports canonical correlations near 1 that are pure overfitting.  Use alpha_x ~ 1e-1 to 1e1
    (times the mean covariance eigenvalue scale of standardized data); the honest check is that
    the CROSS-VALIDATED correlations stay high, not the in-sample ones.
  * Number of components: read it off the cross-validated correlations + n_significant (a
    permutation threshold on held-out folds), not the in-sample spectrum.  Typically only the
    first 1-3 components survive cross-validation for neural<->behaviour.
  * Overfitting failure mode when N ~ T: in-sample correlations ~ 1, cross-validated ~ 0.  Guard
    with ridge AND n_pcs AND the cross-validated numbers; if cv corr collapses while in-sample
    is high, you do not have enough data.
  * Sample size: as a rule of thumb you want T >> N + M (ideally T > ~5-10 x (N + M) after
    pre-reduction) for stable canonical correlations; below that, ridge + n_pcs are mandatory
    and only the leading component is trustworthy.

DSA (dynamical similarity analysis):
  * Delay-embedding dimension (n_delays) and lag (delay): the parameters DSA is MOST sensitive
    to.  n_delays should cover roughly one characteristic timescale of the dynamics
    (n_delays * delay * bin_width ~ the autocorrelation time, often ~10-50 bins for reach
    dynamics); too small misses the dynamics, too large inflates the operator and needs more
    data.  Sweep both and report the range -- do not trust a single (n_delays, delay).
  * Operator rank: the HAVOK/DMD reduced rank.  Set it from the delay-embedded singular
    spectrum (enough modes to capture ~90-95% variance, often 5-20).  Moderate sensitivity:
    too low collapses distinct dynamics together, too high fits noise.
  * Data per condition: DSA needs enough consecutive samples to fit the operator -- at least a
    few hundred bins per system, more with larger n_delays/rank.  With short trials, pool
    trials (the operator is fit on within-trial transitions).
  * Note: unlike RRR/CCA, DSA compares DYNAMICS, so a time-shift of Y barely changes it, and
    the operator is fit on the POOLED transitions -- so circular_shift and trial_shuffle leave
    it essentially unchanged (uninformative).  Use phase_randomize (independent phases) or
    time_shuffle as the DSA null.

Nulls (the actual result):
  * Prefer circular_shift (RRR/CCA) and phase_randomize (DSA) as the primary null -- both
    preserve each channel's autocorrelation/spectrum, so the null is "signals this smooth but
    unrelated", not "white noise".  time_shuffle is a floor check; trial_shuffle is the natural
    null for a trialized design (needs trial boundaries).
  * n_null: 200 for a stable z / percentile; 1000 for a publishable p-value.  Report the effect
    size (z against the null), not just the p-value.

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
import scipy.ndimage

from .. import meta_session
from ..tools import io, plotting
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..tools.stats import run_pca
from ..tools import space_similarity as ss
from ..neural_processing.common.spikes import (
    FILTER_SIGMA, read_nwb_spikes_and_ttl, get_trial_data_spike, fit_session_drift, drift_offset)
from ..neural_plotting.common.pooling import session_neural_context
from ..neural_plotting.common.behaviour import load_timepoints_into_msession
from .encoding import (
    _load_predictor, _component_exists, _trial_period_window, _suffix, _joint_group_suffix,
    JOINT_GROUPS, DEFAULT_JOINT_GROUP, PERIOD_ALL, PERIOD_ACTIVE_MOVEMENT)

SPACE_SIMILARITY_SUBDIR = 'space_similarity'

METHODS = ('rrr', 'cca', 'dsa')

# Behavioural signals a comparison block can be built from -- all reuse the encoding loaders
# (right-arm independent DOFs for the per-DOF ones: joint angles, velocities, torques).
ALIGNMENT_SIGNALS = ('joint_angles', 'joint_velocity', 'torques', 'digit_forces',
                     'segment_forces')

# The three named comparisons (A/B/C).  'signals' lists the behavioural blocks concatenated
# into Y, one block per signal (block structure drives the standardization / weighting and the
# per-block variance breakdown).  'period' is the trial sub-period each is evaluated on; all
# three default to the reach-grasp-retreat movement, matching the encoding models.
COMPARISONS = {
    'joint_angles': {
        'alias': 'A', 'signals': ('joint_angles',), 'period': PERIOD_ACTIVE_MOVEMENT,
        'label': 'neural vs joint angles'},
    'joint_angles_segment_forces': {
        'alias': 'B', 'signals': ('joint_angles', 'segment_forces'),
        'period': PERIOD_ACTIVE_MOVEMENT,
        'label': 'neural vs joint angles + segment forces'},
    'joint_angles_torques': {
        'alias': 'C', 'signals': ('joint_angles', 'torques'), 'period': PERIOD_ACTIVE_MOVEMENT,
        'label': 'neural vs joint angles + joint torques'},
}

# CLI alias -> comparison key (so 'A'/'B'/'C' can be passed as well as the descriptive names)
COMPARISON_ALIASES = {c['alias'].lower(): k for k, c in COMPARISONS.items()}

# Per-method primary null surrogate.  circular_shift preserves each channel's autocorrelation
# while destroying the timepoint alignment (right for the regression/correlation metrics);
# DSA compares dynamics, which a time shift barely perturbs, so it defaults to phase_randomize
# (same power spectrum, scrambled temporal structure).  Override with null=... in compare_spaces.
DEFAULT_NULL = {'rrr': 'circular_shift', 'cca': 'circular_shift', 'dsa': 'phase_randomize'}


# ======================================================================================
# Primary entry point
# ======================================================================================
def compare_spaces(X, Y, method='rrr', blocks=None, block_standardize=True, block_whiten=False,
                   block_weights=None, standardize_x=True, n_pcs=None, n_pcs_y=None, null='auto',
                   n_null=200, trials=None, seed=None, **kwargs):
    """Quantify the shared structure between neural activity X and behaviour Y.

    X is (T, N) neural activity (timepoints x channels); Y is (T, M) behavioural / biomechanical
    variables on the same T bins.  `method` dispatches to reduced-rank regression ('rrr'),
    canonical correlation analysis ('cca') or dynamical similarity analysis ('dsa').

    Block handling (matters for the interpretation): `blocks` is a list of per-block channel
    counts summing to M (e.g. [30, 3] for 30 position + 3 force channels); None treats Y as a
    single block.  With block_standardize (default True) each block is z-scored independently
    so blocks with different units/variances enter comparably; block_whiten additionally ZCA-
    whitens each block; block_weights scales blocks ('equal' -> 1/sqrt(channels), so a wide
    block does not swamp a narrow one).  standardize_x z-scores the neural columns; n_pcs first
    reduces X to its top n_pcs principal components (recommended for CCA/DSA when N approaches
    T).  These transforms change every headline number -- e.g. whitening a block makes its
    contribution scale with channel count, not variance.

    n_pcs_y reduces the (block-standardized) behavioural space Y to its top n_pcs_y principal
    components before the comparison.  Its main use is to make the number of behavioural
    predictors EQUAL across comparisons whose Y has different widths (e.g. joint angles alone vs
    joint angles + forces): set the same n_pcs_y (<= the smallest comparison's channel count) for
    all of them and every comparison enters with exactly n_pcs_y behavioural dimensions, so a
    difference in the headline is not just a difference in dimensionality.  It mixes the blocks,
    so the per-block variance breakdown is not meaningful when n_pcs_y is set.

    Null (the actual result): `null` selects the surrogate(s) that build the null distribution
    by recomputing the headline on a transformed Y with X fixed -- 'auto' (the method default,
    see DEFAULT_NULL), a name from tools.space_similarity.SURROGATES, a list of names, or 'all'.
    `n_null` draws per surrogate; `trials` (per-trial row counts) enables the trial-aware
    surrogates (trial_shuffle, per-trial circular_shift).  The effect size (z against the null)
    is reported, not just a p-value.

    Method parameters are passed through **kwargs (see the module PARAMETER RECOMMENDATIONS):
      rrr: ranks, alpha (default 1.0), n_folds (5);
      cca: alpha_x, alpha_y (default 1.0 each), n_components, n_folds (5), n_perm (100);
      dsa: n_delays (10), delay (1), rank (10), use_package ('auto'), use_pydmd ('auto'),
           dsa_n_restarts (3).

    Returns a dict with a consistent schema across methods:
      'method', 'metric', 'higher_is_more_similar', 'similarity' (scalar headline),
      'diagnostics' (method-specific), 'fit' (fitted arrays/objects), 'null' (the primary
      surrogate null + effect size), 'nulls' (every surrogate computed), 'extra_nulls' (nulls for
      any secondary headline -- for RRR the combined X<->Y mean ('symmetric') and the reverse Y->X
      ('y_to_x'); empty otherwise), and 'params'.
    """
    if method not in METHODS:
        raise ValueError('Unknown method {!r}; expected one of {}.'.format(method, METHODS))
    X, Y = ss.validate_pair(X, Y)

    block_ranges = _resolve_blocks(Y.shape[1], blocks)
    Xp = _preprocess_x(X, standardize_x, n_pcs)
    Yp = _preprocess_y(Y, block_ranges, block_standardize, block_whiten, block_weights)
    if n_pcs_y and n_pcs_y < Yp.shape[1]:
        # reduce Y to a fixed number of components (equalizes the behavioural predictor count
        # across comparisons); PCA mixes the blocks, so Y becomes a single block downstream
        Yp, _ = run_pca(Yp, n_pcs_y)
        block_ranges = [(0, Yp.shape[1])]

    if method == 'rrr':
        similarity, metric, higher, diagnostics, fit, headline, extra = _compare_rrr(
            Xp, Yp, block_ranges, seed=seed, **kwargs)
    elif method == 'cca':
        similarity, metric, higher, diagnostics, fit, headline, extra = _compare_cca(
            Xp, Yp, block_ranges, seed=seed, **kwargs)
    else:
        similarity, metric, higher, diagnostics, fit, headline, extra = _compare_dsa(
            Xp, Yp, seed=seed, **kwargs)

    nulls = _run_nulls(headline, Xp, Yp, method, null, n_null, trials, higher, seed,
                       observed=similarity)
    primary = DEFAULT_NULL[method] if null in ('auto', 'all', None) else (
        null[0] if isinstance(null, (list, tuple)) else null)
    primary = primary if primary in nulls else next(iter(nulls), None)

    # extra direction(s) -- e.g. RRR's reverse Y->X -- each get their own null the same way, so
    # both directions can be shown against their nulls (see _draw_null_grid)
    extra_nulls = {}
    for e in extra:
        e_nulls = _run_nulls(e['headline'], Xp, Yp, method, null, n_null, trials, e['higher'],
                             seed, observed=e['observed'])
        e_primary = primary if primary in e_nulls else next(iter(e_nulls), None)
        extra_nulls[e['label']] = {
            'observed': float(e['observed']), 'metric': e['metric'],
            'higher_is_more_similar': e['higher'], 'null': e_nulls.get(e_primary),
            'nulls': e_nulls}

    return {
        'method': method, 'metric': metric, 'higher_is_more_similar': higher,
        'similarity': float(similarity), 'diagnostics': diagnostics, 'fit': fit,
        'null': nulls.get(primary), 'nulls': nulls, 'extra_nulls': extra_nulls,
        'params': {'block_ranges': block_ranges, 'block_standardize': block_standardize,
                   'block_whiten': block_whiten, 'block_weights': block_weights,
                   'standardize_x': standardize_x, 'n_pcs': n_pcs, 'n_pcs_y': n_pcs_y,
                   'n_null': int(n_null), 'seed': seed, 'n_samples': int(Xp.shape[0]),
                   'n_neural': int(Xp.shape[1]), 'n_behaviour': int(Yp.shape[1]),
                   'method_kwargs': _jsonify(kwargs)},
    }


# --------------------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------------------
def _resolve_blocks(n_cols, blocks):
    """Validate the per-block channel counts against Y's width; None -> a single block."""
    if blocks is None:
        return [(0, n_cols)]
    if sum(blocks) != n_cols:
        raise ValueError('Block sizes {} sum to {} but Y has {} columns.'.format(
            list(blocks), sum(blocks), n_cols))
    return ss.blocks_from_sizes(blocks)


def _preprocess_x(X, standardize_x, n_pcs):
    """Z-score the neural columns and optionally reduce to the top n_pcs principal components."""
    if n_pcs and n_pcs < X.shape[1]:
        scores, _ = run_pca(X, n_pcs)          # run_pca z-scores internally
        return scores
    if standardize_x:
        mean, std = X.mean(axis=0), X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mean) / std
    return X


def _preprocess_y(Y, block_ranges, standardize, whiten, weights):
    """Apply per-block standardization / whitening / weighting to the behavioural space Y."""
    if standardize:
        Y = ss.standardize_blocks(Y, block_ranges)
    if whiten:
        Y = ss.whiten_block(Y, block_ranges)
    if weights is not None:
        Y = ss.apply_block_weights(Y, block_ranges, weights)
    return Y


# --------------------------------------------------------------------------------------
# Per-method comparisons -- each returns (similarity, metric, higher, diagnostics, fit, headline)
# where headline(Xp, Yp_surrogate) recomputes just the scalar headline for the null.
# --------------------------------------------------------------------------------------
def _compare_rrr(X, Y, block_ranges, ranks=None, alpha=1.0, n_folds=5, seed=None):
    """Reduced-rank regression in both directions, with the asymmetry and per-block R2."""
    fwd = ss.rrr_cv(X, Y, ranks=ranks, alpha=alpha, n_folds=n_folds, seed=seed)   # X -> Y
    rev = ss.rrr_cv(Y, X, ranks=ranks, alpha=alpha, n_folds=n_folds, seed=seed)   # Y -> X
    r2_xy, r2_yx = fwd['cv_r2_selected'], rev['cv_r2_selected']

    block_r2 = [float(np.nanmean(fwd['per_target_r2'][a:b])) for a, b in block_ranges]
    diagnostics = {
        'cv_r2_x_to_y': r2_xy, 'cv_r2_y_to_x': r2_yx,
        'asymmetry': float(r2_xy - r2_yx),
        'asymmetry_ratio': float(r2_xy / r2_yx) if np.isfinite(r2_yx) and r2_yx != 0 else np.nan,
        'selected_rank_x_to_y': fwd['selected_rank'], 'selected_rank_y_to_x': rev['selected_rank'],
        'ranks': fwd['ranks'], 'cv_r2_curve_x_to_y': fwd['cv_r2'].tolist(),
        'cv_r2_curve_y_to_x': rev['cv_r2'].tolist(),
        'block_r2_x_to_y': block_r2, 'block_ranges': block_ranges,
        'direction_note': 'X->Y predicts behaviour from neural (decoding-like); '
                          'Y->X predicts neural from behaviour (encoding-like); '
                          'asymmetry = R2(X->Y) - R2(Y->X).'}
    fit = {'A_x_to_y': fwd['A'], 'C_x_to_y': fwd['C'], 'B_x_to_y': fwd['B'],
           'intercept_x_to_y': fwd['intercept'], 'per_target_r2_x_to_y': fwd['per_target_r2'],
           'A_y_to_x': rev['A'], 'C_y_to_x': rev['C'], 'B_y_to_x': rev['B']}

    selected_fwd, selected_rev = fwd['selected_rank'], rev['selected_rank']

    def headline(Xp, Yp):                     # X -> Y (decode behaviour from neural)
        return ss.rrr_cv(Xp, Yp, ranks=[selected_fwd], alpha=alpha, n_folds=n_folds,
                         seed=seed, return_fit=False)['cv_r2_selected']

    def headline_rev(Xp, Yp):                 # Y -> X (encode neural from behaviour; Y predicts X)
        return ss.rrr_cv(Yp, Xp, ranks=[selected_rev], alpha=alpha, n_folds=n_folds,
                         seed=seed, return_fit=False)['cv_r2_selected']

    def headline_sym(Xp, Yp):                 # X <-> Y (both directions, mean of the two R2s)
        return 0.5 * (headline(Xp, Yp) + headline_rev(Xp, Yp))

    # the combined X<->Y panel (symmetric mean) plus each single direction, each with its own null
    extra = [
        {'label': 'symmetric', 'headline': headline_sym, 'observed': 0.5 * (r2_xy + r2_yx),
         'metric': 'cv_r2', 'higher': True},
        {'label': 'y_to_x', 'headline': headline_rev, 'observed': r2_yx, 'metric': 'cv_r2',
         'higher': True}]
    return r2_xy, 'cv_r2', True, diagnostics, fit, headline, extra


def _compare_cca(X, Y, block_ranges, alpha_x=1.0, alpha_y=1.0, n_components=None, n_folds=5,
                 n_perm=100, seed=None):
    """Canonical correlation analysis: cross-validated correlations, significance, block loadings."""
    cv = ss.cca_cv(X, Y, alpha_x=alpha_x, alpha_y=alpha_y, n_components=n_components,
                   n_folds=n_folds, n_perm=n_perm, seed=seed)
    fit = cv['fit']
    headline_val = float(cv['cv_correlations'][0]) if cv['cv_correlations'].size else np.nan

    # per-block mean |loading| on the leading canonical variate -- which blocks the shared
    # component lives in (kinematics vs dynamics)
    lead = 0
    block_loading = [float(np.nanmean(np.abs(fit['y_loadings'][a:b, lead])))
                     for a, b in block_ranges]
    diagnostics = {
        'canonical_correlations_in_sample': fit['correlations'].tolist(),
        'cv_correlations': cv['cv_correlations'].tolist(),
        'n_significant': cv['n_significant'], 'sig_threshold': cv['sig_threshold'],
        'n_components': cv['n_components'],
        'block_loadings_leading': block_loading, 'block_ranges': block_ranges,
        'overfit_note': 'compare in-sample vs cv correlations: a large gap (in-sample ~ 1, '
                        'cv ~ 0) means CCA is overfitting -- raise alpha / n_pcs or get more data.'}
    fitted = {'x_weights': fit['x_weights'], 'y_weights': fit['y_weights'],
              'x_loadings': fit['x_loadings'], 'y_loadings': fit['y_loadings'],
              'correlations': fit['correlations']}

    k = cv['n_components']

    def headline(Xp, Yp):
        res = ss.cca_cv(Xp, Yp, alpha_x=alpha_x, alpha_y=alpha_y, n_components=k,
                        n_folds=n_folds, n_perm=0, seed=seed, return_fit=False)
        return float(res['cv_correlations'][0]) if res['cv_correlations'].size else np.nan

    return headline_val, 'canonical_correlation', True, diagnostics, fitted, headline, []


def _compare_dsa(X, Y, n_delays=10, delay=1, rank=10, use_package='auto', use_pydmd='auto',
                 dsa_n_restarts=3, seed=None):
    """Dynamical Similarity Analysis: delay-embed -> HAVOK/DMD operator -> Procrustes distance."""
    res = ss.dsa_distance(X, Y, n_delays=n_delays, delay=delay, rank=rank,
                          use_package=use_package, use_pydmd=use_pydmd,
                          n_restarts=dsa_n_restarts, seed=seed)
    diagnostics = {
        'distance': res['distance'], 'rank_used': res['rank_used'],
        'n_delays': res['n_delays'], 'delay': res['delay'], 'backend': res['backend'],
        'distance_note': 'DSA distance: LOWER = more similar dynamics (higher_is_more_similar '
                         'is False). The operator is fit on pooled transitions, so circular_shift '
                         'and trial_shuffle barely change it; use phase_randomize (independent) '
                         'or time_shuffle as the null.'}
    fit = {'A_x': res['A_x'], 'A_y': res['A_y']}

    # X is fixed across the null, so its operator A_x is fit once and only Y's is refit per
    # surrogate (the neural delay-embed + DMD is by far the most expensive step -- recomputing
    # it n_null times would dominate the runtime).  The explicit path allows this reuse; when
    # the DSA package is the backend, fall back to the full recompute to keep the metric identical.
    if res['backend'] == 'explicit':
        A_x, rank_used = res['A_x'], res['rank_used']

        def headline(Xp, Yp):
            Ey = ss.delay_embed(Yp, n_delays, delay)
            A_y, _ = ss.dmd_operator(Ey, rank=rank_used, use_pydmd=use_pydmd)
            return ss.procrustes_vector_field_distance(
                A_x, A_y, n_restarts=dsa_n_restarts, seed=seed)[0]
    else:
        def headline(Xp, Yp):
            return ss.dsa_distance(Xp, Yp, n_delays=n_delays, delay=delay, rank=rank,
                                   use_package=use_package, use_pydmd=use_pydmd,
                                   n_restarts=dsa_n_restarts, seed=seed)['distance']

    return res['distance'], 'dsa_distance', False, diagnostics, fit, headline, []


# --------------------------------------------------------------------------------------
# Null distributions
# --------------------------------------------------------------------------------------
def _resolve_null_names(method, null):
    """Which surrogate name(s) to run: 'auto' -> the method default, 'all' -> every surrogate."""
    if null in ('auto', None):
        return [DEFAULT_NULL[method]]
    if null == 'all':
        return list(ss.SURROGATES)
    names = [null] if isinstance(null, str) else list(null)
    bad = [n for n in names if n not in ss.SURROGATES]
    if bad:
        raise ValueError('Unknown surrogate(s) {}; expected from {}.'.format(
            bad, sorted(ss.SURROGATES)))
    return names


def _run_nulls(headline, Xp, Yp, method, null, n_null, trials, higher, seed, observed):
    """Build the null distribution(s) by recomputing the headline on surrogate Y (X fixed).

    For each requested surrogate, n_null transformed copies of Y are generated and the scalar
    headline recomputed; the resulting null values and an effect size (tools.space_similarity.
    effect_size of the `observed` headline against them, oriented by `higher`) are stored.  The
    `observed` value is the headline already computed on the real data, so it is measured
    exactly the way the null is (apples-to-apples).  Surrogates that need trial boundaries are
    skipped with a warning when `trials` is not given.
    """
    names = _resolve_null_names(method, null)
    rng = np.random.RandomState(seed)
    out = {}
    for name in names:
        if name == 'trial_shuffle' and not trials:
            ws('Null {!r} needs per-trial boundaries (trials=...); skipping it.'.format(name))
            continue
        surrogate = ss.SURROGATES[name]
        values = np.full(int(n_null), np.nan)
        for i in range(int(n_null)):
            try:
                Ys = surrogate(Yp, rng, trials=trials)
                values[i] = headline(Xp, Ys)
            except Exception as e:  # noqa: BLE001 - a degenerate surrogate should not abort the run
                ws('Null {} draw {} failed ({}); recorded as NaN.'.format(name, i, e))
        es = ss.effect_size(observed, values, higher)
        es.update({'surrogate': name, 'values': values, 'n': int(np.sum(np.isfinite(values)))})
        out[name] = es
    return out


# ======================================================================================
# Session data assembly (neural X + concatenated behavioural blocks Y on shared bins)
# ======================================================================================
def _signal_exists(trial, signal):
    """Whether a trial has the file(s) for one behavioural signal."""
    return _component_exists(trial, signal)


def _load_alignment_signal(trial, signal, joint_group='all'):
    """Load one behavioural signal as (times, channel_names, values (n_channels, n_times)).

    All signals reuse the encoding loader (joint angles / velocity / torques restricted to the
    right-arm independent DOFs and the requested `joint_group`; digit / segment forces as-is).
    """
    return _load_predictor(trial, signal, joint_group)


def pool_alignment_trials(server, processed_server, session, signals, bin_width=None,
                          filter_sigma=FILTER_SIGMA, use_threshold_crossings=False,
                          period=PERIOD_ALL, drift_correct=True, joint_group='all'):
    """Per-trial (neural rate X, concatenated behaviour Y) segments for a session, shared bins.

    Mirrors analysis.encoding._pool_encoding_trials / analysis.decoding.pool_decoding_trials:
    reads the neural source, pairs TTL pulses to trials positionally (meta_neural skip_ttl /
    skip_ttl_last), and for every successful trial that has ALL the requested `signals`, bins +
    Gaussian-smooths each unit's rate (subtracting the linear session drift when drift_correct)
    and resamples every behavioural signal's channels onto the same bin centres, over the
    signals' overlapping time range (seconds-since-TTL frame).  The signals are concatenated in
    order into one behavioural design, one block per signal.  `period` optionally crops each
    trial to a sub-period (_trial_period_window).  Returns
    (trials_X, trials_Y, unit_ids, block_names, block_sizes, bin_width, fps): per-trial lists of
    (n_bins_i, n_units) and (n_bins_i, sum_channels); block_names is a list of (signal, channel
    names) and block_sizes the per-signal channel counts (the block structure of Y).
    """
    nwb_path, meta_neural, rserv, pserv = session_neural_context(
        server, processed_server, session, use_threshold_crossings)
    spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(nwb_path)
    skip_ttl = resolve_meta_arg(None, meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(None, meta_neural, 'skip_ttl_last', 0)

    mstruct, _, _, msession = meta_session.load_meta_information(rserv, pserv)
    if period != PERIOD_ALL:
        load_timepoints_into_msession(msession, mstruct)

    fps = float(mstruct.get('fps') or 0) or None
    if bin_width is None:
        if not fps:
            raise ValueError("bin_width not given and no 'fps' in meta_structure for {}.".format(
                session))
        bin_width = 1.0 / fps

    # positional pulse<->trial pairing (as in neural_plotting.common.pooling)
    if skip_ttl and skip_ttl > 0:
        events_time = events_time[skip_ttl:]
    elif skip_ttl and skip_ttl < 0:
        msession = msession[-skip_ttl:]
    if skip_ttl_last and skip_ttl_last > 0:
        events_time = events_time[:-skip_ttl_last]
    elif skip_ttl_last and skip_ttl_last < 0:
        msession = msession[:skip_ttl_last]
    if len(events_time) != len(msession):
        raise ValueError('{} TTL pulses vs {} trials (skip_ttl={}, skip_ttl_last={}).'.format(
            len(events_time), len(msession), skip_ttl, skip_ttl_last))

    session_spikes = get_trial_data_spike(spikes, events_time)
    n_units = len(unit_ids)
    for trial_spikes, ev in zip(session_spikes, events_time):
        for i_n in range(n_units):
            trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ev[0]
    slopes, t_ref = fit_session_drift(spikes, events_time) if drift_correct else (None, 0.0)

    freq = 1.0 / bin_width
    sigma_bins = filter_sigma / bin_width

    trials_X, trials_Y, block_names, block_sizes = [], [], None, None
    for trial, tspk, ev in zip(msession, session_spikes, events_time):
        if not trial.success or not all(_signal_exists(trial, s) for s in signals):
            continue
        try:
            loaded = [_load_alignment_signal(trial, s, joint_group) for s in signals]
        except Exception as e:  # noqa: BLE001
            ws('Session {} trial {}: could not read {} ({}); skipping.'.format(
                session, trial.trial_number, signals, e))
            continue
        if any(np.asarray(t).size < 2 for t, _, _ in loaded):
            continue

        # one bin grid over the signals' overlapping time range
        t_lo = max(float(t[0]) for t, _, _ in loaded)
        t_hi = min(float(t[-1]) for t, _, _ in loaded)
        bins = np.arange(t_lo, t_hi + bin_width, bin_width)
        if bins.size < 3:
            continue
        centers = bins[:-1] + bin_width / 2

        rate = np.zeros((centers.size, n_units))
        for i_n in range(n_units):
            counts, _ = np.histogram(tspk[i_n], bins=bins)
            rate[:, i_n] = (scipy.ndimage.gaussian_filter1d(counts * freq, sigma_bins)
                            - drift_offset(slopes, t_ref, i_n, float(ev[0])))

        cols, names_this, sizes_this = [], [], []
        for (t_sig, names_sig, vals_sig), sig in zip(loaded, signals):
            for name, channel in zip(names_sig, vals_sig):
                cols.append(np.interp(centers, np.asarray(t_sig, dtype=float),
                                      np.asarray(channel, dtype=float)))
            names_this.append((sig, list(names_sig)))
            sizes_this.append(len(names_sig))
        design = np.column_stack(cols)
        if block_sizes is None:
            block_names, block_sizes = names_this, sizes_this
        elif sizes_this != block_sizes:
            # a trial with a different channel count would break the block structure -- skip it
            ws('Session {} trial {}: {} channel counts {} != {}; skipping.'.format(
                session, trial.trial_number, signals, sizes_this, block_sizes))
            continue

        # crop to the requested sub-period (whole trial for PERIOD_ALL)
        if period != PERIOD_ALL:
            window = _trial_period_window(trial, period)
            if window is None:
                continue
            keep = (centers >= window[0]) & (centers <= window[1])
            if not np.any(keep):
                continue
            rate, design = rate[keep], design[keep]
        trials_X.append(rate)
        trials_Y.append(design)

    if not trials_X:
        raise ValueError('No usable trials for signals {} in session {}.'.format(signals, session))
    return trials_X, trials_Y, list(unit_ids), block_names, block_sizes, bin_width, fps


# ======================================================================================
# File output / reading (space_similarity/)
# ======================================================================================
def space_similarity_dir(processed_server, session):
    """Path to a session's space_similarity/ output folder."""
    return os.path.join(processed_server, session, SPACE_SIMILARITY_SUBDIR)


def alignment_paths(processed_server, session, comparison, method, use_threshold_crossings=False,
                    joint_group='all'):
    """(json_path, npz_path) for one comparison x method result of a session (and joint group)."""
    base = 'alignment_{}_{}{}{}'.format(comparison, method, _joint_group_suffix(joint_group),
                                        _suffix(use_threshold_crossings))
    directory = space_similarity_dir(processed_server, session)
    return os.path.join(directory, base + '.json'), os.path.join(directory, base + '.npz')


def _jsonify(obj):
    """Recursively convert numpy types / arrays to JSON-friendly python (arrays -> lists)."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonify(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else round(v, 6)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _null_summary(null):
    """Compact, JSON-friendly summary of one null (drops the raw values, kept in the .npz)."""
    if null is None:
        return None
    return {k: _jsonify(v) for k, v in null.items() if k != 'values'}


def save_alignment(processed_server, session, comparison, method, result, block_names,
                   bin_width, fps, period, use_threshold_crossings=False, joint_group='all'):
    """Write one comparison x method result to space_similarity/: JSON summary + companion .npz.

    The JSON holds the headline similarity, diagnostics, per-null summaries and the run
    parameters (human-readable, like encoding/); the .npz holds the heavy arrays -- the fitted
    operators / factors, per-target R2, loadings and the raw null values -- so they can be
    inspected and the figures redrawn without recomputing.
    """
    json_path, npz_path = alignment_paths(processed_server, session, comparison, method,
                                          use_threshold_crossings, joint_group)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    payload = {
        'session': session, 'comparison': comparison, 'alias': COMPARISONS[comparison]['alias'],
        'label': COMPARISONS[comparison]['label'], 'method': method,
        'source': 'threshold_crossings' if use_threshold_crossings else 'sorted',
        'period': period, 'joint_group': joint_group, 'bin_width_s': bin_width, 'fps': fps,
        'metric': result['metric'], 'higher_is_more_similar': result['higher_is_more_similar'],
        'similarity': _jsonify(result['similarity']),
        'signals': [s for s, _ in block_names], 'block_sizes': [len(c) for _, c in block_names],
        'diagnostics': _jsonify(result['diagnostics']),
        'null': _null_summary(result['null']),
        'nulls': {k: _null_summary(v) for k, v in result['nulls'].items()},
        'extra_nulls': {label: {'observed': _jsonify(ex['observed']), 'metric': ex['metric'],
                                'higher_is_more_similar': ex['higher_is_more_similar'],
                                'null': _null_summary(ex['null']),
                                'nulls': {k: _null_summary(v) for k, v in ex['nulls'].items()}}
                        for label, ex in result.get('extra_nulls', {}).items()},
        'params': _jsonify(result['params']),
        'npz': os.path.basename(npz_path),
    }
    io.save_json(payload, json_path)

    arrays = {}
    for key, val in result['fit'].items():
        arrays['fit_' + key] = np.asarray(val)          # preserve shape/dtype for inspection
    for name, null in result['nulls'].items():
        arrays['null_' + name] = np.asarray(null['values'], dtype=float)
    for label, ex in result.get('extra_nulls', {}).items():
        for name, null in ex['nulls'].items():
            arrays['extra_null_{}_{}'.format(label, name)] = np.asarray(null['values'], dtype=float)
    np.savez_compressed(npz_path, **arrays)
    rs('Wrote space similarity {} / {} [{}] -> {}'.format(
        comparison, method, period, os.path.basename(json_path)))
    return json_path


def load_alignment(processed_server, session, comparison, method, use_threshold_crossings=False,
                   with_arrays=False, joint_group='all'):
    """Read a saved comparison x method JSON (and, when with_arrays, its .npz), or None if absent."""
    json_path, npz_path = alignment_paths(processed_server, session, comparison, method,
                                          use_threshold_crossings, joint_group)
    if not os.path.exists(json_path):
        return None
    data = io.load_json(json_path)
    if with_arrays and os.path.exists(npz_path):
        with np.load(npz_path) as npz:
            data['arrays'] = {k: npz[k] for k in npz.files}
    return data


# ======================================================================================
# Orchestration: compute + save, then plot from the saved files
# ======================================================================================
def _resolve_comparisons(comparisons):
    """Normalize a list of comparison keys / aliases (A/B/C) to canonical keys.

    None, an empty list, or a list containing 'all' -> every comparison (COMPARISONS), which is
    the default; otherwise the given keys / aliases in order, de-duplicated.
    """
    if not comparisons or any(str(c).lower() == 'all' for c in comparisons):
        return list(COMPARISONS)
    out = []
    for c in comparisons:
        key = c if c in COMPARISONS else COMPARISON_ALIASES.get(str(c).lower())
        if key is None:
            raise ValueError("Unknown comparison {!r}; expected 'all', keys {} or aliases "
                             "{}.".format(c, list(COMPARISONS), sorted(COMPARISON_ALIASES)))
        if key not in out:
            out.append(key)
    return out


def run_alignment(server, processed_server, sessions, comparisons=None, methods=METHODS,
                  bin_width=None, use_threshold_crossings=False, drift_correct=True,
                  block_standardize=True, block_whiten=False, block_weights=None, n_pcs=None,
                  n_pcs_y=None, null='auto', n_null=200, seed=None, overwrite=False,
                  method_kwargs=None, joint_group=DEFAULT_JOINT_GROUP):
    """Compute and save the space-similarity results for each session, comparison and method.

    For every session (empty `sessions` -> all under processed_server), comparison (COMPARISONS;
    keys or A/B/C aliases) and method (rrr/cca/dsa), assembles the neural + concatenated
    behavioural design (pool_alignment_trials) on the comparison's trial sub-period, runs
    compare_spaces with the given block handling / null settings, and writes the result to
    space_similarity/ (save_alignment).  Existing outputs are skipped unless `overwrite`.
    `method_kwargs` is an optional {method: {kwarg: value}} of per-method parameters (ranks,
    alpha, n_delays, ...).  Returns the list of written JSON paths.
    """
    found = sessions if sessions else meta_session.find_session_dirs(processed_server)
    comps = _resolve_comparisons(comparisons)
    mkw = method_kwargs or {}
    written = []
    for session in found:
        for comparison in comps:
            cfg = COMPARISONS[comparison]
            # assemble once per comparison, reused across methods (skip when all methods exist)
            todo = [m for m in methods if overwrite or load_alignment(
                processed_server, session, comparison, m, use_threshold_crossings,
                joint_group=joint_group) is None]
            if not todo:
                rs('  {} / {} {{{}}}: all methods exist; skipping (use --overwrite).'.format(
                    session, comparison, joint_group))
                continue
            try:
                trials_X, trials_Y, unit_ids, block_names, block_sizes, bw, fps = \
                    pool_alignment_trials(
                        server, processed_server, session, cfg['signals'], bin_width=bin_width,
                        use_threshold_crossings=use_threshold_crossings, period=cfg['period'],
                        drift_correct=drift_correct, joint_group=joint_group)
            except Exception as e:  # noqa: BLE001
                ws('Skipping {} / {} {{{}}}: {}'.format(session, comparison, joint_group, e))
                continue
            X = np.vstack(trials_X)
            Y = np.vstack(trials_Y)
            trials = [xi.shape[0] for xi in trials_X]
            rs('Aligning {} / {} [{}] {{{}}}: {} samples, {} neural, {} behaviour ({} block(s)), '
               '{} trial(s).'.format(session, comparison, cfg['period'], joint_group, X.shape[0],
                                     X.shape[1], Y.shape[1], len(block_sizes), len(trials)))
            for method in todo:
                try:
                    result = compare_spaces(
                        X, Y, method=method, blocks=block_sizes,
                        block_standardize=block_standardize, block_whiten=block_whiten,
                        block_weights=block_weights, n_pcs=n_pcs, n_pcs_y=n_pcs_y, null=null,
                        n_null=n_null, trials=trials, seed=seed, **mkw.get(method, {}))
                except Exception as e:  # noqa: BLE001
                    ws('Skipping {} / {} / {}: {}'.format(session, comparison, method, e))
                    continue
                written.append(save_alignment(
                    processed_server, session, comparison, method, result, block_names, bw, fps,
                    cfg['period'], use_threshold_crossings, joint_group))
                _log_result(session, comparison, method, result)
    return written


def _log_result(session, comparison, method, result):
    """One-line log of a result's headline and primary-null effect size."""
    null = result['null'] or {}
    rs('  {} / {} / {}: {}={:.3f}, null {} z={}, {}.'.format(
        session, comparison, method, result['metric'], result['similarity'],
        null.get('surrogate', '-'),
        '{:.2f}'.format(null['z']) if null.get('z') is not None and np.isfinite(null.get('z'))
        else 'nan',
        _p_text(null.get('p_value'))))


def _p_text(p):
    """Compact p-value text (None -> 'p=n/a')."""
    if p is None or not np.isfinite(p):
        return 'p=n/a'
    return 'p<0.001' if p < 0.001 else 'p={:.3f}'.format(p)


def alignment_analysis(server, processed_server, sessions, comparisons=None, methods=METHODS,
                       overwrite=False, save=True, **kwargs):
    """Compute + save (unless present and not overwrite), then plot from the saved files.

    The convenience top-level the driver script calls: runs run_alignment (which skips existing
    outputs unless `overwrite`) and then plot_alignment, which reads back the saved
    space_similarity/ files.  `kwargs` are forwarded to run_alignment (block handling, null
    settings, method_kwargs, ...).  Returns the list of figures.
    """
    run_alignment(server, processed_server, sessions, comparisons=comparisons, methods=methods,
                  overwrite=overwrite, **kwargs)
    return plot_alignment(processed_server, sessions, comparisons=comparisons, methods=methods,
                          use_threshold_crossings=kwargs.get('use_threshold_crossings', False),
                          save=save, joint_group=kwargs.get('joint_group', DEFAULT_JOINT_GROUP))


# ======================================================================================
# Plotting from the saved files
# ======================================================================================
def plot_alignment(processed_server, sessions, comparisons=None, methods=METHODS,
                   use_threshold_crossings=False, save=True, joint_group=DEFAULT_JOINT_GROUP):
    """Draw the saved space-similarity results per session, reading straight from the files.

    For each session with saved results, produces a null-vs-observed figure (a grid of
    comparison x method panels: the surrogate null distribution with the observed headline
    marked and the effect size annotated) and, when the data are present, an RRR rank-curve
    figure and a per-block variance-breakdown figure.  Figures are saved to
    <session>/prehension_plots/ unless save is False.  Returns the list of figures.
    """
    found = sessions if sessions else meta_session.find_session_dirs(processed_server)
    comps = _resolve_comparisons(comparisons)
    figures = []
    for session in found:
        loaded = {}
        for comparison in comps:
            for method in methods:
                data = load_alignment(processed_server, session, comparison, method,
                                      use_threshold_crossings, with_arrays=True,
                                      joint_group=joint_group)
                if data is not None:
                    loaded[(comparison, method)] = data
        if not loaded:
            ws('No saved space-similarity files for session {} (joint group {!r}); run the '
               'analysis first.'.format(session, joint_group))
            continue
        save_dir = os.path.join(processed_server, session, 'prehension_plots')
        # the combined 'together on one plot' view across comparisons, then the per-comparison
        # detail (null histograms, per-block breakdown)
        figures.append(_draw_comparison_overlay(session, comps, methods, loaded, save_dir,
                                                use_threshold_crossings, save, joint_group))
        figures.append(_draw_null_grid(session, comps, methods, loaded, save_dir,
                                       use_threshold_crossings, save, joint_group))
        rrr_fig = _draw_rrr_overlay(session, comps, loaded, save_dir, use_threshold_crossings,
                                    save, joint_group)
        if rrr_fig is not None:
            figures.append(rrr_fig)
        block_fig = _draw_block_breakdown(session, comps, loaded, save_dir,
                                          use_threshold_crossings, save, joint_group)
        if block_fig is not None:
            figures.append(block_fig)
    return figures


def _null_columns(methods):
    """(title, method, direction) per subplot; RRR gets the combined X<->Y and both directions."""
    columns = []
    for method in methods:
        if method == 'rrr':
            columns.append(('RRR  X$\\leftrightarrow$Y  (both directions, mean)', 'rrr',
                            'symmetric'))
            columns.append(('RRR  X$\\to$Y  (decode behaviour from neural)', 'rrr', 'x_to_y'))
            columns.append(('RRR  Y$\\to$X  (encode neural from behaviour)', 'rrr', 'y_to_x'))
        else:
            columns.append((method.upper(), method, 'x_to_y'))
    return columns


def _draw_null_grid(session, comps, methods, loaded, save_dir, use_threshold_crossings, save,
                    joint_group='all'):
    """Space similarity vs null: one subplot per method (RRR split into -> and <-), nulls overlaid.

    Metrics differ across methods, so each method keeps its own subplot / x-axis; RRR gets two
    subplots -- one per direction (X->Y decode behaviour from neural, Y->X encode neural from
    behaviour), each against its own null.  Within a subplot every comparison's surrogate-null
    distribution is drawn together (a filled histogram coloured per comparison) with that
    comparison's observed headline as a vertical line of the same colour and the effect size (z)
    in the legend -- all distributions on the same plot, which is the point of the 'all' run.
    """
    import matplotlib.pyplot as plt

    colors = _comparison_colors(comps)
    columns = _null_columns(methods)
    n = len(columns)
    fig, axs = plt.subplots(nrows=1, ncols=n, squeeze=False, figsize=(1.0 + 4.0 * n, 4.6))
    for ax, (title, method, direction) in zip(axs[0], columns):
        ax.set_title(title, fontsize=9)
        if not _draw_null_method_panel(ax, method, comps, loaded, colors, direction):
            ax.axis('off')
    fig.suptitle('Neural <-> behaviour space similarity vs null -- {}{}'.format(
        session, ' (threshold crossings)' if use_threshold_crossings else ''))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        plotting.savefig(save_dir, 'alignment_null{}{}'.format(
            _joint_group_suffix(joint_group), _suffix(use_threshold_crossings)), fig=fig)
        rs('Saved space-similarity null overlay for session {}.'.format(session))
    return fig


def _direction_series(data, direction):
    """(observed, null_summary, null_values) for one method-direction, or None if unavailable.

    'x_to_y' is the stored primary headline (result['null'] / 'similarity'); 'y_to_x' is the
    RRR reverse direction from the 'extra_nulls' block written alongside it.
    """
    if direction == 'x_to_y':
        null = data.get('null') or {}
        values = data.get('arrays', {}).get('null_{}'.format(null.get('surrogate', '-')))
        return data.get('similarity'), null, values
    ex = (data.get('extra_nulls') or {}).get(direction)
    if not ex:
        return None
    null = ex.get('null') or {}
    values = data.get('arrays', {}).get('extra_null_{}_{}'.format(
        direction, null.get('surrogate', '-')))
    return ex.get('observed'), null, values


def _draw_null_method_panel(ax, method, comps, loaded, colors, direction='x_to_y'):
    """Overlay every comparison's null distribution + observed value for one method-direction on `ax`.

    Each comparison is drawn in its own colour (filled null histogram, matching observed line),
    with its observed value and effect size (z) in the legend label.  Returns True when at least
    one comparison was drawn (else the caller turns the subplot off) -- so an RRR reverse subplot
    is turned off when no comparison saved a reverse null.
    """
    drew, metric = False, 'similarity'
    for comparison in comps:
        data = loaded.get((comparison, method))
        if data is None:
            continue
        series = _direction_series(data, direction)
        if series is None:
            continue
        observed, null, values = series
        metric = data.get('metric', metric)
        z = null.get('z')
        color = colors[comparison]
        label = '{} [{}]  obs={}, z={}'.format(
            COMPARISONS[comparison]['alias'], comparison,
            '{:.3f}'.format(observed) if observed is not None else 'n/a',
            '{:+.1f}'.format(z) if z is not None and np.isfinite(z) else 'n/a')
        if values is not None and np.any(np.isfinite(values)):
            ax.hist(values[np.isfinite(values)], bins=30, density=True, histtype='stepfilled',
                    color=color, alpha=0.35, edgecolor=color, linewidth=0.9, label=label)
            drew = True
        if observed is not None:
            ax.axvline(observed, color=color, linewidth=1.8, label=None if drew else label)
            drew = True
    if drew:
        ax.set_xlabel(metric, fontsize=8)
        ax.set_ylabel('null density', fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='best')
    return drew


def _comparison_colors(comps):
    """Stable {comparison: colour} map so a comparison keeps its colour across the overlay figures."""
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('tab10')
    return {c: cmap(i % 10) for i, c in enumerate(comps)}


def _draw_comparison_overlay(session, comps, methods, loaded, save_dir, use_threshold_crossings,
                             save, joint_group='all'):
    """All comparisons together on one axis per method: observed headline vs its null + effect size.

    One panel per method (metrics differ across methods, so they cannot share a y-axis): the
    comparisons run along x, each drawn as an observed-similarity bar with its surrogate null
    overlaid as a black mean +/- sd marker and the effect size (z) annotated.  This is the
    'plot them together' view -- it answers, for each method, which behavioural description the
    neural space aligns with best.
    """
    import matplotlib.pyplot as plt

    colors = _comparison_colors(comps)
    n = len(methods)
    fig, axs = plt.subplots(nrows=1, ncols=n, squeeze=False, figsize=(1.0 + 4.6 * n, 4.8))
    axs = axs[0]
    for ax, method in zip(axs, methods):
        present = [(c, loaded[(c, method)]) for c in comps if (c, method) in loaded]
        if not present:
            ax.set_title(method.upper(), fontsize=11)
            ax.axis('off')
            continue
        higher = present[0][1].get('higher_is_more_similar', True)
        metric = present[0][1].get('metric', 'similarity')
        for k, (comparison, data) in enumerate(present):
            obs = data.get('similarity')
            ax.bar(k, obs if obs is not None else np.nan, width=0.62, color=colors[comparison],
                   label='observed' if k == 0 else None)
            null = data.get('null') or {}
            nm, nsd = null.get('null_mean'), null.get('null_std') or 0.0
            if nm is not None and np.isfinite(nm):
                ax.errorbar(k, nm, yerr=nsd, fmt='o', color='k', markersize=4, capsize=4,
                            label='null (mean$\\pm$sd)' if k == 0 else None)
            z = null.get('z')
            ax.annotate('z={:+.1f}'.format(z) if z is not None and np.isfinite(z) else 'z=n/a',
                        (k, obs if obs is not None else 0.0), textcoords='offset points',
                        xytext=(0, 3), ha='center', fontsize=7)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(['{} [{}]'.format(COMPARISONS[c]['alias'], c) for c, _ in present],
                           rotation=25, ha='right', fontsize=7)
        ax.axhline(0.0, color='0.6', linewidth=0.8, linestyle=':')
        ax.set_ylabel(metric + ('  (lower = more similar)' if not higher else ''), fontsize=8)
        ax.set_title(method.upper(), fontsize=11)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc='best')
    fig.suptitle('Neural <-> behaviour space similarity across comparisons -- {}{}'.format(
        session, ' (threshold crossings)' if use_threshold_crossings else ''))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if save:
        plotting.savefig(save_dir, 'alignment_comparisons{}{}'.format(
            _joint_group_suffix(joint_group), _suffix(use_threshold_crossings)), fig=fig)
        rs('Saved combined comparison figure for session {}.'.format(session))
    return fig


def _draw_rrr_overlay(session, comps, loaded, save_dir, use_threshold_crossings, save,
                      joint_group='all'):
    """RRR held-out R2 vs rank with every comparison overlaid on the same axes, per direction.

    Two panels (X->Y decode behaviour from neural; Y->X encode neural from behaviour); each
    comparison is one line, so the rank spectra are compared directly.  Returns None when no
    comparison has an RRR result.
    """
    import matplotlib.pyplot as plt

    present = [c for c in comps if (c, 'rrr') in loaded]
    if not present:
        return None
    colors = _comparison_colors(comps)
    fig, axs = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(13, 4.8))
    axs = axs[0]
    directions = (('cv_r2_curve_x_to_y', 'selected_rank_x_to_y', '-o',
                   'X->Y (decode behaviour from neural)'),
                  ('cv_r2_curve_y_to_x', 'selected_rank_y_to_x', '-s',
                   'Y->X (encode neural from behaviour)'))
    for ax, (curve_key, rank_key, style, title) in zip(axs, directions):
        for comparison in present:
            diag = loaded[(comparison, 'rrr')]['diagnostics']
            ax.plot(diag['ranks'], diag[curve_key], style, color=colors[comparison], markersize=3,
                    label='{} [{}]'.format(COMPARISONS[comparison]['alias'], comparison))
            ax.axvline(diag[rank_key], color=colors[comparison], linestyle='--', linewidth=0.6)
        ax.axhline(0.0, color='0.6', linewidth=0.8, linestyle=':')
        ax.set_xlabel('rank', fontsize=8)
        ax.set_ylabel('held-out R$^2$', fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=7)
    fig.suptitle('Reduced-rank regression: held-out R$^2$ vs rank across comparisons -- {}'.format(
        session))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    if save:
        plotting.savefig(save_dir, 'alignment_rrr_overlay{}{}'.format(
            _joint_group_suffix(joint_group), _suffix(use_threshold_crossings)), fig=fig)
        rs('Saved RRR rank overlay for session {}.'.format(session))
    return fig


def _draw_block_breakdown(session, comps, loaded, save_dir, use_threshold_crossings, save,
                          joint_group='all'):
    """Per-block variance breakdown for multi-block comparisons (RRR block R2 / CCA loadings)."""
    import matplotlib.pyplot as plt

    panels = []
    for comparison in comps:
        for method in ('rrr', 'cca'):
            data = loaded.get((comparison, method))
            signals = data.get('signals', []) if data else []
            if data is None or len(signals) < 2:
                continue
            diag = data['diagnostics']
            vals = (diag.get('block_r2_x_to_y') if method == 'rrr'
                    else diag.get('block_loadings_leading'))
            # a per-block vector shorter than the signal list means the blocks were mixed
            # (n_pcs_y reduced Y to PCs) -- the per-block breakdown is then not meaningful
            if not vals or len(vals) != len(signals):
                continue
            panels.append((comparison, method, signals, vals))
    if not panels:
        return None
    xn, yn = plotting.xy_numsubplots(len(panels))
    fig, axs = plt.subplots(nrows=yn, ncols=xn, squeeze=False, figsize=(1.0 + 4.2 * xn,
                                                                        1.0 + 3.5 * yn))
    axs = axs.flatten()
    for ax, (comparison, method, signals, vals) in zip(axs, panels):
        ylabel = ('per-block held-out R$^2$ (X->Y)' if method == 'rrr'
                  else 'leading CCA |loading| per block')
        ax.bar(range(len(signals)), vals, color='0.5', edgecolor='white')
        ax.set_xticks(range(len(signals)))
        ax.set_xticklabels(signals, rotation=20, ha='right', fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(0.0, color='0.6', linewidth=0.8, linestyle=':')
        ax.set_title('{} [{}] -- {}'.format(COMPARISONS[comparison]['alias'],
                                            COMPARISONS[comparison]['label'], method.upper()),
                     fontsize=8)
        ax.tick_params(labelsize=6)
    for ax in axs[len(panels):]:
        ax.axis('off')
    fig.suptitle('Does the neural space track kinematics, dynamics, or both? '
                 '(per-block breakdown) -- {}'.format(session))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        plotting.savefig(save_dir, 'alignment_blocks{}{}'.format(
            _joint_group_suffix(joint_group), _suffix(use_threshold_crossings)), fig=fig)
        rs('Saved per-block breakdown for session {}.'.format(session))
    return fig
