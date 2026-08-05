#!python3
# -*- coding: utf-8 -*-
"""
Stats-related functions.

Copyright (C) 2025 Anton Sobinov
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
import numpy as np
import scipy.stats


def run_pca(X, n_components):
    '''Z-score each column of X and run PCA via SVD.

    Returns (scores, explained_variance_ratio) with scores shape (n_samples, k),
    k = min(n_components, rank).  Columns with zero variance are left unscaled.
    '''
    Xc = X.astype(float)
    mean = Xc.mean(axis=0)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xz = (Xc - mean) / std

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    k = int(min(n_components, S.size))
    scores = U[:, :k] * S[:k]
    total = np.sum(S ** 2)
    evr = (S[:k] ** 2) / total if total > 0 else np.zeros(k)
    return scores, evr


def format_p(p):
    '''Formats p-value according to recommendations with additional zero.

    Relevant papers:
    https://journals.physiology.org/doi/full/10.1152/ajpendo.00213.2004
    https://journals.physiology.org/doi/full/10.1152/advan.00022.2007
    https://journals.physiology.org/doi/full/10.1152/advan.00231.2023
    '''
    if p < 0.0001:
        ps = 'p < 0.0001'
    elif p < 0.001:
        ps = f'p = {round(p, 4):.4f}'
    elif p < 0.01:
        ps = f'p = {round(p, 3):.3f}'
    else:
        ps = f'p = {round(p, 2):.2f}'
    return ps


def data_stats(data, percentile=95, return_dic=False):
    '''Print various measurements of data. Expects an array.
    '''
    quantile_from = (1 - percentile / 100) / 2
    quantile_to = 1 - quantile_from

    # normal
    mean_val = np.mean(data)
    std_val = np.std(data)

    # non-normal
    low_q, q25, median_val, q75, high_q = np.quantile(
        data, [quantile_from, 0.25, 0.5, 0.75, quantile_to])
    iqr = q75 - q25

    if return_dic:
        d = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'confidence interval': [low_q, high_q],
            'quartiles': [q25, q75],
            'iqr': iqr
        }

        return (
            f"mean {mean_val:.2f}, median {median_val:.2f}, std {std_val:.2f}, "
            f"IQR [{q25:.2f}, {q75:.2f}], "
            f"{percentile} percentile [{low_q:.2f}, {high_q:.2f}]"
        ), d

    return (
        f"mean {mean_val:.2f}, median {median_val:.2f}, std {std_val:.2f}, "
        f"IQR [{q25:.2f}, {q75:.2f}], "
        f"{percentile} percentile [{low_q:.2f}, {high_q:.2f}]"
    )


def nonparam_mwu_rbc(pop1, pop2, alternative="two-sided", nan_policy="omit"):
    """
    Run a Mann-Whitney U test and compute rank-biserial correlation.

    It is a NON PAIRED comparison.

    Arguments:
        pop1, pop2 : array-like
            Two independent samples.
        alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
            Alternative hypothesis passed to scipy.stats.mannwhitneyu.
        nan_policy : {'omit', 'raise'}, default='omit'
            How to handle NaNs.

    Returns:
        dict -- Dictionary with:
            - 'u_statistic'
            - 'p_value'
            - 'rank_biserial'
            - 'n1'
            - 'n2'

    Notes:
    Rank-biserial correlation is computed as:
        r_rb = 1 - 2*U/(n1*n2)

    where U is the Mann-Whitney statistic for pop1.

    With this convention:
    - positive values mean pop1 tends to have larger values than pop2
    - negative values mean pop2 tends to have larger values than pop1
    """
    pop1 = np.asarray(pop1, dtype=float).ravel()
    pop2 = np.asarray(pop2, dtype=float).ravel()

    if nan_policy == "omit":
        pop1 = pop1[~np.isnan(pop1)]
        pop2 = pop2[~np.isnan(pop2)]
    elif nan_policy == "raise":
        if np.isnan(pop1).any() or np.isnan(pop2).any():
            raise ValueError("NaNs found in input.")
    else:
        raise ValueError("nan_policy must be 'omit' or 'raise'.")

    n1 = len(pop1)
    n2 = len(pop2)

    if n1 == 0 or n2 == 0:
        raise ValueError("Both samples must contain at least one valid value.")

    res = scipy.stats.mannwhitneyu(pop1, pop2, alternative=alternative)
    u = res.statistic
    p = res.pvalue

    rank_biserial = 1 - 2 * u / (n1 * n2)

    return {
        "u_statistic": u,
        "p_value": p,
        "rank_biserial": rank_biserial,
        "n1": n1,
        "n2": n2,
    }


def format_nonparam_mwu_rbc(*args, **kwargs):
    d = nonparam_mwu_rbc(*args, **kwargs)
    return (f'{format_p(d["p_value"])}, n1={d["n1"]}, n2={d["n2"]},'
            f' U={d["u_statistic"]:.2f}, r={d["rank_biserial"]:.2f}')


def nonparam_signedrank_rbc(
        pop1, pop2, alternative="two-sided",
        zero_method="wilcox", nan_policy="omit",
        correction=False):
    """
    Run a Wilcoxon signed-rank test and compute rank-biserial correlation.

    It is a PAIRED comparison.

    Arguments:
        pop1, pop2 : array-like
            Paired samples of equal length.
        alternative : {'two-sided', 'less', 'greater'}, default='two-sided'
            Alternative hypothesis passed to scipy.stats.wilcoxon.
        zero_method : {'wilcox', 'pratt', 'zsplit'}, default='wilcox'
            How to handle zero differences, matching scipy.stats.wilcoxon.
        nan_policy : {'omit', 'raise'}, default='omit'
            How to handle NaNs.
        correction : bool, default=False
            Whether to apply continuity correction in scipy.stats.wilcoxon.

    Returns:
        results : dict
            Dictionary with:
            - 'statistic'
            - 'p_value'
            - 'rank_biserial'
            - 'n'

    Notes:
    Rank-biserial correlation is computed from the signed ranks:
        r_rb = (W_pos - W_neg) / (W_pos + W_neg)

    where W_pos is the sum of ranks for positive differences and
    W_neg is the sum of ranks for negative differences.

    With this convention:
    - positive values mean pop1 tends to be larger than pop2
    - negative values mean pop2 tends to be larger than pop1
    """
    pop1 = np.asarray(pop1, dtype=float).ravel()
    pop2 = np.asarray(pop2, dtype=float).ravel()

    if pop1.shape != pop2.shape:
        raise ValueError("pop1 and pop2 must have the same shape.")

    if nan_policy == "omit":
        mask = ~(np.isnan(pop1) | np.isnan(pop2))
        pop1 = pop1[mask]
        pop2 = pop2[mask]
    elif nan_policy == "raise":
        if np.isnan(pop1).any() or np.isnan(pop2).any():
            raise ValueError("NaNs found in input.")
    else:
        raise ValueError("nan_policy must be 'omit' or 'raise'.")

    if len(pop1) == 0:
        raise ValueError("Inputs must contain at least one valid paired observation.")

    diff = pop1 - pop2

    # SciPy test result
    res = scipy.stats.wilcoxon(
        pop1, pop2,
        alternative=alternative,
        zero_method=zero_method,
        correction=correction
    )

    # Compute rank-biserial correlation manually
    abs_diff = np.abs(diff)

    if zero_method == "wilcox":
        keep = diff != 0
        diff_r = diff[keep]
        abs_r = abs_diff[keep]
        if len(diff_r) == 0:
            rbc = np.nan
        else:
            ranks = scipy.stats.rankdata(abs_r, method="average")
            w_pos = np.sum(ranks[diff_r > 0])
            w_neg = np.sum(ranks[diff_r < 0])
            rbc = (w_pos - w_neg) / (w_pos + w_neg)

    elif zero_method in {"pratt", "zsplit"}:
        ranks = scipy.stats.rankdata(abs_diff, method="average")
        pos = diff > 0
        neg = diff < 0
        zer = diff == 0

        w_pos = np.sum(ranks[pos])
        w_neg = np.sum(ranks[neg])

        if zero_method == "zsplit":
            w_zero = np.sum(ranks[zer])
            w_pos += 0.5 * w_zero
            w_neg += 0.5 * w_zero

        denom = w_pos + w_neg
        rbc = np.nan if denom == 0 else (w_pos - w_neg) / denom

    else:
        raise ValueError("zero_method must be 'wilcox', 'pratt', or 'zsplit'.")

    return {
        "statistic": res.statistic,
        "p_value": res.pvalue,
        "rank_biserial": rbc,
        "n": len(diff),
    }


def format_nonparam_signedrank_rbc(*args, **kwargs):
    d = nonparam_signedrank_rbc(*args, **kwargs)
    return (f'{format_p(d["p_value"])}, n={d["n"]},'
            f' Z={d["statistic"]:.2f}, r={d["rank_biserial"]:.2f}')
