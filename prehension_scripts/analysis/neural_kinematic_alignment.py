#!python3
# -*- coding: utf-8 -*-
"""
Quantify how much structure the neural population shares with kinematic / dynamic descriptions
of movement, per session, with reduced-rank regression, canonical correlation analysis and
dynamical similarity analysis.

For each session, comparison and method, assembles the neural firing-rate space and one or more
behavioural blocks on a shared bin grid and runs compare_spaces, saving a headline similarity,
diagnostics, the fitted objects and a surrogate null distribution (with an effect size) to the
session's space_similarity/ folder (a JSON summary + a companion .npz), in the same style as
encoding/.  The three comparisons:
  A (joint_angles)                 neural vs joint angles
  B (joint_angles_segment_forces)  neural vs joint angles + per-segment forces
  C (joint_angles_torques)         neural vs joint angles + joint torques
For B and C the behavioural blocks have different units / variances and are standardized (and,
optionally, whitened / weighted) per block.  Existing results are skipped unless --overwrite;
the figures are then drawn straight from the saved files (per session, into prehension_plots/).

Methods: rrr (cross-validated R2 vs rank, both directions + asymmetry, B = A @ C), cca
(cross-validated canonical correlations, significance, weights / loadings, ridge), dsa (delay
embedding -> HAVOK/DMD operator -> Procrustes-over-vector-fields distance; wraps the optional
DSA / PyDMD packages).  The neural source is the sorted neural.nwb, or the threshold crossings
with --threshold_crossings.  See the module docstring of
prehension.analysis.neural_kinematic_alignment (and documentation/neural_kinematic_alignment.md)
for the parameter recommendations.

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
import argparse
import datetime
import time

import matplotlib.pyplot as plt

from prehension import preset
from prehension.tools import cmd_args
from prehension.tools.space_similarity import SURROGATES
from prehension.analysis.neural_kinematic_alignment import (
    alignment_analysis, COMPARISONS, COMPARISON_ALIASES, METHODS, JOINT_GROUPS,
    DEFAULT_JOINT_GROUP)

if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    # accept 'all', the descriptive keys, and the A/B/C aliases in either case (case-insensitive)
    comparison_choices = (['all'] + list(COMPARISONS) + sorted(COMPARISON_ALIASES)
                          + [a.upper() for a in sorted(COMPARISON_ALIASES)])

    parser = argparse.ArgumentParser(
        description="Per-session neural <-> kinematic/dynamic space similarity with reduced-rank "
                    "regression, canonical correlation analysis and dynamical similarity "
                    "analysis; saved to each session's space_similarity/ folder with a surrogate "
                    "null, and plotted from there.")
    cmd_args.add_default_kwarguments(
        parser, {"server": current_preset["default_server"],
                 "processed_server": current_preset["processed_server"]})
    cmd_args.add_default_arguments(parser, ("sessions", "overwrite", "drift_correct"))
    parser.add_argument(
        "--comparisons", "--comparison", dest="comparisons", nargs="+", default=["all"],
        choices=comparison_choices, metavar="COMPARISON",
        help="Comparisons to run -- 'all' (default; runs every comparison and overlays them on a "
             "shared 'alignment_comparisons' plot), the keys ({}) or the aliases A/B/C.".format(
                 ' '.join(COMPARISONS)))
    parser.add_argument(
        "--methods", "--method", dest="methods", nargs="+", default=list(METHODS),
        choices=METHODS, metavar="METHOD",
        help="Methods to run. Default (all): {}.".format(' '.join(METHODS)))
    # null / statistics
    parser.add_argument(
        "--null", default="auto", metavar="SURROGATE",
        help="Null surrogate: 'auto' (per-method default: circular_shift for rrr/cca, "
             "phase_randomize for dsa), 'all', or one of {}.".format(sorted(SURROGATES)))
    parser.add_argument(
        "--n_null", type=int, default=200, metavar="N",
        help="Surrogate draws for the null distribution. Default: 200 (use ~1000 for a "
             "publishable p-value).")
    parser.add_argument(
        "--seed", type=int, default=None, metavar="S",
        help="Random seed for the cross-validation / surrogates. Default: None (random).")
    # block handling (comparisons B and C concatenate blocks with different units)
    parser.add_argument(
        "--no_block_standardize", dest="block_standardize", action="store_false",
        help="Do not z-score each behavioural block independently (standardization is ON by "
             "default so blocks with different units enter comparably).")
    parser.add_argument(
        "--block_whiten", action="store_true",
        help="ZCA-whiten each behavioural block (a block then contributes an isotropic "
             "unit-variance subspace). Default: off.")
    parser.add_argument(
        "--block_weights", default=None, metavar="MODE",
        help="Per-block weighting: 'equal' scales each block by 1/sqrt(channels) so a wide "
             "block does not swamp a narrow one. Default: none.")
    parser.add_argument(
        "--n_pcs", type=int, default=None, metavar="N",
        help="Reduce the neural space to its top N principal components before the comparison "
             "(recommended for cca / dsa when the channel count approaches the sample count). "
             "Default: off.")
    parser.add_argument(
        "--joint_group", "--joints", dest="joint_group", choices=sorted(JOINT_GROUPS),
        default=DEFAULT_JOINT_GROUP, metavar="GROUP",
        help="Restrict the joint-angle / torque blocks to a joint group: 'hand' (distal DOFs), "
             "'proximal' (shoulder + elbow) or 'all' (every independent right-arm DOF). Applied "
             "identically to positions, velocities and torques; forces are unaffected. Non-'all' "
             "groups are appended to the output file / figure names. Default: {}.".format(
                 DEFAULT_JOINT_GROUP))
    parser.add_argument(
        "--n_pcs_y", type=int, default=None, metavar="N",
        help="Reduce the BEHAVIOURAL space to its top N principal components before the "
             "comparison. Use this to make the number of behavioural predictors equal across "
             "comparisons (set N <= the smallest comparison's channel count -- i.e. the "
             "joint-angle DOF count -- so joint_angles, joint_angles_segment_forces and "
             "joint_angles_torques all enter with exactly N behavioural dimensions). Mixes the "
             "blocks, so the per-block breakdown is disabled. Default: off.")
    # method parameters
    parser.add_argument(
        "--alpha", type=float, default=1.0, metavar="A",
        help="RRR ridge penalty (also the CCA ridge for both views). Default: 1.0.")
    parser.add_argument(
        "--n_folds", type=int, default=5, metavar="K",
        help="Cross-validation folds for rrr / cca. Default: 5.")
    parser.add_argument(
        "--n_delays", type=int, default=10, metavar="D",
        help="DSA delay-embedding dimension. Default: 10 (sweep this -- DSA is sensitive to it).")
    parser.add_argument(
        "--delay", type=int, default=1, metavar="L",
        help="DSA delay-embedding lag in bins. Default: 1.")
    parser.add_argument(
        "--dsa_rank", type=int, default=10, metavar="R",
        help="DSA HAVOK/DMD operator rank. Default: 10.")
    parser.add_argument(
        "--bin_width", type=float, default=None, metavar="SECONDS",
        help="Firing-rate / behaviour bin width. Default: the session kinematic/video "
             "period 1/fps.")
    parser.add_argument(
        "--threshold_crossings", action="store_true",
        help="Use the threshold-crossing channels (neural_threshold_crossings.nwb) instead of "
             "the sorted units.")
    parser.add_argument(
        "--no_save", dest="save", action="store_false",
        help="Do not save the figures (saving is on by default, into each "
             "<session>/prehension_plots/). The space_similarity/ result files are always "
             "written.")

    args = parser.parse_args(args=argv)
    sessions = cmd_args.resolve_sessions(args.sessions, args.processed_server)

    method_kwargs = {
        'rrr': {'alpha': args.alpha, 'n_folds': args.n_folds},
        'cca': {'alpha_x': args.alpha, 'alpha_y': args.alpha, 'n_folds': args.n_folds},
        'dsa': {'n_delays': args.n_delays, 'delay': args.delay, 'rank': args.dsa_rank},
    }

    start_time = time.time()
    alignment_analysis(
        args.server, args.processed_server, sessions, comparisons=args.comparisons,
        methods=args.methods, overwrite=args.overwrite, save=args.save,
        bin_width=args.bin_width, use_threshold_crossings=args.threshold_crossings,
        drift_correct=args.drift_correct, block_standardize=args.block_standardize,
        block_whiten=args.block_whiten, block_weights=args.block_weights, n_pcs=args.n_pcs,
        n_pcs_y=args.n_pcs_y, null=args.null, n_null=args.n_null, seed=args.seed,
        method_kwargs=method_kwargs, joint_group=args.joint_group)
    print("Program took {}.".format(datetime.timedelta(seconds=time.time() - start_time)))

    plt.show()
