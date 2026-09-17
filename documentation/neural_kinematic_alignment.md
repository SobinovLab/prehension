# Neural ↔ kinematic/dynamic space alignment

`prehension.analysis.neural_kinematic_alignment` quantifies how much structure the neural
population activity shares with several kinematic / dynamic descriptions of the same movement.
The numerical kernels live in `prehension.tools.space_similarity`; the per-session assembly,
file output and plotting live in the analysis module; the CLI is
`prehension_scripts/analysis/neural_kinematic_alignment.py`.

One entry point does all three methods:

```python
from prehension.analysis.neural_kinematic_alignment import compare_spaces
result = compare_spaces(X, Y, method="rrr")   # or "cca", "dsa"
```

* `X` — `(T, N)` neural activity (timepoints × channels: spike rates or LFP features)
* `Y` — `(T, M)` behavioural / biomechanical variables on the same `T` bins
* returns a dict with a **consistent schema** across methods: `similarity` (scalar headline),
  `diagnostics`, `fit` (fitted objects), `null` (a surrogate null distribution + effect size),
  `nulls` (every surrogate requested), plus `method`, `metric`, `higher_is_more_similar` and
  `params`.

The three named comparisons (thin configs in `COMPARISONS`, run per session and saved to
`space_similarity/`):

| key                           | alias | Y blocks                          |
|-------------------------------|-------|-----------------------------------|
| `joint_angles`                | A     | joint angles                      |
| `joint_angles_segment_forces` | B     | joint angles + per-segment forces |
| `joint_angles_torques`        | C     | joint angles + joint torques      |

For B and C, `Y` concatenates blocks with **different units and very different variances**
(joint angles in degrees vs forces in newtons vs torques in newton-metres). This is handled
explicitly (per-block standardization by default; optional per-block whitening; optional block
weighting `block_weights='equal'` → `1/sqrt(channels)` so a wide block does not swamp a narrow
one), and a **block-wise variance-explained breakdown** is reported (RRR per-block held-out R²,
CCA per-block loadings) so you can see whether the neural space tracks kinematics, dynamics, or
both. Block structure changes the interpretation of every headline number.

All behavioural signals are the TTL-aligned, per-trial signals used by the encoding/decoding
models (joint angles / velocity / torques on the right-arm independent DOFs; digit / segment
forces). No 3D endpoint or marker locations are used.

---

## Why the null is the actual result

All three metrics read **comfortably non-zero on unrelated data**: RRR R² and CCA correlations
are positive for any two smooth multivariate signals, and the DSA distance between two arbitrary
low-dimensional linear systems is finite and often small. The headline number is only
interpretable **relative to a null** that preserves the nuisance structure (autocorrelation,
spectra, trial structure) while destroying the genuine neural↔behaviour relationship. Every
method therefore returns a null distribution and an **effect size** (z against the null),
not just a p-value.

Surrogates (in `tools.space_similarity.SURROGATES`; each transforms `Y` with `X` fixed):

* **`time_shuffle`** — permute the time order of Y. Destroys everything (autocorrelation +
  alignment). A floor check, not a realistic null.
* **`circular_shift`** — roll Y in time. Preserves each channel's autocorrelation and Y's
  internal cross-correlations; destroys the alignment to X. **Recommended default for RRR/CCA.**
  With trial boundaries, shifts each trial independently.
* **`phase_randomize`** — randomize phases keeping each channel's power spectrum. Default
  `independent=True` (per-channel; the classic univariate surrogate — scrambles cross-structure
  and alignment). `independent=False` uses one shared phase rotation (Prichard & Theiler 1994),
  additionally preserving the cross-spectrum.
* **`trial_shuffle`** — permute whole trials of Y (breaks the trial-to-trial correspondence with
  X). The natural null for a trialized design; needs `trials` (per-trial row counts).

Per-method **primary** null (`DEFAULT_NULL`): `circular_shift` for RRR/CCA, `phase_randomize`
(independent) for DSA. Pass `null='all'` to compute all four, or a name / list of names.

> **DSA nulls are special.** DSA compares *dynamics*, and the operator is fit on the *pooled*
> transitions, so `circular_shift` (a time shift) and `trial_shuffle` (a reordering) leave it
> essentially unchanged — they are **uninformative** for DSA. Use `phase_randomize` (independent)
> or `time_shuffle`.

`n_null`: 200 for a stable z / percentile; ~1000 for a publishable p-value.

---

## Parameter recommendations

The defaults target `T ~ 1e3–1e4` bins (a session of trials at `1/fps`) and `N ~ 50–300` units.
"Matters a lot / barely matters" is relative to the headline number and its effect size.

### Shared preprocessing

* **Bin width** — the session kinematic/video period `1/fps` (as in encoding/decoding). *Matters
  a lot*: wider bins give fewer, less autocorrelated samples; narrower give more samples but make
  the result depend on the smoothing. This is the single biggest lever on every method.
* **`n_pcs` (reduce X to its top PCs before the comparison)** — *matters a lot for CCA/DSA* when
  `N` approaches `T`; 20–40 neural PCs usually keep the population structure while removing the
  directions that let CCA/DSA overfit. *Matters little for RRR* (the rank constraint already
  regularizes).
* **Block standardization** — on by default; without it a high-variance block dominates. Whiten
  and/or weight blocks when you want each block to contribute independently of its raw variance
  or channel count.
* **`--joint_group` (which joints enter the position / velocity / torque blocks)** — restricts the
  per-DOF blocks (joint angles and torques here) to a joint group, applied identically to all:
  `hand` (the default; the distal DOFs — wrist + thumb + fingers, `constants.DISTAL_DOFS`),
  `proximal` (shoulder + elbow, `constants.PROXIMAL_DOFS`), or `all` (every independent right-arm
  DOF, `constants.ALL_DOFS` — the previous behaviour). Forces are unaffected. Non-`all` groups are
  appended to the output-file and figure names, so groups never clobber. The same option exists on
  the encoding scripts (`encoding.py` / `encoding_lag.py` / `encoding_scatter.py` /
  `encoding_summary.py`), also defaulting to `hand`.
* **`n_pcs_y` (equalize the behavioural predictor count across comparisons)** — the comparisons
  have different behavioural widths (`joint_angles` is just the joint DOFs; `joint_angles_torques`
  doubles that; `joint_angles_segment_forces` adds the force channels), and more predictors can
  raise a headline for free. To compare them fairly, PCA-reduce the behavioural side to a common
  number of dimensions with `--n_pcs_y K`, choosing **K ≤ the smallest comparison's channel
  count** (the joint-angle DOF count) so all three enter with exactly `K` behavioural predictors.
  It mixes the blocks, so the per-block breakdown is disabled when it is set. This is the
  recommended knob when a comparison's advantage might just be extra dimensionality.

### RRR — reduced-rank regression

* **Rank range** — scan `1..min(N, M)`. The interpretable quantity is where held-out R² *plateaus*;
  report the whole curve, not just the argmax. **Rank barely matters past the plateau** (with
  ridge, extra ranks neither help nor hurt much).
* **Regularization `alpha`** — *matters a lot when `T` is small relative to `N`*. Start at
  `alpha=1.0` on standardized inputs; raise it (10–100) if even rank-1 held-out R² is negative
  (overfitting), lower it (1e-2) if full rank still underfits. Choose it by held-out R², never
  in-sample.
* **CV folds** — 5 is fine (**fold count barely matters**, 5 vs 10). Prefer splits that hold out
  *whole trials* when bins are strongly autocorrelated, or adjacent bins leak across folds and
  inflate R².
* **Both directions** — `X→Y` (predict behaviour from neural, decoding-like) and `Y→X` (predict
  neural from behaviour, encoding-like) are both reported, with the **asymmetry** =
  `R²(X→Y) − R²(Y→X)`. The asymmetry is informative: e.g. behaviour predicted from neural far
  better than the reverse means the neural space contains the behaviour plus much else.
* **`T` small relative to `N`** — raise `alpha`, cap the rank scan (e.g. `1..20`), and/or
  pre-reduce `X` with `n_pcs`. With `T < ~5N`, trust the rank-1/2 numbers and the null far more
  than full rank.

### CCA — canonical correlation analysis

* **Ridge (`alpha_x`, `alpha_y`)** — **the parameter that matters most.** Unregularized CCA with
  `N ~ T` reports canonical correlations near 1 that are pure overfitting. Use `alpha ~ 1e-1`–`1e1`
  on standardized data; the honest check is that the **cross-validated** correlations stay high,
  not the in-sample ones.
* **Number of components** — read it from the **cross-validated** correlations + `n_significant`
  (a within-fold permutation threshold), *not* the in-sample spectrum. Typically only the first
  1–3 components survive cross-validation for neural↔behaviour.
* **Overfitting failure mode when `N ~ T`** — in-sample correlations ≈ 1, cross-validated ≈ 0.
  Guard with ridge **and** `n_pcs` **and** the cross-validated numbers. If cv correlations
  collapse while in-sample is high, you do not have enough data.
* **Sample size** — as a rule of thumb `T >> N + M` (ideally `T > ~5–10 × (N + M)` after
  pre-reduction) for stable canonical correlations. Below that, ridge + `n_pcs` are mandatory and
  only the leading component is trustworthy.

### DSA — dynamical similarity analysis

* **Delay-embedding dimension `n_delays` and lag `delay`** — **the parameters DSA is most
  sensitive to.** `n_delays * delay * bin_width` should cover roughly one characteristic
  timescale of the dynamics (often ~10–50 bins for reach dynamics); too small misses the
  dynamics, too large inflates the operator and needs more data. **Sweep both and report the
  range** — never trust a single `(n_delays, delay)`.
* **Operator rank** — the HAVOK/DMD reduced rank; set it from the delay-embedded singular
  spectrum (enough modes for ~90–95% variance, often 5–20). *Moderate sensitivity*: too low
  collapses distinct dynamics together, too high fits noise.
* **Data per condition** — DSA needs enough consecutive samples to fit the operator (at least a
  few hundred bins per system, more with larger `n_delays`/rank). With short trials, pool trials
  (the operator is fit on within-trial transitions).
* **Optional packages** — the canonical metric from the [`DSA`](https://github.com/mitchellostrow/DSA)
  package is used when importable (`use_package='auto'`); otherwise an explicit
  delay-embed → DMD → Procrustes-over-vector-fields fallback runs. `PyDMD` (`pip install pydmd`)
  is used for the reduced operator when available (`use_pydmd='auto'`); the built-in numpy
  least-squares fit is exact, so the fallback is not an approximation. Passing `use_package=True`
  / `use_pydmd=True` raises a clear `ImportError` naming the pip install if the package is missing.

---

## Output files

Per session, per comparison × method, written to `<processed_server>/<session>/space_similarity/`
(same style as `encoding/`):

* `alignment_<comparison>_<method>[_tx].json` — human-readable: headline `similarity`,
  `diagnostics`, per-null summaries (`z`, `p_value`, `percentile`, `null_mean/std`), the block
  structure and the run `params`.
* `alignment_<comparison>_<method>[_tx].npz` — the heavy arrays for inspection: fitted operators
  / factors (`fit_*`), per-target R², loadings, and the raw null values (`null_<surrogate>`).

`_tx` marks the threshold-crossing neural source. When the files are present and `--overwrite`
was not given, the plotting reads straight from them (per session, into `prehension_plots/`):

* `alignment_comparisons` — all comparisons together on one axis per method: observed headline
  bar vs its null (mean ± sd) with the effect size (z) annotated.
* `alignment_null` — space similarity vs null, **one subplot per method with every comparison's
  null distribution overlaid** (filled histogram per comparison, matching observed line, z in the
  legend). This is the "all distributions on the same plot" view. **RRR gets three subplots** —
  the combined X↔Y (mean of both directions), X→Y (decode behaviour from neural), and Y→X (encode
  neural from behaviour) — each against its own null (the combined and reverse nulls are stored
  under `extra_nulls['symmetric']` / `extra_nulls['y_to_x']` in the JSON and
  `extra_null_symmetric_<surrogate>` / `extra_null_y_to_x_<surrogate>` in the .npz).
* `alignment_rrr_overlay` — RRR held-out R² vs rank with every comparison overlaid, one panel per
  direction (X→Y, Y→X).
* `alignment_blocks` — per-block variance breakdown (skipped when `--n_pcs_y` mixes the blocks).

## CLI examples

```bash
# all comparisons (default) and methods for one session, overlaid on the shared plots
py -3.11 prehension_scripts/analysis/neural_kinematic_alignment.py <preset> --sessions 2021_04_29

# all comparisons, but equalize the behavioural predictor count so the comparison is fair
# (K <= the joint-angle DOF count -> every comparison enters with 15 behavioural dimensions)
py -3.11 prehension_scripts/analysis/neural_kinematic_alignment.py <preset> --sessions 2021_04_29 \
    --comparison all --n_pcs_y 15

# proximal (shoulder + elbow) joints instead of the default hand (distal) joints
py -3.11 prehension_scripts/analysis/neural_kinematic_alignment.py <preset> --sessions 2021_04_29 \
    --joint_group proximal

# comparison B and C, CCA only, ridge-reduced neural space, all four nulls
py -3.11 prehension_scripts/analysis/neural_kinematic_alignment.py <preset> --sessions 2021_04_29 \
    --comparison B C --method cca --n_pcs 30 --alpha 1.0 --null all --n_null 1000

# DSA sweep of the delay-embedding dimension, block weighting on the concatenated Y
py -3.11 prehension_scripts/analysis/neural_kinematic_alignment.py <preset> --sessions 2021_04_29 \
    --method dsa --n_delays 20 --delay 1 --dsa_rank 10 --block_weights equal --overwrite
```
