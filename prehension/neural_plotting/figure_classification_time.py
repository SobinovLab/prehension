#!python3
# -*- coding: utf-8 -*-
"""
Cross-validated classification of an object condition through time, pooled across sessions.

Built on the figure_spaces_dpca_pooled structure and session-handling.  Reuses the
figure_peth session/NWB machinery to window each successful trial's spikes around an
alignment timepoint, converts them to a firing-rate trace smoothed by a *causal*
half-gaussian filter and square-rooted, keeps the neurons whose mean rate exceeds
--min_rate Hz (as the dPCA script does), and then, at every time bin, trains a linear
discriminant (LDA) to predict the trial's condition (default targetForce(N)) with
k-fold cross-validation (default 5-fold, trials held out).

The classification is run:
  * per session (each session's own neurons) -> thin grey lines,
  * on a pseudo-population pooled across sessions (neurons concatenated across
    sessions, trials matched by condition) -> one thick black line.
A second session set (--sessions2) can be pooled and classified separately and drawn
on the same axes in a second colour for comparison; the two sets never share neurons
or trials.
Chance is estimated by label-shuffling: the label vector is permuted once at each
time bin, the resulting cross-validated accuracies are pooled across all time bins,
and their 90th percentile is taken as a single chance level per dataset (drawn as a
green dashed line for every session and for the pool).  The theoretical chance level
(1 / n_conditions) is drawn as a green solid line.

Each time bin is an independent classification problem, so the per-bin work (real
fit + shuffles) is farmed out to a multiprocessing.Pool; the number of processes is
selectable (see the CLI --processes).

scikit-learn provides the LDA and the cross-validation; it is imported lazily inside
the worker, so importing this module stays cheap and free of the dependency.

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
import multiprocessing
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage

from .. import meta_session
from ..tools.logs import rs, ws
from ..neural_processing import config as npconfig
from .figure_peth import (
    ALIGN_TIMEPOINT, GROUP_COLUMN, BEFORE, AFTER, BIN_WIDTH,
    read_nwb_spikes_and_ttl, load_timepoints_into_msession, get_trial_data_spike,
    get_timepoint, get_target_force, resolve_neuron_selection)
from .figure_spaces_pooled import MIN_RATE_HZ

# defaults; overridable through the calling function / script
CAUSAL_SIGMA = 0.05        # s, SD of the causal half-gaussian rate filter
N_FOLDS = 5                # cross-validation folds (trials held out)
SHUFFLE_PERCENTILE = 90.0  # percentile of the pooled shuffled accuracies -> chance level


# ---------------------------------------------------------------------------
# Causal rate filtering
# ---------------------------------------------------------------------------
def causal_halfgaussian_kernel(sigma_s, bin_width, n_sd=4):
    """Return a normalized causal half-gaussian smoothing kernel (in bins).

    The kernel spans the current bin and the preceding ~n_sd*sigma bins (no future
    bins), so filtering with it never uses activity from after a bin -- appropriate
    for a read-out that sweeps forward through time.  The weights are the right half
    of a Gaussian (index 0 = current bin) normalized to sum to 1.  A non-positive
    sigma degenerates to a pass-through kernel ([1.0]).
    """
    sigma_bins = float(sigma_s) / float(bin_width)
    if sigma_bins <= 0:
        return np.array([1.0])
    half_len = int(np.ceil(n_sd * sigma_bins))
    delays = np.arange(0, half_len + 1)
    kernel = np.exp(-0.5 * (delays / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def apply_causal_filter(rate, kernel):
    """Causally smooth a 1-D rate trace with `kernel` (index 0 weights the current bin).

    Output bin i is sum_d kernel[d] * rate[i - d] for d >= 0, i.e. only the present
    and past bins contribute; the trace length is preserved.
    """
    return np.convolve(rate, kernel, mode='full')[:len(rate)]


# ---------------------------------------------------------------------------
# Stage 1: per-trial activity pooled per session
# ---------------------------------------------------------------------------
def pool_trials(server, processed_server, sessions, align_key=ALIGN_TIMEPOINT,
                group_column=GROUP_COLUMN, before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                causal_sigma=CAUSAL_SIGMA, avg_window=None, only_good=False,
                min_rate=MIN_RATE_HZ):
    """Build the per-session, per-trial activity tensors used for classification.

    Mirrors figure_peth_pooled.pool_neurons' session-handling (probe type, skip_ttl,
    positional pulse<->trial pairing, successful-trial selection) but, instead of
    per-condition averages, keeps the single-trial activity.  For every kept neuron
    and successful trial, the spikes in [tp - before, tp + after] are binned, turned
    into a firing rate, smoothed with a causal half-gaussian (SD `causal_sigma`) and
    square-rooted.  When `avg_window` (s) is given, the activity is additionally
    averaged over a centred moving window of that width at each time bin.  Neurons are
    then kept if their mean firing rate over the window exceeds `min_rate` Hz (same
    activity criterion the dPCA script uses).

    Returns (sessions_data, bin_centers) where each entry of sessions_data is
    {'session': str, 'X': (n_trials, n_neurons, n_time) sqrt-rate activity,
    'labels': (n_trials,) condition per trial, 'neuron_labels': [unit id, ...]}.
    Sessions lacking neural data / meta / a matching pulse count / active neurons /
    a valid alignment timepoint are skipped with a warning.
    """
    found = ([s for s in sessions if os.path.isdir(os.path.join(server, s))]
             if sessions else meta_session.find_session_dirs(server))

    # shared bin grid + filters (identical for every session so tensors align)
    bins = np.arange(-before - bin_width / 2, after + bin_width / 2, bin_width)
    bin_centers = bins[:-1] + bin_width / 2
    numbins = len(bin_centers)
    freq = 1.0 / bin_width
    duration = float(before + after)
    kernel = causal_halfgaussian_kernel(causal_sigma, bin_width)
    win_bins = max(1, int(round(avg_window / bin_width))) if avg_window else None

    sessions_data = []
    for session in found:
        try:
            probe_type = npconfig.probe_type_from_meta(server, processed_server, session)
            cfg = npconfig.NeuralConfig(server, processed_server, session, probe_type)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        if only_good:
            neuron_ids = cfg.meta_neural.get('good_neurons') or []
            if not neuron_ids:
                ws("Skipping session {}: only_good set but 'good_neurons' is empty.".format(
                    session))
                continue
        else:
            neuron_ids = None

        skip_ttl = npconfig.resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl', 0)
        skip_ttl_last = npconfig.resolve_meta_arg(None, cfg.meta_neural, 'skip_ttl_last', 0)

        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                cfg.rserv, cfg.pserv)
            load_timepoints_into_msession(msession, mstruct)
            spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(cfg.nwb_path)
        except Exception as e:  # noqa: BLE001
            ws('Skipping session {}: {}'.format(session, e))
            continue

        # positional pulse<->trial offset: skip_ttl >0 drops leading pulses, <0 drops
        # leading trials; skip_ttl_last trims the end (pulses if >0, trials if <0).
        if skip_ttl and skip_ttl > 0:
            events_time = events_time[skip_ttl:]
        elif skip_ttl and skip_ttl < 0:
            msession = msession[-skip_ttl:]
        if skip_ttl_last and skip_ttl_last > 0:
            events_time = events_time[:-skip_ttl_last]
        elif skip_ttl_last and skip_ttl_last < 0:
            msession = msession[:skip_ttl_last]
        if len(events_time) != len(msession):
            ws('Skipping session {}: {} TTL pulses vs {} trials (after skip_ttl={}, '
               'skip_ttl_last={}); inspect with figure_ttl_alignment.'.format(
                   session, len(events_time), len(msession), skip_ttl, skip_ttl_last))
            continue

        session_spikes = get_trial_data_spike(spikes, events_time)
        num_neurons = len(unit_ids)
        for trial_spikes, trial_events in zip(session_spikes, events_time):
            ttl_start = trial_events[0]
            for i_n in range(num_neurons):
                trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ttl_start
        for i_trial, trial in zip(range(len(session_spikes)), msession):
            trial.spikes = session_spikes[i_trial]
        used_msession = [t for t in msession[:len(session_spikes)]
                         if t.success and hasattr(t, 'spikes')]

        try:
            neuron_selection, neuron_labels = resolve_neuron_selection(unit_ids, neuron_ids)
        except ValueError as e:
            ws('Skipping session {}: {}'.format(session, e))
            continue

        # successful trials with a valid alignment timepoint
        trials = [(t, get_timepoint(t, align_key)) for t in used_msession]
        trials = [(t, tp) for t, tp in trials if tp is not None]
        if not trials:
            ws("Skipping session {}: no trials with a valid '{}' timepoint.".format(
                session, align_key))
            continue
        used = [t for t, _ in trials]
        tps = [tp for _, tp in trials]
        labels = np.array([get_target_force(mobject, t.object_id, group_column) for t in used])

        # per-trial, per-neuron sqrt(causal-smoothed rate); mean rate kept for selection
        n_trials, n_sel = len(used), len(neuron_selection)
        activity = np.zeros((n_trials, n_sel, numbins))
        mean_rate = np.zeros(n_sel)
        for i_t, (trial, tp) in enumerate(zip(used, tps)):
            for j, sel in enumerate(neuron_selection):
                s = np.asarray(trial.spikes[sel])
                s = s[(s > tp - before) & (s < tp + after)] - tp
                counts, _ = np.histogram(s, bins=bins)
                mean_rate[j] += len(s) / duration
                rate = apply_causal_filter(counts * freq, kernel)
                activity[i_t, j, :] = np.sqrt(np.clip(rate, 0.0, None))
        mean_rate /= max(n_trials, 1)

        if win_bins:
            activity = scipy.ndimage.uniform_filter1d(
                activity, size=win_bins, axis=2, mode='nearest')

        keep = np.nonzero(mean_rate > min_rate)[0]
        rs('Session {}: activity filter kept {} / {} unit(s) with mean rate > {} Hz.'.format(
            session, len(keep), n_sel, min_rate))
        if keep.size == 0:
            ws('Skipping session {}: no unit exceeds the {} Hz activity threshold.'.format(
                session, min_rate))
            continue

        sessions_data.append({
            'session': session,
            'X': activity[:, keep, :],
            'labels': labels,
            'neuron_labels': [neuron_labels[j] for j in keep],
        })
        rs('Session {}: {} trials, {} neurons, conditions {}.'.format(
            session, n_trials, len(keep), sorted(set(labels.tolist()))))

    return sessions_data, bin_centers


def build_pooled_pseudopopulation(sessions_data, seed=0):
    """Concatenate neurons across sessions into one condition-matched pseudo-population.

    Because sessions do not share trials, a pseudo-population is built per condition:
    for every condition common to all sessions, the smallest per-session trial count
    n_c is used, n_c trials are drawn (without replacement, order shuffled independently
    per session so noise correlations are not fabricated) from each session, and the
    per-session neuron blocks are concatenated along the neuron axis.  This yields
    sum_c n_c pseudo-trials, each carrying every session's neurons.

    Returns (X_pooled (n_pseudo_trials, n_pooled_neurons, n_time), labels_pooled,
    conditions).  Raises ValueError if no condition is common to all sessions or if
    the common conditions have too few trials to classify.
    """
    if not sessions_data:
        raise ValueError('No sessions to pool.')

    common = set(np.unique(sessions_data[0]['labels']).tolist())
    for sd in sessions_data[1:]:
        common &= set(np.unique(sd['labels']).tolist())
    conditions = sorted(common)
    if len(conditions) < 2:
        raise ValueError('Fewer than 2 conditions common to all sessions; nothing to classify.')
    dropped = sorted(set().union(
        *[set(sd['labels'].tolist()) for sd in sessions_data]) - set(conditions))
    if dropped:
        ws('Dropping conditions not present in every session for pooling: {}.'.format(dropped))

    rng = np.random.RandomState(seed)
    blocks, labels_pooled = [], []
    for c in conditions:
        n_c = min(int(np.sum(sd['labels'] == c)) for sd in sessions_data)
        if n_c == 0:
            continue
        per_session = []
        for sd in sessions_data:
            idx = np.nonzero(sd['labels'] == c)[0]
            rng.shuffle(idx)
            per_session.append(sd['X'][idx[:n_c]])          # (n_c, n_neurons_s, n_time)
        blocks.append(np.concatenate(per_session, axis=1))   # (n_c, n_pooled_neurons, n_time)
        labels_pooled.extend([c] * n_c)

    if not blocks:
        raise ValueError('No condition has trials in every session; cannot pool.')
    return np.concatenate(blocks, axis=0), np.array(labels_pooled), conditions


# ---------------------------------------------------------------------------
# Stage 2: cross-validated classification through time (parallel over time bins)
# ---------------------------------------------------------------------------
def _cv_lda_accuracy(features, labels, n_folds, seed):
    """Mean k-fold cross-validated LDA accuracy for one (trials, neurons) feature set.

    Folds are stratified so every class appears in each split; the fold count is capped
    at the smallest class size.  Returns np.nan when there are fewer than 2 classes or
    fewer than 2 samples in the smallest class (classification undefined).
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2 or counts.min() < 2:
        return np.nan
    folds = int(min(n_folds, counts.min()))
    if folds < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(LinearDiscriminantAnalysis(), features, labels, cv=skf)
    return float(np.mean(scores))


def _classify_timepoint(task):
    """Worker: real and single-shuffle cross-validated accuracy for one time bin.

    task = (features (n_trials, n_neurons), labels, n_folds, seed).  The label vector
    is permuted once (a single shuffle) for the chance estimate.  Returns
    (accuracy, shuffle_accuracy).  Top-level (picklable) so it works with
    multiprocessing under spawn (Windows).
    """
    features, labels, n_folds, seed = task
    accuracy = _cv_lda_accuracy(features, labels, n_folds, seed)
    rng = np.random.RandomState(seed + 1)
    shuffle = _cv_lda_accuracy(
        features, labels[rng.permutation(len(labels))], n_folds, seed + 2)
    return accuracy, shuffle


def classify_through_time(X, labels, n_folds=N_FOLDS, processes=1, seed=0):
    """Classify the condition at every time bin, optionally over a process pool.

    X is (n_trials, n_neurons, n_time); each time bin is an independent classification
    (features = the neurons at that bin), evaluated on the real labels and once on a
    shuffled copy.  The per-bin work is dispatched to a multiprocessing.Pool of
    `processes` workers (serial when processes <= 1).  Each bin gets a distinct seed
    (seed + bin index) so folds/shuffle are reproducible and differ across bins.

    Returns (accuracy (n_time,), shuffle (n_time,)); empty neuron sets give all-nan
    results.  The single-shuffle accuracies are meant to be pooled across time and
    reduced to a chance level by the caller (see chance_level).
    """
    n_time = X.shape[2]
    if X.shape[1] == 0:
        return np.full(n_time, np.nan), np.full(n_time, np.nan)

    tasks = [(np.ascontiguousarray(X[:, :, t]), labels, n_folds, seed + t)
             for t in range(n_time)]
    if processes and processes > 1:
        with multiprocessing.Pool(processes=processes) as pool:
            results = pool.map(_classify_timepoint, tasks)
    else:
        results = [_classify_timepoint(task) for task in tasks]

    accuracy = np.array([r[0] for r in results])
    shuffle = np.array([r[1] for r in results])
    return accuracy, shuffle


def chance_level(shuffle, percentile=SHUFFLE_PERCENTILE):
    """Pool the per-time-bin shuffled accuracies and return their `percentile` (chance).

    A single shuffle is drawn per time bin (see classify_through_time); pooling those
    across all bins and taking the given percentile yields one scalar chance level for
    the dataset.  Returns np.nan when no finite shuffle accuracy is available.
    """
    shuffle = np.asarray(shuffle, dtype=float)
    if not np.any(np.isfinite(shuffle)):
        return np.nan
    return float(np.nanpercentile(shuffle, percentile))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_classification_time(bin_centers, groups, theoretical_chance, shuffle_percentile,
                             align_key, group_column, save_dir):
    """Plot classification accuracy through time for one or more session sets.

    Each group in `groups` (see figure_classification_time / GROUP_STYLES) is drawn in
    its own colour: individual sessions as thin accuracy lines each with its own dashed
    chance line (the `shuffle_percentile`th percentile of that session's pooled shuffled
    accuracies), and the pooled pseudo-population as a thick accuracy line with its own
    dashed chance line.  A single green solid line marks the theoretical chance level
    (1 / n_conditions).  When more than one group is plotted, legend entries are prefixed
    with the group label.  Saves classification_time_<align>.png into save_dir and
    returns the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    def _finite(v):
        return v is not None and np.isfinite(v)

    multi = len(groups) > 1
    for g in groups:
        prefix = '{}: '.format(g['label']) if multi else ''
        s_color, c_color, p_color = g['session_color'], g['chance_color'], g['pooled_color']

        for r in g['per_session_results']:
            ax.plot(bin_centers, r['accuracy'], color=s_color, linewidth=0.8, alpha=0.7)
            if _finite(r.get('chance')):
                ax.axhline(r['chance'], color=c_color, linewidth=0.6, linestyle='--', alpha=0.4)
        if g['per_session_results']:
            ax.plot([], [], color=s_color, linewidth=0.8, alpha=0.7,
                    label='{}individual sessions'.format(prefix))
            ax.plot([], [], color=c_color, linewidth=0.6, linestyle='--', alpha=0.4,
                    label='{}session chance (shuffle {:.0f}th pct)'.format(
                        prefix, shuffle_percentile))

        pooled_result = g['pooled_result']
        if pooled_result is not None:
            accuracy = pooled_result['accuracy']
            chance = pooled_result.get('chance')
            if _finite(chance):
                ax.axhline(chance, color=c_color, linewidth=1.3, linestyle='--',
                           label='{}pooled chance (shuffle {:.0f}th pct)'.format(
                               prefix, shuffle_percentile))
            ax.plot(bin_centers, accuracy, color=p_color, linewidth=2.5,
                    label='{}pooled ({} neurons)'.format(prefix, pooled_result['n_neurons']))

    if _finite(theoretical_chance):
        ax.axhline(theoretical_chance, color='g', linewidth=1.3, linestyle='-',
                   label='theoretical chance ({:.2f})'.format(theoretical_chance))

    ax.axvline(0.0, color='k', linewidth=0.8, linestyle=':')
    ax.set_xlabel('Time, s')
    ax.set_ylabel('Classification accuracy')
    ax.set_ylim(0.0, 1.0)
    ax.set_title('Cross-validated LDA classification of {} through time, aligned to {}'.format(
        group_column, align_key))
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, 'classification_time_{}.png'.format(align_key))
        fig.savefig(out, dpi=150, bbox_inches='tight')
        rs('Saved {}'.format(out))
    return fig


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
# style per plotted session set; extended if more sets are ever added
GROUP_STYLES = [
    {'label': 'set 1', 'pooled_color': 'k', 'session_color': '0.6', 'chance_color': 'g'},
    {'label': 'set 2', 'pooled_color': 'tab:red', 'session_color': 'lightcoral',
     'chance_color': 'tab:red'},
]


def _pool_and_classify(server, processed_server, sessions, align_timepoint, group_column,
                       before, after, bin_width, causal_sigma, avg_window, n_folds,
                       shuffle_percentile, only_good, min_rate, processes, seed):
    """Pool one set of sessions and classify it per session and as a pooled pseudo-population.

    Runs the full pipeline for a single session set: pool_trials -> per-session k-fold
    cross-validated LDA through time (each with its own shuffle-based chance level) ->
    condition-matched pooled pseudo-population classification.  Returns a group dict
    {'per_session_results', 'pooled_result', 'bin_centers', 'all_conditions'} (the last
    being the set of conditions seen, for the theoretical chance level), or None when the
    set has no usable neural data.
    """
    sessions_data, bin_centers = pool_trials(
        server, processed_server, sessions, align_key=align_timepoint,
        group_column=group_column, before=before, after=after, bin_width=bin_width,
        causal_sigma=causal_sigma, avg_window=avg_window, only_good=only_good,
        min_rate=min_rate)
    if not sessions_data:
        return None

    # per-session classification, each with its own shuffle-based chance level
    per_session_results = []
    for sd in sessions_data:
        accuracy, shuffle = classify_through_time(
            sd['X'], sd['labels'], n_folds=n_folds, processes=processes, seed=seed)
        chance = chance_level(shuffle, shuffle_percentile)
        per_session_results.append({
            'session': sd['session'], 'accuracy': accuracy, 'chance': chance,
            'n_neurons': sd['X'].shape[1]})
        rs('Session {}: classified {} time bins (peak accuracy {:.2f}, chance {:.2f}).'.format(
            sd['session'], len(accuracy),
            np.nanmax(accuracy) if accuracy.size else np.nan, chance))

    all_conditions = set().union(*[set(sd['labels'].tolist()) for sd in sessions_data])

    # pooled pseudo-population classification + its shuffle-based chance level
    pooled_result = None
    try:
        X_pooled, labels_pooled, conditions = build_pooled_pseudopopulation(
            sessions_data, seed=seed)
        accuracy, shuffle = classify_through_time(
            X_pooled, labels_pooled, n_folds=n_folds, processes=processes, seed=seed)
        chance = chance_level(shuffle, shuffle_percentile)
        pooled_result = {'accuracy': accuracy, 'chance': chance,
                         'conditions': conditions, 'n_neurons': X_pooled.shape[1]}
        rs('Pooled {} neurons across {} sessions, {} conditions, {} pseudo-trials '
           '(peak accuracy {:.2f}, chance {:.2f}).'.format(
               X_pooled.shape[1], len(sessions_data), len(conditions), len(labels_pooled),
               np.nanmax(accuracy) if accuracy.size else np.nan, chance))
    except ValueError as e:
        ws('No pooled pseudo-population: {}'.format(e))

    return {'per_session_results': per_session_results, 'pooled_result': pooled_result,
            'bin_centers': bin_centers, 'all_conditions': all_conditions}


def figure_classification_time(server, processed_server, sessions, sessions2=None,
                               align_timepoint=ALIGN_TIMEPOINT, group_column=GROUP_COLUMN,
                               before=BEFORE, after=AFTER, bin_width=BIN_WIDTH,
                               causal_sigma=CAUSAL_SIGMA, avg_window=None,
                               n_folds=N_FOLDS, shuffle_percentile=SHUFFLE_PERCENTILE,
                               only_good=False, min_rate=MIN_RATE_HZ, processes=1,
                               sessions_label=None, sessions2_label=None,
                               save_dir=None, seed=0):
    """Classify the condition (`group_column`) through time for one or two session sets.

    Pools per-trial, causally-smoothed, square-rooted activity (pool_trials), then runs
    k-fold cross-validated LDA at every time bin -- once per session and once on a
    condition-matched pseudo-population pooled across each set's sessions.  A chance level
    is estimated for every dataset by shuffling the labels once per bin, pooling those
    accuracies over time and taking their `shuffle_percentile`th percentile.

    When `sessions2` is given (a non-empty list), it is pooled and classified separately
    and drawn on the same axes in a second colour, for comparison; the two sets never
    share neurons or trials.  Plots each set (individual sessions thin with their chance
    lines, the pooled pseudo-population thick with its chance line) plus a single
    theoretical 1 / n_conditions chance line.

    `sessions` is a list (empty -> all sessions); per-session recording/skip_ttl/
    good_neurons come from each session's meta_neural.json.  `sessions_label` /
    `sessions2_label` set the legend label for each set (e.g. the raw --sessions /
    --sessions2 token string); default to the GROUP_STYLES labels when not given.
    `processes` sets the size of the per-time-bin process pool.  save_dir defaults to
    processed_server.  Returns the list of plotted group dicts.
    """
    def _run(sess):
        return _pool_and_classify(
            server, processed_server, sess, align_timepoint, group_column, before, after,
            bin_width, causal_sigma, avg_window, n_folds, shuffle_percentile, only_good,
            min_rate, processes, seed)

    group1 = _run(sessions)
    if group1 is None:
        raise ValueError('No sessions with usable neural data among {}.'.format(sessions))
    group1.update(GROUP_STYLES[0])
    if sessions_label:
        group1['label'] = sessions_label
    groups = [group1]

    if sessions2:
        group2 = _run(sessions2)
        if group2 is None:
            ws('No sessions with usable neural data among --sessions2 {}; plotting the '
               'first set only.'.format(sessions2))
        else:
            group2.update(GROUP_STYLES[1])
            if sessions2_label:
                group2['label'] = sessions2_label
            groups.append(group2)

    # the bin grid is identical across sets (same before/after/bin_width); theoretical
    # chance = 1 / n_conditions over the conditions seen across every plotted set.
    bin_centers = groups[0]['bin_centers']
    all_conditions = set().union(*[g['all_conditions'] for g in groups])
    theoretical_chance = 1.0 / len(all_conditions) if all_conditions else np.nan

    save_dir = processed_server if save_dir is None else save_dir
    plot_classification_time(bin_centers, groups, theoretical_chance, shuffle_percentile,
                             align_timepoint, group_column, save_dir)
    return groups
