#!python3
# -*- coding: utf-8 -*-
"""
UMAP embeddings of single-trial state, per session, at three moments of the reach-to-grasp.

For a session, builds one point per trial from a chosen modality -- kinematics (joint-angle
position, restricted to the right-arm independent joint DOFs as in the encoding models),
segment forces, torques, or neural activity -- sampled at a single moment of the trial,
standardizes the features and embeds them in 2D with UMAP (umap-learn).  A separate
embedding is computed for each modality (all by default; restrict with `modalities`) and each
of three timepoints: the beginning of the movement (the earliest limb-movement onset), the
initial grasp (first_grasp_start) and the middle of the grasp (halfway from first_grasp_start
to release).  All timepoints are read per trial from timepoints.csv (create_timepoints).

The base figure colours each point by the trial's target force (the repo's yellow->red
'autumn_r' force colormap, tools.plotting.cmap_norm) and shapes its marker by the trial's
kinematic condition -- the object's geometry (the non-force object-definition columns).  In
addition, a purely force-coloured figure (one marker, same force colormap) and one figure per
object geometry parameter that varies across the used trials (e.g. aperture, tilt) recolour the
same embeddings -- the parameter figures by that parameter's value (PARAMETER_COLORMAPS), all
points one marker.  Every figure is a grid of modalities (rows) x timepoints (columns) saved to
the session's prehension_plots/ folder.

Behavioural features are the signal channels interpolated at the timepoint; the neural feature
is each unit's spike rate in a short window centred on the timepoint (with the optional linear
session-drift correction shared with the other poolers).  UMAP is per session because the
neural feature dimension (the sorted units) is session-specific.

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
import re

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .. import meta_session
from ..tools import io, plotting
from ..tools.cmd_args import resolve_meta_arg
from ..tools.logs import rs, ws
from ..neural_processing.common.spikes import (
    GROUP_COLUMN, read_nwb_spikes_and_ttl, get_trial_data_spike, fit_session_drift, drift_offset)
from ..neural_plotting.common.pooling import session_neural_context
from ..neural_plotting.common.behaviour import (
    load_timepoints_into_msession, get_timepoint, get_target_force)
from .behavior_pooling import SIGNAL_SPECS
from .encoding import _filter_ra_dofs

# Modalities the embedding supports.  Each behavioural one maps to one or more
# behavior_pooling.SIGNAL_SPECS signals whose channels are concatenated (kinematics ==
# joint-angle position; kinematics_torques == joint angles + torques joined); kinematic and
# torque channels are restricted to the right-arm independent joint DOFs as in the encoding
# models (_filter_ra_dofs); 'neural' is the binned spike rate.
MODALITIES = ('kinematics', 'segment_forces', 'torques', 'kinematics_torques', 'neural')
_BEHAVIOUR_SIGNALS = {
    'kinematics': ('joint_angles',),
    'segment_forces': ('segment_forces',),
    'torques': ('torques',),
    'kinematics_torques': ('joint_angles', 'torques'),
}

# The three moments each embedding is computed at, read per trial from timepoints.csv.
TIMEPOINTS = ('movement_onset', 'initial_grasp', 'mid_grasp')
_TIMEPOINT_LABELS = {
    'movement_onset': 'movement onset',
    'initial_grasp': 'initial grasp',
    'mid_grasp': 'mid grasp',
}
# 'movement onset' is the earliest available limb-movement onset (the reach begins with
# whichever joint moves first); these are the movement-onset columns, proximal to distal.
_MOVEMENT_ONSET_KEYS = ('shoulder_onset', 'elbow_onset', 'wrist_onset', 'finger_onset')

# Marker shapes cycled over the distinct kinematic conditions (object geometries).
MARKERS = ('o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>', 'p', 'h')

# Custom colour gradients (value low -> high) for the per-parameter figures, matched to an
# object-definition column by substring.  Any other varying parameter falls back to viridis.
PARAMETER_COLORMAPS = {
    'aperture': ('#74C275', '#1c6533'),   # narrow -> large
    'tilt': ('#bbe3f2', '#2c2f85'),       # negative -> positive
}

NEURAL_WINDOW_S = 0.2      # window (s) for the single-timepoint neural spike-count rate
UMAP_N_NEIGHBORS = 15      # umap-learn default; capped at n_trials - 1 for small sessions
UMAP_MIN_DIST = 0.1        # umap-learn default
MIN_TRIALS = 5             # skip an embedding with fewer usable trials than this


def _object_def(mobject, object_id):
    """The object-definition dict for a trial's object (str/int key fallback, as elsewhere)."""
    key = object_id
    if key not in mobject and str(key) in mobject:
        key = str(key)
    return mobject[key]['def']


def _condition_label(mobject, object_id, group_column):
    """Kinematic-condition label for a trial: the object's non-force geometry columns.

    The force axis is shown by colour, so the marker shape encodes only the object's
    geometry -- every object-definition column except the force group column and any other
    force column (e.g. targetForceRelRange*).  Returns 'all' when no geometry column exists.
    """
    d = _object_def(mobject, object_id)
    parts = ['{}={}'.format(k, v) for k, v in d.items()
             if k != group_column and 'force' not in k.lower()]
    return ', '.join(parts) if parts else 'all'


def _timepoint_time(trial, timepoint):
    """Time (s since TTL) of `timepoint` for a trial, or None when it is undefined.

    movement_onset -> the earliest available limb-movement onset (MOVEMENT_ONSET_KEYS);
    initial_grasp  -> first_grasp_start; mid_grasp -> halfway from first_grasp_start to
    release.  All read via get_timepoint (load_timepoints_into_msession must have run).
    """
    if timepoint == 'movement_onset':
        onsets = [t for t in (get_timepoint(trial, k) for k in _MOVEMENT_ONSET_KEYS)
                  if t is not None]
        return min(onsets) if onsets else None
    if timepoint == 'initial_grasp':
        return get_timepoint(trial, 'first_grasp_start')
    if timepoint == 'mid_grasp':
        start = get_timepoint(trial, 'first_grasp_start')
        end = get_timepoint(trial, 'release')
        if start is None or end is None or end <= start:
            return None
        return 0.5 * (start + end)
    raise ValueError('Unknown timepoint {!r} (expected one of {}).'.format(timepoint, TIMEPOINTS))


def _behaviour_point_features(msession, signals, timepoint):
    """(features, object_ids) for one or more behavioural signals sampled at `timepoint`.

    `signals` is a tuple of behavior_pooling.SIGNAL_SPECS keys whose channels are concatenated
    (e.g. joint angles + torques for the joined modality).  One row per successful trial that has
    every signal's file and a defined `timepoint`: each channel of each signal CSV linearly
    interpolated at that time (the CSV times are seconds since the TTL pulse, the same frame as
    the timepoint).  Kinematic and torque signals are restricted to the right-arm independent
    joint DOFs, the same restriction as the encoding models (encoding._filter_ra_dofs).  features
    is (n_trials, n_channels); object_ids is the per-row trial object id, from which the caller
    derives the force and object-parameter colouring.
    """
    specs = [(s,) + SIGNAL_SPECS[s] for s in signals]   # (signal, attr, exists)
    feats, object_ids = [], []
    for trial in msession:
        if not trial.success or not all(getattr(trial, exists)() for _, _, exists in specs):
            continue
        tp = _timepoint_time(trial, timepoint)
        if tp is None:
            continue
        row, ok = [], True
        for signal, attr, _ in specs:
            try:
                times, names, values = io.import_timed_csv(getattr(trial, attr))
            except Exception as e:  # noqa: BLE001
                ws('Trial {}: could not read {} ({}); skipping.'.format(
                    trial.trial_number, signal, e))
                ok = False
                break
            times = np.asarray(times, dtype=float)
            if times.size < 2:
                ok = False
                break
            _, values = _filter_ra_dofs(signal, names, np.asarray(values, dtype=float))
            row.extend(float(np.interp(tp, times, v)) for v in values)
        if not ok:
            continue
        feats.append(row)
        object_ids.append(trial.object_id)
    return np.asarray(feats, dtype=float), object_ids


def _neural_point_features(neural_trials, timepoint, window, slopes, t_ref):
    """(features, object_ids) for the neural rate in a window at `timepoint`.

    One row per successful trial (with spikes attached, see _load_session_spikes) that has a
    defined `timepoint`: each unit's spike count in [tp - window/2, tp + window/2] divided by
    the window (Hz), with the per-unit linear session drift subtracted (drift_offset returns 0
    when drift correction is off).  features is (n_trials, n_units); object_ids the per-row
    trial object id.
    """
    half = window / 2.0
    feats, object_ids = [], []
    for trial in neural_trials:
        if not trial.success:
            continue
        tp = _timepoint_time(trial, timepoint)
        if tp is None:
            continue
        rates = []
        for i_u, spikes in enumerate(trial.spikes):
            spikes = np.asarray(spikes)
            count = int(np.count_nonzero((spikes >= tp - half) & (spikes < tp + half)))
            rates.append(count / window - drift_offset(slopes, t_ref, i_u, trial.ttl_start))
        feats.append(rates)
        object_ids.append(trial.object_id)
    return np.asarray(feats, dtype=float), object_ids


def _load_session_spikes(server, processed_server, session, msession, use_threshold_crossings,
                         drift_correct):
    """Attach per-trial, TTL-zeroed spikes to a session's trials for the neural modality.

    Reads the neural source (session_neural_context), pairs TTL pulses to trials positionally
    (meta_neural skip_ttl / skip_ttl_last, as in neural_plotting.common.pooling) and zeroes
    each trial's spikes to its TTL start.  Returns (trials, slopes, t_ref): trials the paired
    trials (each with .spikes and .ttl_start), and the linear session-drift fit (slopes None
    when drift_correct is False).  Returns (None, None, 0.0) when the neural source / pulse
    pairing is unusable, so the caller drops the neural modality for the session.  Slicing acts
    on a local list, so the shared msession the behavioural modalities use is left intact.
    """
    try:
        nwb_path, meta_neural, _, _ = session_neural_context(
            server, processed_server, session, use_threshold_crossings)
        spikes, unit_ids, events_time = read_nwb_spikes_and_ttl(nwb_path)
    except Exception as e:  # noqa: BLE001
        ws('No neural data for session {}: {}'.format(session, e))
        return None, None, 0.0

    skip_ttl = resolve_meta_arg(None, meta_neural, 'skip_ttl', 0)
    skip_ttl_last = resolve_meta_arg(None, meta_neural, 'skip_ttl_last', 0)
    trials = msession
    if skip_ttl and skip_ttl > 0:
        events_time = events_time[skip_ttl:]
    elif skip_ttl and skip_ttl < 0:
        trials = trials[-skip_ttl:]
    if skip_ttl_last and skip_ttl_last > 0:
        events_time = events_time[:-skip_ttl_last]
    elif skip_ttl_last and skip_ttl_last < 0:
        trials = trials[:skip_ttl_last]
    if len(events_time) != len(trials):
        ws('No neural for session {}: {} TTL pulses vs {} trials (skip_ttl={}, '
           'skip_ttl_last={}).'.format(session, len(events_time), len(trials), skip_ttl,
                                       skip_ttl_last))
        return None, None, 0.0

    session_spikes = get_trial_data_spike(spikes, events_time)
    n_units = len(unit_ids)
    for trial_spikes, ev in zip(session_spikes, events_time):
        for i_n in range(n_units):
            trial_spikes[i_n] = np.asarray(trial_spikes[i_n]) - ev[0]
    paired = []
    for trial, tspk, ev in zip(trials, session_spikes, events_time):
        trial.spikes = tspk
        trial.ttl_start = float(ev[0])
        paired.append(trial)

    slopes, t_ref = fit_session_drift(spikes, events_time) if drift_correct else (None, 0.0)
    return paired, slopes, t_ref


def _embed(features, n_neighbors, min_dist):
    """2D UMAP embedding of standardized `features` (n_trials, n_features) via umap-learn.

    Columns are z-scored first (StandardScaler) so channels on different scales -- joint
    angles, forces, torques, firing rates -- contribute comparably.  n_neighbors is capped at
    n_trials - 1 so small sessions still embed.  The UMAP random_state is left unset (the
    embedding is not seeded), so umap-learn runs multi-threaded and each run may differ.
    """
    import umap
    from sklearn.preprocessing import StandardScaler

    n_trials = features.shape[0]
    n_neighbors = int(min(n_neighbors, max(2, n_trials - 1)))
    scaled = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    return reducer.fit_transform(scaled)


def _modality_features(modality, timepoint, msession, neural_trials, slopes, t_ref, window):
    """Dispatch to the behavioural or neural extractor: (features, object_ids)."""
    if modality == 'neural':
        if neural_trials is None:
            return np.empty((0, 0)), []
        return _neural_point_features(neural_trials, timepoint, window, slopes, t_ref)
    return _behaviour_point_features(msession, _BEHAVIOUR_SIGNALS[modality], timepoint)


def _geometry_columns(mobject, group_column):
    """Object-definition columns describing geometry (every non-force column, in order)."""
    for obj in mobject.values():
        return [k for k in obj['def'] if k != group_column and 'force' not in k.lower()]
    return []


def _parameter_colormap(column):
    """Colormap for colouring by object parameter `column` (value low -> high).

    Aperture and tilt use the project's custom gradients (PARAMETER_COLORMAPS); any other
    parameter uses viridis.  Matched by substring so 'pos_aperture(mm)' / 'pos_tilt(deg)' resolve.
    """
    for key, (c_lo, c_hi) in PARAMETER_COLORMAPS.items():
        if key in column.lower():
            return mpl.colors.LinearSegmentedColormap.from_list(key, [c_lo, c_hi])
    return plt.get_cmap('viridis')


def _safe_name(text):
    """A filesystem-safe token from an object-parameter column name (for the output filename)."""
    return re.sub(r'[^0-9A-Za-z]+', '_', text).strip('_')


def _new_grid(n_rows, n_cols):
    """A modalities x timepoints subplot grid sized for the UMAP panels."""
    return plt.subplots(n_rows, n_cols, squeeze=False,
                        figsize=(1.5 + 4.0 * n_cols, 1.5 + 3.6 * n_rows))


def _blank_panel(ax, modality, timepoint):
    """Turn a skipped panel's axis off with an 'insufficient data' note."""
    ax.axis('off')
    ax.set_title('{} @ {}\n(insufficient data)'.format(
        modality, _TIMEPOINT_LABELS[timepoint]), fontsize=8)


def _label_panel(ax, modality, timepoint, n_trials):
    """Blank the ticks and set the axis labels / title of a drawn UMAP panel."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('UMAP-1', fontsize=7)
    ax.set_ylabel('UMAP-2', fontsize=7)
    ax.set_title('{} @ {} (n={})'.format(
        modality, _TIMEPOINT_LABELS[timepoint], n_trials), fontsize=8)


def _finish_figure(fig, title, bottom):
    """Add the suptitle and lay out, reserving `bottom` fraction (e.g. for a legend)."""
    fig.suptitle(title)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))


def _draw_condition_panel(ax, embedding, forces, conditions, cond_marker, cmap, norm):
    """Scatter one embedding coloured by force, marker shape by kinematic condition."""
    conditions = np.asarray(conditions, dtype=object)
    for cond, marker in cond_marker.items():
        idx = np.nonzero(conditions == cond)[0]
        if idx.size == 0:
            continue
        ax.scatter(embedding[idx, 0], embedding[idx, 1], c=forces[idx], cmap=cmap, norm=norm,
                   marker=marker, s=28, edgecolor='k', linewidths=0.3, alpha=0.9)


def _draw_force_figure(session, mods, tps, panels, mobject, group_column, save_dir, tx, save):
    """UMAP grid coloured by target force, marker-shaped by kinematic condition (with a legend).

    `panels` maps (modality, timepoint) -> (embedding, object_ids) or None; the force and
    condition of each point are derived from its object id, on a force scale and condition ->
    marker map shared across the whole figure.
    """
    drawn, all_forces, all_conditions = {}, [], set()
    for key, panel in panels.items():
        if panel is None:
            drawn[key] = None
            continue
        embedding, object_ids = panel
        forces = np.array([get_target_force(mobject, o, group_column) for o in object_ids],
                          dtype=float)
        conditions = [_condition_label(mobject, o, group_column) for o in object_ids]
        drawn[key] = (embedding, forces, conditions)
        all_forces.extend(forces.tolist())
        all_conditions.update(conditions)

    max_force = max(all_forces) if all_forces else 1.0
    cond_order = sorted(all_conditions)
    cond_marker = {c: MARKERS[i % len(MARKERS)] for i, c in enumerate(cond_order)}
    cmap, norm, _ = plotting.cmap_norm([], max_force)   # repo's yellow->red force colormap

    fig, axs = _new_grid(len(mods), len(tps))
    for i_m, modality in enumerate(mods):
        for i_tp, timepoint in enumerate(tps):
            ax = axs[i_m][i_tp]
            panel = drawn[(modality, timepoint)]
            if panel is None:
                _blank_panel(ax, modality, timepoint)
                continue
            embedding, forces, conditions = panel
            _draw_condition_panel(ax, embedding, forces, conditions, cond_marker, cmap, norm)
            _label_panel(ax, modality, timepoint, embedding.shape[0])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01).set_label(group_column)
    handles = [Line2D([0], [0], marker=cond_marker[c], color='0.4', linestyle='none',
                      markersize=6, markerfacecolor='0.7', markeredgecolor='k')
               for c in cond_order]
    fig.legend(handles, cond_order, title='kinematic condition', loc='lower center',
               ncol=min(max(len(cond_order), 1), 4), fontsize=6, title_fontsize=7)
    _finish_figure(fig, 'UMAP of single-trial state -- {}{}'.format(
        session, ' (threshold crossings)' if tx else ''), bottom=0.08)
    if save:
        name = 'figure_umap{}{}'.format('_' + mods[0] if len(mods) == 1 else '', tx)
        plotting.savefig(save_dir, name, fig=fig)
        rs('Saved UMAP force figure for session {}.'.format(session))
    return fig


def _panel_values(panels, value_fn):
    """Per-panel scalar values for a colouring: (drawn, all_values).

    drawn maps (modality, timepoint) -> (embedding, values) or None; all_values is the flat list
    over every point, for building a shared colour scale.  value_fn maps a trial object id to a
    float (the force or an object parameter).
    """
    drawn, all_values = {}, []
    for key, panel in panels.items():
        if panel is None:
            drawn[key] = None
            continue
        embedding, object_ids = panel
        values = np.array([value_fn(o) for o in object_ids], dtype=float)
        drawn[key] = (embedding, values)
        all_values.extend(values.tolist())
    return drawn, all_values


def _draw_value_figure(session, mods, tps, drawn, cmap, norm, label, name, save_dir, tx, save):
    """UMAP grid with one marker, coloured by a scalar (colorbar, no legend); shared scale.

    `drawn` is a _panel_values map of (embedding, values) per panel.  Shared by the purely
    force-coloured figure and the per-object-parameter figures -- only cmap / norm / label /
    name differ.
    """
    fig, axs = _new_grid(len(mods), len(tps))
    for i_m, modality in enumerate(mods):
        for i_tp, timepoint in enumerate(tps):
            ax = axs[i_m][i_tp]
            panel = drawn[(modality, timepoint)]
            if panel is None:
                _blank_panel(ax, modality, timepoint)
                continue
            embedding, values = panel
            ax.scatter(embedding[:, 0], embedding[:, 1], c=values, cmap=cmap, norm=norm,
                       marker='o', s=28, edgecolor='k', linewidths=0.3, alpha=0.9)
            _label_panel(ax, modality, timepoint, embedding.shape[0])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=axs.ravel().tolist(), fraction=0.02, pad=0.01).set_label(label)
    _finish_figure(fig, 'UMAP of single-trial state by {} -- {}{}'.format(
        label, session, ' (threshold crossings)' if tx else ''), bottom=0.04)
    if save:
        plotting.savefig(save_dir, name, fig=fig)
        rs('Saved UMAP {} figure for session {}.'.format(label, session))
    return fig


def _draw_force_only_figure(session, mods, tps, panels, mobject, group_column, save_dir, tx, save):
    """UMAP grid coloured purely by target force (one marker, the repo's yellow->red colormap).

    The force-coloured counterpart of the parameter figures: same embeddings as the main force
    figure but a single marker and no condition legend, so only the force gradient is read.
    """
    drawn, all_values = _panel_values(
        panels, lambda o: get_target_force(mobject, o, group_column))
    max_force = max(all_values) if all_values else 1.0
    cmap, norm, _ = plotting.cmap_norm([], max_force)   # repo's yellow->red force colormap
    name = 'figure_umap_force{}{}'.format('_' + mods[0] if len(mods) == 1 else '', tx)
    return _draw_value_figure(session, mods, tps, drawn, cmap, norm, group_column, name,
                              save_dir, tx, save)


def _draw_parameter_figure(session, mods, tps, panels, mobject, column, save_dir, tx, save):
    """UMAP grid coloured by one object parameter `column` (single marker, shared value scale).

    Reuses the same embeddings as the force figure; each point is coloured by its object's value
    of `column` (_parameter_colormap), so structure that tracks that parameter stands out.
    """
    drawn, all_values = _panel_values(
        panels, lambda o: float(_object_def(mobject, o)[column]))
    lo, hi = (min(all_values), max(all_values)) if all_values else (0.0, 1.0)
    norm = mpl.colors.Normalize(vmin=lo, vmax=hi if hi > lo else lo + 1.0)
    name = 'figure_umap_{}{}{}'.format(
        _safe_name(column), '_' + mods[0] if len(mods) == 1 else '', tx)
    return _draw_value_figure(session, mods, tps, drawn, _parameter_colormap(column), norm,
                              column, name, save_dir, tx, save)


def figure_umap(server, processed_server, sessions, modalities=MODALITIES, timepoints=TIMEPOINTS,
                group_column=GROUP_COLUMN, window=NEURAL_WINDOW_S, n_neighbors=UMAP_N_NEIGHBORS,
                min_dist=UMAP_MIN_DIST, use_threshold_crossings=False, drift_correct=True,
                min_trials=MIN_TRIALS, save=True):
    """Compute and plot per-session UMAP embeddings for each modality and timepoint.

    For every session (empty `sessions` -> all under processed_server) and every requested
    `modality` (kinematics / segment_forces / torques / neural) and `timepoint`
    (movement_onset / initial_grasp / mid_grasp), builds one point per trial (features sampled
    at the timepoint), standardizes and embeds them in 2D with UMAP, on a grid of modalities
    (rows) x timepoints (columns).  The embeddings are computed once and drawn in several
    figures per session that differ only in the point colouring:

      * a force figure -- points coloured by target force (`group_column`, autumn_r colormap)
        and marker-shaped by kinematic condition (object geometry), with a condition legend;
      * a purely force-coloured figure -- the same force colormap but a single marker and no
        condition legend, so only the force gradient is read;
      * one figure per object geometry parameter that varies across the used trials (e.g.
        aperture, tilt) -- all points one marker, coloured by that parameter's value with its
        gradient (PARAMETER_COLORMAPS / _parameter_colormap).

    `window` is the neural spike-count window (s); `n_neighbors` / `min_dist` are UMAP
    parameters (the embedding is not seeded).  Panels with fewer than `min_trials` usable trials
    are left empty with a note.  Figures are saved to <session>/prehension_plots/ (unless save is
    False).  Returns the list of figures.
    """
    found = sessions if sessions else meta_session.find_session_dirs(processed_server)
    mods = [m for m in MODALITIES if m in modalities]
    tps = [t for t in TIMEPOINTS if t in timepoints]

    figures = []
    for session in found:
        try:
            mstruct, _, mobject, msession = meta_session.load_meta_information(
                os.path.join(server, session), os.path.join(processed_server, session))
            load_timepoints_into_msession(msession, mstruct)
        except Exception as e:  # noqa: BLE001
            ws('Skipping session {}: {}'.format(session, e))
            continue

        # load the neural source once per session (only if a neural embedding was requested)
        neural_trials, slopes, t_ref = None, None, 0.0
        if 'neural' in mods:
            neural_trials, slopes, t_ref = _load_session_spikes(
                server, processed_server, session, msession, use_threshold_crossings,
                drift_correct)

        # embed every (modality, timepoint) once; every figure reuses these embeddings, so only
        # the colouring differs.  Each panel keeps its per-trial object ids for the colourings.
        panels = {}
        for modality in mods:
            for timepoint in tps:
                features, object_ids = _modality_features(
                    modality, timepoint, msession, neural_trials, slopes, t_ref, window)
                if features.shape[0] < min_trials or features.ndim != 2 or features.shape[1] == 0:
                    panels[(modality, timepoint)] = None
                    ws('  {} / {} @ {}: only {} usable trial(s); skipping panel.'.format(
                        session, modality, timepoint, features.shape[0]))
                    continue
                panels[(modality, timepoint)] = (
                    _embed(features, n_neighbors, min_dist), object_ids)
                rs('  {} / {} @ {}: embedded {} trials, {} features.'.format(
                    session, modality, timepoint, features.shape[0], features.shape[1]))

        if not any(panels.values()):
            ws('Skipping session {}: no modality/timepoint had enough data.'.format(session))
            continue

        save_dir = os.path.join(processed_server, session, 'prehension_plots')
        tx = '_tx' if use_threshold_crossings else ''

        # force-coloured figure (marker = kinematic condition)
        figures.append(_draw_force_figure(session, mods, tps, panels, mobject, group_column,
                                          save_dir, tx, save))
        # purely force-coloured figure (one marker, the force colormap, no condition legend)
        figures.append(_draw_force_only_figure(session, mods, tps, panels, mobject, group_column,
                                               save_dir, tx, save))

        # one figure per object geometry parameter that varies across the used trials
        used_ids = [o for panel in panels.values() if panel is not None for o in panel[1]]
        for column in _geometry_columns(mobject, group_column):
            try:
                values = [float(_object_def(mobject, o)[column]) for o in used_ids]
            except (TypeError, ValueError):
                continue   # non-numeric parameter -> cannot colour continuously
            if len(set(values)) < 2:
                continue   # parameter constant across these trials -> nothing to show
            figures.append(_draw_parameter_figure(session, mods, tps, panels, mobject, column,
                                                  save_dir, tx, save))

    return figures
