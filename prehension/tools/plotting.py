#!python3
# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting in matplotlib.

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
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


############ Export
def savefig(dirname, filename, fig=None, dpi=None, bbox_inches=None):
    '''Save a figure as both PNG and PDF into dirname (created if needed).

    Saves the current pyplot figure by default, or ``fig`` when given.  ``dpi`` and
    ``bbox_inches`` are forwarded to savefig when provided.
    '''
    if dirname is None:
        return
    os.makedirs(dirname, exist_ok=True)

    saver = fig.savefig if fig is not None else plt.savefig
    kwargs = {}
    if dpi is not None:
        kwargs['dpi'] = dpi
    if bbox_inches is not None:
        kwargs['bbox_inches'] = bbox_inches
    saver(os.path.join(dirname, filename + '.png'), **kwargs)
    saver(os.path.join(dirname, filename + '.pdf'), **kwargs)


############ Colors
def cmap_norm(values, vmax, cmap_name='autumn_r'):
    '''Build a colormap + linear norm (0..vmax) and per-value colours.

    Returns (cmap, norm, [cmap(norm(v)) for v in values]).  vmax <= 0 falls back to
    a range of 1 so the norm stays valid.
    '''
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=0, vmax=vmax if vmax > 0 else 1)
    return cmap, norm, [cmap(norm(v)) for v in values]


############# Drawing
def actual_vline(ax, x, **kwargs):
    '''Draws a vline covering the whole y-range, preserving the previous ylim.'''
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ymin, ymax)


def actual_hline(ax, x, **kwargs):
    '''Draws a hline covering the whole x-range, preserving the previous xlim.'''
    xmin, xmax = ax.get_xlim()
    ax.hlines(x, xmin, xmax, **kwargs)
    ax.set_xlim(xmin, xmax)


def annotated_vbar(ax, x, title, ha='center', color='#F44336', linestyle='--', continue_axs=None):
    '''Makes and annotates a vline, continues it onto other axes.'''
    _, ymax = ax.get_ylim()
    actual_vline(ax, x, color=color, linestyle=linestyle)
    ax.annotate(
        title, (x, ymax),
        xycoords='data', ha=ha, va='bottom')
    if continue_axs is not None:
        for ax_ in continue_axs:
            actual_vline(ax_, x, color=color, linestyle=linestyle)


def annotated_vspan(ax, xmin, xmax, title,
                    ha='center', color='#F44336', alpha=0.3, continue_axs=None):
    '''Makes and annotates a vspan (shaded box), continues it onto other axes.'''
    _, ymax = ax.get_ylim()
    ax.axvspan(xmin, xmax, color=color, alpha=alpha)
    ax.annotate(
        title, ((xmax-xmin)/2, ymax),
        xycoords='data', ha=ha, va='bottom')
    if continue_axs is not None:
        for ax_ in continue_axs:
            ax_.axvspan(xmin, xmax, color=color, alpha=alpha)


############# Arranging subplots
def xy_numsubplots(numsubplots):
    '''Calculates the number of columns/rows to fit numsubplots approximately in a square.'''
    yn_subplots = int(np.ceil(np.sqrt(numsubplots)))
    xn_subplots = int(np.ceil(numsubplots / yn_subplots))
    return xn_subplots, yn_subplots


############# Interval tracks
def interval_step(starts, stops):
    '''Build x/y arrays of a 0/1 step signal that is high during each [start, stop].

    Overlapping intervals are clamped to 1.  Returns (xs, ys) numpy arrays.
    '''
    edges = [(float(s), 1) for s in starts]
    edges += [(float(e), -1) for e in stops if np.isfinite(e)]
    edges.sort(key=lambda p: (p[0], -p[1]))
    if not edges:
        return np.array([0.0]), np.array([0])

    xs, ys, level = [edges[0][0]], [0], 0
    for t, d in edges:
        xs.append(t)
        ys.append(level)
        level = max(0, min(1, level + d))  # clamp for overlapping intervals
        xs.append(t)
        ys.append(level)
    xs.append(edges[-1][0])
    ys.append(level)
    return np.array(xs), np.array(ys)


def plot_interval_track(ax, starts, stops, indices, ylabel, up_label, down_label,
                        max_labels=40):
    '''Draw a 0/1 track high during each [start, stop], with sparse index labels.

    Rising edges are marked in green (labelled ``up_label``), falling edges in cyan
    (``down_label``); at most ``max_labels`` of the ``indices`` are annotated.
    '''
    starts = np.asarray(starts, dtype=float)
    stops = np.asarray(stops, dtype=float)
    xs, ys = interval_step(starts, stops)
    ax.plot(xs, ys, color='k', linewidth=1.0)
    ax.plot(starts, np.ones_like(starts), '|', color='tab:green', markersize=10,
            label=up_label)
    finite_stops = stops[np.isfinite(stops)]
    ax.plot(finite_stops, np.zeros_like(finite_stops), '|', color='tab:cyan',
            markersize=10, label=down_label)
    ax.axvline(0.0, color='tab:blue', linestyle='--', linewidth=0.8)
    ax.set_ylim(-0.2, 1.4)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['low', 'high'])
    ax.set_ylabel(ylabel)
    ax.legend(loc='upper right', fontsize=8)

    label_every = max(1, int(np.ceil(len(starts) / max_labels)))
    for k, (s, idx) in enumerate(zip(starts, indices)):
        if k % label_every == 0:
            ax.annotate(str(idx), xy=(s, 1.0), xytext=(s, 1.12), ha='center',
                        va='bottom', fontsize=7, color='tab:green', rotation=90)


############# Ranges
def match_yaxes_ranges(axs):
    '''Makes the y-ranges (spans) of axes [y_max;y_min] match between axes.

    Preserves the middle of the range.
    '''
    max_yrange = -np.inf
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        yrange = ymax - ymin
        max_yrange = max(yrange, max_yrange)

    if np.isinf(max_yrange):
        return

    yhrange = max_yrange / 2
    for ax in axs:
        ymin, ymax = ax.get_ylim()
        ymid = ymin + (ymax - ymin) / 2
        ax.set_ylim((ymid - yhrange, ymid + yhrange))


def share_ylim(axs):
    '''Forces the axes to share the limit without using linkaxes.'''
    min_ylim = min([ax.get_ylim()[0] for ax in axs])
    max_ylim = max([ax.get_ylim()[1] for ax in axs])

    for ax in axs:
        ax.set_ylim([min_ylim, max_ylim])


def symmetrize_y_axis(ax):
    '''Makes the axes symmetrical around 0.'''
    ymin, ymax = ax.get_ylim()
    ymax = max(abs(ymin), abs(ymax))
    ymin = -ymax
    ax.set_ylim([ymin, ymax])
