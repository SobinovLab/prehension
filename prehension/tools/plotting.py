#!python3
# -*- coding: utf-8 -*-
"""
Miscellaneous functions for plotting in matplotlib.

Copyright (C) 2019-2024 Anton Sobinov
https://github.com/BensmaiaLab/prehension

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


############ Export
def savefig(dirname, filename):
    if dirname is None:
        return
    os.makedirs(dirname, exist_ok=True)

    plt.savefig(os.path.join(dirname, filename + '.png'))
    plt.savefig(os.path.join(dirname, filename + '.pdf'))


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
