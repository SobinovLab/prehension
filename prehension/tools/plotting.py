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


def savefig(dirname, filename):
    if dirname is None:
        return
    os.makedirs(dirname, exist_ok=True)

    plt.savefig(os.path.join(dirname, filename + ".png"))
    plt.savefig(os.path.join(dirname, filename + ".pdf"))


def actual_vline(ax, x, **kwargs):
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ymin, ymax)


def xy_numsubplots(numsubplots):
    yn_subplots = int(np.ceil(np.sqrt(numsubplots)))
    xn_subplots = int(np.ceil(numsubplots / yn_subplots))
    return xn_subplots, yn_subplots


def match_yaxes_ranges(axs):
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
