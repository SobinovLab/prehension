#!python3
# -*- coding: utf-8 -*-
"""
Animates pressure sensors from a trial.

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

import matplotlib.animation as mpl_ani
import matplotlib.pyplot as plt
import numpy as np


from .tools.constants import RPS_NAME
from .tools import misc
from .tools import io

from . import meta_session


class PSDisplayer:
    def __init__(self, ax, ps_times, ps_matrices):
        self.ax = ax
        self.ps_times = ps_times
        self.ps_matrices = ps_matrices
        self.generate_internal_data()

        # generate figure template and axes
        self.setup_ps_ax()
        self.image = None

    def setup_ps_ax(self):
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_ylim([-0.5, self.nsenselsr - 0.5])  # default is upside down

    def generate_internal_data(self):
        self.nsenselsr = np.shape(self.ps_matrices)[1]  # number of sensels one direction
        self.sensor_total = np.sum(self.ps_matrices, axis=(1, 2))
        self.st_max = np.max(self.sensor_total)
        self.ps_vmax = np.max(self.ps_matrices)

    def force_map_transform(self, matrix):
        return np.sqrt(matrix)

    def display_ps_frame(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        else:
            self.i_frame = i_frame
        if self.image is not None:
            self.image.remove()
        matrix = self.ps_matrices[i_frame] / self.ps_vmax
        self.image = self.ax.imshow(self.force_map_transform(matrix),
                                    vmin=0, vmax=1, cmap='Greys')

    def display_time(self, time):
        i_frame = misc.find_first(self.ps_times >= time)

        self.display_ps_frame(i_frame=i_frame)


def animate_ps(mstruct, trial):
    ps_name = RPS_NAME
    speed = 10

    # load data
    times_transformed, matrices_transformed = io.import_matrices(
        trial.transformed_ps_filenames[ps_name])
    times_filtered, matrices_filtered = io.import_matrices(
        trial.filtered_ps_filenames[ps_name])

    times = times_transformed
    dt = np.median(np.diff(times)) * 1000  # msec

    # generate figure
    fig = plt.figure(figsize=(16, 9))
    ax_tr = fig.add_axes([0.0, 0.3, 0.5, 0.6])
    ax_fi = fig.add_axes([0.5, 0.3, 0.5, 0.6])
    ax_to = fig.add_axes([0.05, 0.05, 0.9, 0.2])

    psd_tr = PSDisplayer(ax_tr, times_transformed, matrices_transformed)
    psd_fi = PSDisplayer(ax_fi, times_filtered, matrices_filtered)
    ps_vmax = psd_fi.ps_vmax
    psd_tr.ps_vmax = psd_fi.ps_vmax = ps_vmax
    st_max = psd_fi.st_max
    psd_tr.st_max = psd_fi.st_max = st_max

    ax_to.plot(times_transformed, psd_tr.sensor_total, 'k')
    ax_to.plot(times_filtered, psd_fi.sensor_total, 'r--')
    ax_to.set_xlim([times[0], times[-1]])
    yrange = [0, st_max*1.05]
    yrange = [0, st_max*0.05]
    ax_to.set_ylim(yrange)
    line, = ax_to.plot([times[0]]*2, yrange, 'g')
    ax_to.set_xlabel('Time, s')
    ax_to.set_ylabel('Force, N')

    def update(frame):
        psd_tr.display_time(times[frame])
        psd_fi.display_time(times[frame])
        line.set_data([times[frame]]*2, yrange)
        return [psd_tr.image, psd_fi.image, line]

    ani = mpl_ani.FuncAnimation(
        fig, update,
        frames=range(len(times)), blit=True, repeat=False,
        interval=dt/speed)

    ani.save('mymovie_larger.mp4', writer='ffmpeg', fps=int(1000/dt), dpi=120, codec='h264',
             progress_callback=lambda i, n: print(f'Saving frame {i}/{n}', end='\r'))

    # plt.show()


def animate_pressure_sensors(server, session, trial_number):
    """Shows the forces exerted by each finger as measured manually and matched automatically.

    Arguments:
        server {str} --- Folder where the sessions are located.
        session {str} --- Session directory to use.
        trial_number {int} --- Trial to do adjustment on.
    """
    if len(session) == 0:
        session = meta_session.find_session_dirs(server)[0]

    server_session = os.path.join(server, session)
    mstruct, _, _, msession = meta_session.load_meta_information(server_session)

    trial = meta_session.find_trial(msession, trial_number)

    if trial is None:
        raise ValueError('Could not find the trial.')
    if not trial.do_matched_contacts_files_exist():
        raise ValueError('Associated matched contacts files do not exist.')

    animate_ps(mstruct, trial)
