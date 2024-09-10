#!python3.7
import os
import sys
import inspect
import time
import datetime
import re
import argparse
import cv2

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_ani
import scipy
import scipy.io

import ncams

# include local library functions - TB included in NCams
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common import preset
from common import tools
from common import io_tools
from common import meta_session
from common.materialsio_colors import materialsio_colors_rgb as micolors


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
        i_frame = tools.find_first(self.ps_times >= time)

        self.display_ps_frame(i_frame=i_frame)




def animate_ps(mstruct, trial):
    ps_name = tools.RPS_NAME
    speed = 10

    # load data
    times_transformed, matrices_transformed = io_tools.import_matrices(
        trial.transformed_ps_filenames[ps_name])
    times_filtered, matrices_filtered = io_tools.import_matrices(
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


def main(server, session, trial_number):
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


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Shows the forces exerted by each finger as measured manually and matched'
                     ' automatically.'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})
    tools.add_default_arguments(
        parser, ('session', 'trial'))

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.session, args.trial)
    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))

