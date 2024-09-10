#!python3.7
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import io_tools
from . import meta_session
from . import tools
from .materialsio_colors import materialsio_colors_rgb as micolors

LPS_NAME = 'medial_sensor'
RPS_NAME = 'lateral_sensor'
SEGMENT_DIGIT_GROUPS = {
    'thumb': lambda v: re.search('[RL]A[0-9][MPD]1_.*', v),
    'index': lambda v: re.search('[RL]A[0-9][MPD]2_.*', v),
    'middle': lambda v: re.search('[RL]A[0-9][MPD]3_.*', v),
    'ring': lambda v: re.search('[RL]A[0-9][MPD]4_.*', v),
    'pinky': lambda v: re.search('[RL]A[0-9][MPD]5_.*', v),
    'None': lambda v: True
}

DIGITS = tools.DIGITS
UNCLAIMED_NAME = tools.UNCLAIMED_NAME
UNCLAIMED_INDEX = tools.UNCLAIMED_INDEX


def get_matched_contacts(mstruct, trial):
    ps_matrices = {}
    matched_contacts = {}
    segments_set = set()
    for ps_name in mstruct['ps_dic'].keys():
        ps_times, ps_matrices[ps_name] = io_tools.import_matrices(
            trial.get_post_ps_filenames()[ps_name])
        matched_contacts[ps_name] = io_tools.import_matched_contacts(
            trial.matched_contacts_filenames[ps_name])
        for mc in matched_contacts[ps_name]:
            segments_set = segments_set.union(list(mc.keys()))
    segments_set = sorted(list(segments_set), key=lambda v: int(v[4])*10+int(v[2]))

    print('Found {} segments making contacts with pressure plates: {}'.format(
        len(segments_set), ', '.join(segments_set)))

    # build up lists of forces per segment
    data = {}
    num_sensels = {}
    avg_dist = {}
    residual_force = {}
    for ps_name in mstruct['ps_dic'].keys():
        # sugar
        ps_matrix = ps_matrices[ps_name]
        matched_sensels = np.zeros(np.shape(ps_matrix), dtype=bool)
        matched_contact = matched_contacts[ps_name]
        dat = [[] for _ in segments_set]
        nss = [[] for _ in segments_set]
        avd = [[] for _ in segments_set]

        for psm, mc, ms in zip(ps_matrix, matched_contact, matched_sensels):
            for iseg, segment in enumerate(segments_set):
                if segment not in mc.keys():
                    dat[iseg].append(0)
                    nss[iseg].append(0)
                    avd[iseg].append(np.nan)
                else:
                    dat[iseg].append(sum(psm[contact[0]][contact[1]] for contact in mc[segment]))
                    nss[iseg].append(len(mc[segment]))
                    avd[iseg].append(np.mean([contact[2] for contact in mc[segment]]))
                    for contact in mc[segment]:
                        ms[contact[0], contact[1]] = True

        # save
        data[ps_name] = dat
        num_sensels[ps_name] = nss
        avg_dist[ps_name] = avd
        residual_force[ps_name] = np.sum(np.logical_not(matched_sensels) * np.array(ps_matrix),
                                         axis=(1, 2))

    # group up
    segment_digit_groups = []
    for segment in segments_set:
        for i_sdg, sdg_l in enumerate(SEGMENT_DIGIT_GROUPS.values()):
            if sdg_l(segment):
                segment_digit_groups.append(i_sdg)
                break
    # report on digit-segment groups
    print('Groups:')
    for i_sdg, sdg_n in enumerate(SEGMENT_DIGIT_GROUPS.keys()):
        print('\t{}: {}'.format(
            sdg_n, ', '.join(segment for segment, sdg in zip(segments_set, segment_digit_groups)
                             if sdg == i_sdg)))

    # Group the data
    data_digits = {}
    for ps_name in mstruct['ps_dic'].keys():
        data_digits[ps_name] = [np.zeros(np.shape(ps_times)) for _ in SEGMENT_DIGIT_GROUPS]
        for i_sdg, d in zip(segment_digit_groups, data[ps_name]):
            data_digits[ps_name][i_sdg] = data_digits[ps_name][i_sdg] + d
        # adding residual to None
        data_digits[ps_name][-1] += residual_force[ps_name]

    # group across pressure sensors
    data_digits_aps = []
    # assuming there is two pressure sensors
    for i_sdg, _ in enumerate(SEGMENT_DIGIT_GROUPS):
        data_digits_aps.append(data_digits['medial_sensor'][i_sdg] +
                               data_digits['lateral_sensor'][i_sdg])


    return data_digits_aps, ps_times, ps_matrices


class GraspAnimation:
    def __init__(self, trial, ps_times, ps_matrices,
                 lps_video, rps_video, fps, num_frames, data_digits_aps):
        self.trial = trial
        self.ps_times = ps_times
        self.ps_matrices = ps_matrices
        self.lps_video = lps_video
        self.rps_video = rps_video
        self.fps = fps
        self.num_frames = num_frames
        self.data_digits_aps = data_digits_aps
        self.generate_internal_data()

        # generate figure template and axes
        self.fig = plt.figure(figsize=(14, 9))

        self.ax_lps = self.fig.add_axes([0.0, 0.0, 0.25, 0.5])
        self.ax_rps = self.fig.add_axes([0.25, 0.0, 0.25, 0.5])
        self.setup_ps_axes()
        self.lps_image = None
        self.rps_image = None

        self.ax_lps_video = self.fig.add_axes([0.0, 0.5, 0.25, 0.5])
        self.ax_rps_video = self.fig.add_axes([0.25, 0.5, 0.25, 0.5])
        self.setup_vid_axes()
        self.lps_videoframe = None
        self.rps_videoframe = None

        self.ax_mujoco = self.fig.add_axes([0.5, 0.5, 0.5, 0.5])
        self.setup_mujoco_axes()

        self.ax_fingers = self.fig.add_axes([0.5, 0, 0.5, 0.5])
        self.setup_fingers_axes()

        self.lps_auto_colormask = None
        self.rps_auto_colormask = None

        self.matched_contacts = {
            LPS_NAME: io_tools.import_matched_contacts(
                self.trial.matched_contacts_filenames[LPS_NAME]),
            RPS_NAME: io_tools.import_matched_contacts(
                self.trial.matched_contacts_filenames[RPS_NAME])
        }


    def setup_ps_axes(self):
        self.ax_lps.set_xticks([])
        self.ax_lps.set_yticks([])
        self.ax_lps.set_ylim([-0.5, self.nsenselsr - 0.5])  # default is upside down

        self.ax_rps.set_xticks([])
        self.ax_rps.set_yticks([])
        self.ax_rps.set_ylim([-0.5, self.nsenselsr - 0.5])
        self.ax_rps.set_xlim([self.nsenselsr - 0.5, -0.5])  # monkey in the middle

    def setup_vid_axes(self):
        self.ax_lps_video.set_xticks([])
        self.ax_lps_video.set_yticks([])
        # self.ax_lps_video.set_ylim(top=400, bottom=1080)
        self.ax_rps_video.set_xticks([])
        self.ax_rps_video.set_yticks([])
        # self.ax_rps_video.set_ylim(top=400, bottom=1080)

    def setup_mujoco_axes(self):
        self.ax_mujoco.set_xticks([])
        self.ax_mujoco.set_yticks([])
        self.ax_mujoco.spines[['right', 'top']].set_visible(False)

    def setup_fingers_axes(self):
        self.ax_fingers.set_xticks([])
        self.ax_fingers.set_yticks([])

        self.ax_fingers.set_xlim([self.ps_times[0], self.ps_times[-1]])
        self.ax_fingers.set_ylim([0, self.max_finger_force * 1.05])
        self.ax_fingers.spines[['right', 'top']].set_visible(False)

    def generate_internal_data(self):
        self.nsenselsr = np.shape(self.ps_matrices[LPS_NAME])[1]  # number of sensels one direction
        self.left_sensor_total = np.sum(self.ps_matrices[LPS_NAME], axis=(1, 2))
        self.right_sensor_total = np.sum(self.ps_matrices[RPS_NAME], axis=(1, 2))
        self.ps_vmax_d = {
            LPS_NAME: np.max(self.ps_matrices[LPS_NAME]),
            RPS_NAME: np.max(self.ps_matrices[RPS_NAME]),
        }
        self.ps_vmax = max(self.ps_vmax_d[LPS_NAME], self.ps_vmax_d[RPS_NAME])
        # self.camera_pstime_diff = self.trial.ttl_to_grasp - self.trial.ttl_to_ja_start
        self.camera_pstime_diff = - self.trial.ttl_to_ja_start

        self.sdg = {k: v for ik, (k, v) in enumerate(SEGMENT_DIGIT_GROUPS.items())
                    if ik < len(SEGMENT_DIGIT_GROUPS) - 1}
        self.sdg_colors = [DIGITS[k]['c'] for ik, k in enumerate(SEGMENT_DIGIT_GROUPS.keys())
                           if ik < len(SEGMENT_DIGIT_GROUPS) - 1]

        self.max_finger_force = np.max(self.data_digits_aps)

    def force_map_transform(self, matrix):
        return np.power(matrix, 0.25)

    def display_ps_frame(self, i_frame=None):
        print('Loading pressure sensor data...', end='')
        # self.display_load_msg()
        if i_frame is None:
            i_frame = self.i_frame
        if self.lps_image is not None:
            self.lps_image.remove()
        if self.rps_image is not None:
            self.rps_image.remove()
        matrix = self.ps_matrices[LPS_NAME][i_frame] / self.ps_vmax_d[LPS_NAME]
        self.lps_image = self.ax_lps.imshow(self.force_map_transform(matrix),
                                            vmin=0, vmax=1, cmap='Greys')
        matrix = self.ps_matrices[RPS_NAME][i_frame] / self.ps_vmax_d[RPS_NAME]
        self.rps_image = self.ax_rps.imshow(self.force_map_transform(matrix),
                                            vmin=0, vmax=1, cmap='Greys')

        if self.lps_auto_colormask is not None:
            self.lps_auto_colormask.remove()
        if self.rps_auto_colormask is not None:
            self.rps_auto_colormask.remove()
        # show auto stuff
        lps_digit_color_mask = np.ones((self.nsenselsr, self.nsenselsr, 3))
        rps_digit_color_mask = np.ones((self.nsenselsr, self.nsenselsr, 3))

        for idigit, ds in enumerate(DIGITS.values()):
            if idigit == UNCLAIMED_INDEX:
                continue
            else:
                # print(ds)
                color = micolors[ds['c']][600]
            lpsdmask = tools.get_matched_contact_frame_mask(
                ds['exp'], self.matched_contacts[LPS_NAME][i_frame],
                (self.nsenselsr, self.nsenselsr))
            lps_digit_color_mask[lpsdmask, 0] = color[0]
            lps_digit_color_mask[lpsdmask, 1] = color[1]
            lps_digit_color_mask[lpsdmask, 2] = color[2]

            rpsdmask = tools.get_matched_contact_frame_mask(
                ds['exp'], self.matched_contacts[RPS_NAME][i_frame],
                (self.nsenselsr, self.nsenselsr))
            rps_digit_color_mask[rpsdmask, 0] = color[0]
            rps_digit_color_mask[rpsdmask, 1] = color[1]
            rps_digit_color_mask[rpsdmask, 2] = color[2]
        self.lps_auto_colormask = self.ax_lps.imshow(lps_digit_color_mask, alpha=0.6)
        self.rps_auto_colormask = self.ax_rps.imshow(rps_digit_color_mask, alpha=0.6)

        # self.clear_ps_msg()
        print(' Loaded.')

    def display_videoframe(self, video, ax, cameraframe):
        video.set(cv2.CAP_PROP_POS_FRAMES, cameraframe)
        fe, frame = video.read()
        if fe is False:
            print('Could not read the frame #{}.'.format(cameraframe))
            return
        frame_rgb = frame[..., ::-1].copy()
        return ax.imshow(frame_rgb)

    def display_videos_frame(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        # the frame difference
        cameraframe = int(round((self.ps_times[i_frame] + self.camera_pstime_diff) * self.fps)) - 1
        cameraframe = max(0, cameraframe)
        cameraframe = min(cameraframe, self.num_frames-1)

        if self.lps_videoframe is not None:
            self.lps_videoframe.remove()
        if self.rps_videoframe is not None:
            self.rps_videoframe.remove()

        self.lps_videoframe = self.display_videoframe(self.lps_video, self.ax_lps_video, cameraframe)
        self.rps_videoframe = self.display_videoframe(self.rps_video, self.ax_rps_video, cameraframe)
        self.fig.canvas.draw()

    def display_force_trace(self, time):
        sbs = self.ps_times <= time
        times = self.ps_times[sbs]

        for i_sdg, sdg_color in enumerate(self.sdg_colors):
            vals = self.data_digits_aps[i_sdg][sbs]
            self.ax_fingers.plot(times, vals, color=sdg_color)



    def display_mujoco_video(self, i_frame=None):
        pass

    def display_time(self, time):
        i_frame = tools.find_first(self.ps_times >= time)

        self.display_videos_frame(i_frame=i_frame)
        self.display_ps_frame(i_frame=i_frame)
        self.display_force_trace(time)
        self.display_mujoco_video(i_frame=i_frame)

        self.fig.canvas.draw()


def animate_grasp_all(mstruct, trial):
    lps_ref_camera_name = '19194005'
    rps_ref_camera_name = '19340396'
    # load data
    data_digits_aps, ps_times, ps_matrices = get_matched_contacts(mstruct, trial)
    video = cv2.VideoCapture(trial.videos[lps_ref_camera_name])
    video_rps = cv2.VideoCapture(trial.videos[rps_ref_camera_name])
    fps = int(video.get(cv2.CAP_PROP_FPS))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    ga = GraspAnimation(trial, ps_times, ps_matrices, video, video_rps, fps, num_frames, data_digits_aps)

    for i_frame, time in enumerate(ps_times[ps_times >= -1.8]):
        ga.display_time(time)
        ga.fig.savefig(
            os.path.join('E:\\', 'video',
                         'img{:04d}.png'.format(i_frame)),
            dpi=300)


def animate_grasp(server, session, trial_number):
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

    animate_grasp_all(mstruct, trial)
