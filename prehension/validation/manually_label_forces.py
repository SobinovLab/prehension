#!python3.7
import os
import warnings
from itertools import product

import cv2
import matplotlib.font_manager as font_manager
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from .. import io_tools
from .. import meta_session
from .. import tools
from ..materialsio_colors import materialsio_colors_rgb as micolors
from ..tools import rs, ws

LPS_NAME = 'medial_sensor'
RPS_NAME = 'lateral_sensor'
SHIFT_JUMP = 10
DIGITS = tools.DIGITS
UNCLAIMED_NAME = tools.UNCLAIMED_NAME
UNCLAIMED_INDEX = tools.UNCLAIMED_INDEX

# 0.5 for standard view, 0.25 for enhanced visibility of patterns
FORCE_POWER_TRANSFORM_FOR_MAP = 0.5


def rel_rect_to_fig(rect, figsize):
    return (rect[0]*figsize[0], rect[1]*figsize[1], rect[2]*figsize[0], rect[3]*figsize[1])


def apply_padding(rect, padding):
    '''Padding inputs:
        <all sides>
        [<all sides>]
        [<horizontal>, <vertical>]
        [<left>, <bottom>, <right>, <top>]
    '''
    if isinstance(padding, (int, float)):
        padding = [padding]
    if len(padding) == 1:
        return (rect[0] + padding[0], rect[1] + padding[0],
                rect[2] - 2 * padding[0], rect[3] - 2 * padding[0])
    if len(padding) == 2:
        return (rect[0] + padding[0], rect[1] + padding[1],
                rect[2] - 2 * padding[0], rect[3] - 2 * padding[1])
    if len(padding) == 4:
        return (rect[0] + padding[0], rect[1] + padding[1],
                rect[2] - (padding[0] + padding[2]), rect[3] - (padding[1] + padding[3]))
    raise ValueError('Could not recognize format of padding.')


class ForceLabellingInterface:
    def __init__(self, figsize, trial,
                 lps_video, rps_video, fps, num_frames, show_automatic):
        # generate figure template and axes
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.manager.set_window_title(
            'Force Labelling Trial #{}'.format(trial.trial_number))
        self.show_automatic = show_automatic

        # sizes designed for a 16x9 format
        padding = 0.01
        # LEFT PANEL
        lpl = 0  # left panel left
        lpw = 9/16  # left panel width
        lphw = lpw / 2  # left panel halfwidth
        lpm = lpl + lpw / 2  # left panel middle
        if show_automatic:
            # vertical space is divided into thirds
            lpeh = 0.33  # left panel element height
        else:
            # into halves
            lpeh = 0.5  # left panel element height

        # left PS video reference
        self.ax_lps_vid = self.fig.add_axes(
            apply_padding((lpl, 1. - lpeh, lphw, lpeh), padding))

        # right PS video reference
        self.ax_rps_vid = self.fig.add_axes(
            apply_padding((lpm, 1. - lpeh, lphw, lpeh), padding))

        # left pressure sensor
        self.ax_lps = self.fig.add_axes(apply_padding(
            (lpl, 1. - 2*lpeh, lphw, lpeh), padding))
        # right pressure sensor
        self.ax_rps = self.fig.add_axes(apply_padding(
            (lpm, 1. - 2*lpeh, lphw, lpeh), padding))
        # monkey sits here
        self.add_monkey_location(lpm, 1. - 1.5*lpeh)

        # underneath the manually labeled show auto labelled
        if show_automatic:
            # automatic left pressure sensor
            self.ax_lps_auto = self.fig.add_axes(
                apply_padding((lpl, 0, lphw, lpeh), padding))
            # automatic right pressure sensor
            self.ax_rps_auto = self.fig.add_axes(
                apply_padding((lpm, 0, lphw, lpeh), padding))
            # also monkey sits here
            self.add_monkey_location(lpm, 1. - 2.5*lpeh)

        # RIGHT PANEL
        rpl = lpl + lpw  # right panel left
        rpw = 1. - rpl  # right panel width
        rphw = rpw / 2  # right panel halfwidth
        rpm = rpl + rphw

        # top of the panel
        if show_automatic:
            rpteh = 3./9  # right panel top element height
        else:
            rpteh = 5./9  # right panel top element height
        rptseh = rpteh * 0.5  # right panel top subelement height

        # Average left pressure sensor
        self.ax_lps_avg = self.fig.add_axes(
            (rpl + 1/32, 1. - rptseh, rphw, rptseh))

        # Average right pressure sensor
        self.ax_rps_avg = self.fig.add_axes(
            (rpm, 1. - rptseh, rphw - 1/16, rptseh))

        # digit selector
        self.ax_digit_selector = self.fig.add_axes(
            (rpl + 1/16, 1 - rpteh, rphw - 1/16, rptseh))

        # save load
        self.ax_saveload = self.fig.add_axes(
            (rpm, 1 - rpteh, rphw - 1/16, rptseh))
        self.make_saveload_btns()

        # Bottom of the panel
        rpbeh = 1 - rpteh  # right panel bottom element height
        if show_automatic:
            rpbseh = rpbeh / 4  # right panel bottom subelement height
        else:
            rpbseh = rpbeh * 0.5  # right panel bottom subelement height

        # total force
        self.ax_tf = self.fig.add_axes(apply_padding((rpl, rpbeh - rpbseh, rpw, rpbseh),
                                                     (0.05, 0.01, 0.05, 0.04)))
        self.ax_tf.set_ylabel('Total forces, N')

        # matched force
        self.ax_mf = self.fig.add_axes(apply_padding((rpl, rpbeh - 2*rpbseh, rpw, rpbseh),
                                                     (0.05, 0.05, 0.05, 0.01)))

        if show_automatic:
            # automatic matched forces
            self.ax_mf_auto = self.fig.add_axes(apply_padding((rpl, rpbseh, rpw, rpbseh),
                                                              (0.05, 0.05, 0.05, 0.01)))
            # difference bw matched forces
            self.ax_mf_diff = self.fig.add_axes(apply_padding((rpl, 0, rpw, rpbseh),
                                                              (0.05, 0.05, 0.05, 0.01)))

        # save data to object instance
        self.trial = trial
        self.lps_video = lps_video
        self.rps_video = rps_video
        self.fps = fps
        self.num_frames = num_frames
        self.generate_internal_data()

        # setup static things
        self.plot_total_force()
        self.setup_matched_force_figure()
        self.plot_average_force_map()
        self.setup_digit_selector()
        self.make_iframe_highlights()
        self.setup_ps_axes()
        self.setup_vid_axes()
        self.setup_automatic_matched_force_figure()

        # cleanup variable
        self.lps_colormask = None
        self.rps_colormask = None
        self.avg_lps_colormask = None
        self.avg_rps_colormask = None
        self.lps_image = None
        self.rps_image = None
        if self.show_automatic:
            self.lps_auto_colormask = None
            self.rps_auto_colormask = None
            self.lps_auto_image = None
            self.rps_auto_image = None
        self.lps_videoframe = None
        self.rps_videoframe = None
        # set initial positions of all objects
        self.find_starting_index()
        self.display_ps_frame()
        self.update_vertical_lines()
        self.load_maps()
        self.display_digit_areas()
        self.calculate_digit_portions()
        self.plot_digit_portions()
        self.display_videos_frame()

        # connect the events
        self.highlight_pressevent = None
        self.savebtn_pressevent = None
        self.loadbtn_pressevent = None
        self.lps_avg_pressevent = None
        self.rps_avg_pressevent = None
        self.shift = False
        self.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.fig.canvas.mpl_connect(
            'key_release_event', self.on_key_release)

    def add_monkey_location(self, x, y):
        fontlocation = os.path.join(
            '../../../stereo_inverse_kinematics', 'common', 'NotoEmoji-Regular.ttf')
        if os.path.exists(fontlocation):
            prop = font_manager.FontProperties()
            prop.set_file(fontlocation)
            self.fig.text(x, y, u"\U0001F412", ha='center', va='center', size='xx-large',
                          fontproperties=prop)
        else:
            self.fig.text(x, y, 'MONKEY', ha='center', va='center', size='xx-large',
                          rotation=90)

    def generate_internal_data(self):
        # ps_times should be the same for all
        self.ps_matrices = {}
        self.ps_times, self.ps_matrices[LPS_NAME] = io_tools.import_matrices(
            self.trial.get_post_ps_filenames()[LPS_NAME])
        _, self.ps_matrices[RPS_NAME] = io_tools.import_matrices(
            self.trial.get_post_ps_filenames()[RPS_NAME])

        self.nsenselsr = np.shape(self.ps_matrices[LPS_NAME])[
            1]  # number of sensels one direction
        self.left_sensor_total = np.sum(
            self.ps_matrices[LPS_NAME], axis=(1, 2))
        self.right_sensor_total = np.sum(
            self.ps_matrices[RPS_NAME], axis=(1, 2))
        self.dt = np.median(np.diff(self.ps_times))
        self.dt_visualize = self.dt * 5
        self.dt_offset_times = self.ps_times - self.dt_visualize
        self.ps_vmax_d = {
            LPS_NAME: np.max(self.ps_matrices[LPS_NAME]),
            RPS_NAME: np.max(self.ps_matrices[RPS_NAME]),
        }
        self.ps_vmax = max(
            self.ps_vmax_d[LPS_NAME], self.ps_vmax_d[RPS_NAME])
        self.sensor_total = np.sum(np.vstack((self.left_sensor_total, self.right_sensor_total)),
                                   axis=0)
        self.max_sensor_total = np.amax(self.sensor_total)
        self.sum_sensor_total = np.sum(self.sensor_total)
        # self.camera_pstime_diff = self.trial.ttl_to_grasp - self.trial.ttl_to_ja_start
        self.camera_pstime_diff = - self.trial.ttl_to_ja_start

        # load matched data
        if self.show_automatic:
            if (not os.path.exists(self.trial.matched_contacts_filenames[LPS_NAME]) or
                    not os.path.exists(self.trial.matched_contacts_filenames[RPS_NAME])):
                ws('One or more of the matched contacts files do not exist.')
                self.show_automatic = False
            else:
                # load
                self.matched_contacts = {
                    LPS_NAME: io_tools.import_matched_contacts(
                        self.trial.matched_contacts_filenames[LPS_NAME]),
                    RPS_NAME: io_tools.import_matched_contacts(
                        self.trial.matched_contacts_filenames[RPS_NAME])
                }
                # get traces
                self.total_auto_force_traces = {}
                for name, d in DIGITS.items():
                    if name == UNCLAIMED_NAME:
                        continue
                    self.total_auto_force_traces[name] = [np.sum(
                        self.ps_matrices[LPS_NAME][i_frame],
                        where=tools.get_matched_contact_frame_mask(
                            d['exp'], self.matched_contacts[LPS_NAME][i_frame],
                            (self.nsenselsr, self.nsenselsr))) + np.sum(
                        self.ps_matrices[RPS_NAME][i_frame],
                        where=tools.get_matched_contact_frame_mask(
                            d['exp'], self.matched_contacts[RPS_NAME][i_frame],
                            (self.nsenselsr, self.nsenselsr)))
                        for i_frame in range(len(self.ps_times))]

    def plot_total_force(self):
        self.ax_tf.plot(self.ps_times, self.left_sensor_total, label='Left',
                        color=micolors['orange'][600])
        self.ax_tf.plot(self.ps_times, self.right_sensor_total, label='Right',
                        color=micolors['blue'][600])
        self.ax_tf.legend()
        self.ax_tf.set_xlim([self.ps_times[0], self.ps_times[-1]])
        self.ax_tf.set_ylim([0, self.max_sensor_total])
        self.ax_tf.set_xticklabels([])

    def setup_matched_force_figure(self):
        self.ax_mf.set_xlim([self.ps_times[0], self.ps_times[-1]])
        self.ax_mf.set_ylim([0, self.max_sensor_total])
        if self.show_automatic:
            self.ax_mf.set_xticklabels([])
        else:
            self.ax_mf.set_xlabel('Time, s')
        self.ax_mf.set_ylabel('Matched forces, N')

    def setup_automatic_matched_force_figure(self):
        if not self.show_automatic:
            return

        # automatically matched forces
        self.ax_mf_auto.set_xlim(
            [self.ps_times[0], self.ps_times[-1]])
        self.ax_mf_auto.set_ylim([0, self.max_sensor_total])
        self.ax_mf_auto.set_ylabel('Automatically\nmatched forces, N')
        self.ax_mf_auto.set_xticklabels([])

        # plot
        for ds in self.digit_selectors:
            if ds['name'] == UNCLAIMED_NAME:
                continue
            self.ax_mf_auto.plot(self.ps_times, self.total_auto_force_traces[ds['name']],
                                 label=ds['name'], color=ds['c'][600])

        # Difference in matched forces
        self.ax_mf_diff.set_xlim(
            [self.ps_times[0], self.ps_times[-1]])
        self.ax_mf_diff.set_ylim([0, self.max_sensor_total])
        self.ax_mf_diff.set_xlabel('Time, s')
        self.ax_mf_diff.set_ylabel('Difference in\nmatched forces, N')

    def setup_ps_axes(self):
        self.ax_lps.set_xticks([])
        self.ax_lps.set_yticks([])
        # default is upside down
        self.ax_lps.set_ylim([-0.5, self.nsenselsr - 0.5])

        self.ax_rps.set_xticks([])
        self.ax_rps.set_yticks([])
        self.ax_rps.set_ylim([-0.5, self.nsenselsr - 0.5])
        # monkey in the middle
        self.ax_rps.set_xlim([self.nsenselsr - 0.5, -0.5])

        self.ax_lps_avg.set_xticks([])
        self.ax_lps_avg.set_yticks([])
        self.ax_lps_avg.set_ylim([-0.5, self.nsenselsr - 0.5])

        self.ax_rps_avg.set_xticks([])
        self.ax_rps_avg.set_yticks([])
        self.ax_rps_avg.set_ylim([-0.5, self.nsenselsr - 0.5])
        self.ax_rps_avg.set_xlim([self.nsenselsr - 0.5, -0.5])

        if self.show_automatic:
            self.ax_lps_auto.set_xticks([])
            self.ax_lps_auto.set_yticks([])
            # default is upside down
            self.ax_lps_auto.set_ylim([-0.5, self.nsenselsr - 0.5])

            self.ax_rps_auto.set_xticks([])
            self.ax_rps_auto.set_yticks([])
            self.ax_rps_auto.set_ylim([-0.5, self.nsenselsr - 0.5])
            self.ax_rps_auto.set_xlim(
                [self.nsenselsr - 0.5, -0.5])  # monkey in the middle

    def setup_vid_axes(self):
        self.ax_lps_vid.set_xticks([])
        self.ax_lps_vid.set_yticks([])

        self.ax_rps_vid.set_xticks([])
        self.ax_rps_vid.set_yticks([])

    def find_starting_index(self):
        self.i_frame = np.where(
            self.sensor_total == self.max_sensor_total)[0][0]

    def force_map_transform(self, matrix):
        # return np.sqrt(matrix)
        return np.power(matrix, FORCE_POWER_TRANSFORM_FOR_MAP)

    def plot_force_map(self, ax, ps_matrix):
        ps_avg = np.mean(self.force_map_transform(
            ps_matrix), where=ps_matrix > 0, axis=0)
        ps_avg /= np.nanmax(ps_avg)
        ax.imshow(ps_avg, vmin=0, vmax=1, cmap='Greys')
        ax.text(0.5 * self.nsenselsr, 0.95 * self.nsenselsr, 'OVERVIEW',
                color=micolors['red'][400], ha='center', va='center')

    def plot_average_force_map(self):
        self.plot_force_map(
            self.ax_lps_avg, self.ps_matrices[LPS_NAME])
        self.plot_force_map(
            self.ax_rps_avg, self.ps_matrices[RPS_NAME])

    def setup_digit_selector(self):
        self.ax_digit_selector.set_xlim([0, 1])
        self.ax_digit_selector.set_ylim([0, 1])
        plt.sca(self.ax_digit_selector)
        plt.axis('off')

        # generate digit selectors
        self.digit_selectors = []
        self.selected_digit = None
        self.press_selected_digit = None
        self.lps_highlight_pressevent = None
        self.rps_highlight_pressevent = None
        self.lps_digit_mask = UNCLAIMED_INDEX * \
            np.ones((self.nsenselsr, self.nsenselsr), dtype=int)
        self.rps_digit_mask = UNCLAIMED_INDEX * \
            np.ones((self.nsenselsr, self.nsenselsr), dtype=int)
        self.isensels_offset_x = [v for v in range(self.nsenselsr)]
        self.isensels_offset_y = [v for v in range(self.nsenselsr)]
        padding = 0.01
        height = (1. - 2 * padding) / len(DIGITS)
        for idigit, (name, d) in enumerate(DIGITS.items()):
            patch_text = self.ax_digit_selector.text(
                0.5, 1 - (padding + (idigit + 0.5) * height), name,
                color='k', ha='center', va='center')
            ds_patch = patches.Rectangle(
                (0, 1 - (padding + (idigit + 1) * height)), 1, height,
                color=micolors[d['c']][100])
            self.ax_digit_selector.add_patch(ds_patch)
            self.digit_selectors.append({
                'name': name,
                'patch': ds_patch,
                'patch_text': patch_text,
                'c': micolors[d['c']],
                'exp': d['exp']})

    def make_iframe_highlights(self):
        self.tf_highlight = patches.Rectangle(
            (self.dt_offset_times[0],
             0), self.dt_visualize, self.max_sensor_total,
            color=micolors['green'][600], alpha=0.4, zorder=np.inf)
        self.ax_tf.add_patch(self.tf_highlight)
        self.mf_highlight = patches.Rectangle(
            (self.dt_offset_times[0],
             0), self.dt_visualize, self.max_sensor_total,
            color=micolors['green'][600], alpha=0.4, zorder=np.inf)
        self.ax_mf.add_patch(self.mf_highlight)

    def make_saveload_btns(self):
        self.ax_saveload.set_xlim([0, 1])
        self.ax_saveload.set_ylim([0, 1])
        plt.sca(self.ax_saveload)
        plt.axis('off')

        padding = 0.01
        self.ax_saveload.text(
            0.5, 0.75 - padding, 'SAVE',
            color='k', ha='center', va='center', size='x-large')
        self.savebtn = patches.Rectangle(
            (0, 0.5), 1, 0.5 - padding,
            color=micolors['green'][400])
        self.ax_saveload.add_patch(self.savebtn)
        self.ax_saveload.text(
            0.5, 0.25, 'LOAD',
            color='k', ha='center', va='center', size='x-large')
        self.loadbtn = patches.Rectangle(
            (0, padding), 1, 0.5-padding,
            color=micolors['red'][400])
        self.ax_saveload.add_patch(self.loadbtn)

    def display_ps_frame(self, i_frame=None):
        print('Loading pressure sensor data...', end='')
        # self.display_load_msg()
        if i_frame is None:
            i_frame = self.i_frame
        if self.lps_image is not None:
            self.lps_image.remove()
        if self.rps_image is not None:
            self.rps_image.remove()
        matrix = self.ps_matrices[LPS_NAME][i_frame] / \
            self.ps_vmax_d[LPS_NAME]
        self.lps_image = self.ax_lps.imshow(self.force_map_transform(matrix),
                                            vmin=0, vmax=1, cmap='Greys')
        if self.show_automatic:
            if self.lps_auto_image is not None:
                self.lps_auto_image.remove()
            self.lps_auto_image = self.ax_lps_auto.imshow(self.force_map_transform(matrix),
                                                          vmin=0, vmax=1, cmap='Greys')
        matrix = self.ps_matrices[RPS_NAME][i_frame] / \
            self.ps_vmax_d[RPS_NAME]
        self.rps_image = self.ax_rps.imshow(self.force_map_transform(matrix),
                                            vmin=0, vmax=1, cmap='Greys')
        if self.show_automatic:
            if self.rps_auto_image is not None:
                self.rps_auto_image.remove()
            self.rps_auto_image = self.ax_rps_auto.imshow(self.force_map_transform(matrix),
                                                          vmin=0, vmax=1, cmap='Greys')

        self.fig.canvas.draw()
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
        cameraframe = int(round(
            (self.ps_times[i_frame] + self.camera_pstime_diff) * self.fps)) - 1
        cameraframe = max(0, cameraframe)
        cameraframe = min(cameraframe, self.num_frames-1)

        if self.lps_videoframe is not None:
            self.lps_videoframe.remove()
        if self.rps_videoframe is not None:
            self.rps_videoframe.remove()

        self.lps_videoframe = self.display_videoframe(
            self.lps_video, self.ax_lps_vid, cameraframe)
        self.rps_videoframe = self.display_videoframe(
            self.rps_video, self.ax_rps_vid, cameraframe)
        self.fig.canvas.draw()

    def display_digit_areas(self):
        if self.lps_colormask is not None:
            self.lps_colormask.remove()
        if self.rps_colormask is not None:
            self.rps_colormask.remove()
        if self.avg_lps_colormask is not None:
            self.avg_lps_colormask.remove()
        if self.avg_rps_colormask is not None:
            self.avg_rps_colormask.remove()
        lps_digit_color_mask = np.ones(
            (self.nsenselsr, self.nsenselsr, 3))
        rps_digit_color_mask = np.ones(
            (self.nsenselsr, self.nsenselsr, 3))
        for idigit, ds in enumerate(self.digit_selectors):
            if idigit == UNCLAIMED_INDEX:
                color = (1, 1, 1)  # white
            else:
                color = ds['c'][600]
            lpsdmask = self.lps_digit_mask == idigit
            lps_digit_color_mask[lpsdmask, 0] = color[0]
            lps_digit_color_mask[lpsdmask, 1] = color[1]
            lps_digit_color_mask[lpsdmask, 2] = color[2]
            rpsdmask = self.rps_digit_mask == idigit
            rps_digit_color_mask[rpsdmask, 0] = color[0]
            rps_digit_color_mask[rpsdmask, 1] = color[1]
            rps_digit_color_mask[rpsdmask, 2] = color[2]
        self.lps_colormask = self.ax_lps.imshow(
            lps_digit_color_mask, alpha=0.2)
        self.rps_colormask = self.ax_rps.imshow(
            rps_digit_color_mask, alpha=0.2)
        # add the same to average plot
        self.avg_lps_colormask = self.ax_lps_avg.imshow(
            lps_digit_color_mask, alpha=0.2)
        self.avg_rps_colormask = self.ax_rps_avg.imshow(
            rps_digit_color_mask, alpha=0.2)

        if self.show_automatic:
            if self.lps_auto_colormask is not None:
                self.lps_auto_colormask.remove()
            if self.rps_auto_colormask is not None:
                self.rps_auto_colormask.remove()
            # show auto stuff
            lps_digit_color_mask = np.ones(
                (self.nsenselsr, self.nsenselsr, 3))
            rps_digit_color_mask = np.ones(
                (self.nsenselsr, self.nsenselsr, 3))

            for idigit, ds in enumerate(self.digit_selectors):
                if idigit == UNCLAIMED_INDEX:
                    continue
                else:
                    color = ds['c'][600]
                lpsdmask = tools.get_matched_contact_frame_mask(
                    ds['exp'], self.matched_contacts[LPS_NAME][self.i_frame],
                    (self.nsenselsr, self.nsenselsr))
                lps_digit_color_mask[lpsdmask, 0] = color[0]
                lps_digit_color_mask[lpsdmask, 1] = color[1]
                lps_digit_color_mask[lpsdmask, 2] = color[2]

                rpsdmask = tools.get_matched_contact_frame_mask(
                    ds['exp'], self.matched_contacts[RPS_NAME][self.i_frame],
                    (self.nsenselsr, self.nsenselsr))
                rps_digit_color_mask[rpsdmask, 0] = color[0]
                rps_digit_color_mask[rpsdmask, 1] = color[1]
                rps_digit_color_mask[rpsdmask, 2] = color[2]
            self.lps_auto_colormask = self.ax_lps_auto.imshow(
                lps_digit_color_mask, alpha=0.6)
            self.rps_auto_colormask = self.ax_rps_auto.imshow(
                rps_digit_color_mask, alpha=0.6)

        self.fig.canvas.draw()

    def update_vertical_lines(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        self.tf_highlight.set_x(self.dt_offset_times[i_frame])
        self.mf_highlight.set_x(self.dt_offset_times[i_frame])
        self.fig.canvas.draw()

    def select_digit(self, idigit):
        # de highlight
        if self.selected_digit is not None:
            if idigit == self.selected_digit:  # already
                return
            self.digit_selectors[self.selected_digit]['patch'].set_color(
                self.digit_selectors[self.selected_digit]['c'][100])
        # highlight
        self.selected_digit = idigit
        self.digit_selectors[self.selected_digit]['patch'].set_color(
            self.digit_selectors[self.selected_digit]['c'][600])
        self.fig.canvas.draw()

    def calculate_digit_portions(self):
        self.total_force_traces = []
        for idigit, ds in enumerate(self.digit_selectors):
            self.total_force_traces.append(
                np.sum(self.ps_matrices[LPS_NAME],
                       axis=(1, 2), where=self.lps_digit_mask == idigit) +
                np.sum(self.ps_matrices[RPS_NAME],
                       axis=(1, 2), where=self.rps_digit_mask == idigit))

    def plot_digit_portions(self):
        # can only be called after calculate_digit_portions
        for _ in range(len(self.ax_mf.lines)):
            self.ax_mf.lines.pop(0)
        for tft, ds in zip(self.total_force_traces, self.digit_selectors):
            self.ax_mf.plot(self.ps_times, tft,
                            label=ds['name'], color=ds['c'][600])

        # display info about portion unset
        portion_unset = np.sum(
            self.total_force_traces[UNCLAIMED_INDEX]) / self.sum_sensor_total
        self.digit_selectors[UNCLAIMED_INDEX]['patch_text'].set_text('{} ({:.2%})'.format(
            UNCLAIMED_NAME, portion_unset))

        if self.show_automatic:
            for _ in range(len(self.ax_mf_diff.lines)):
                self.ax_mf_diff.lines.pop(0)
            for tft, ds in zip(self.total_force_traces, self.digit_selectors):
                if ds['name'] == UNCLAIMED_NAME:
                    continue
                self.ax_mf_diff.plot(
                    self.ps_times, abs(
                        tft - self.total_auto_force_traces[ds['name']]),
                    label=ds['name'], color=ds['c'][600])

        self.fig.canvas.draw()

    def display_load_msg(self):
        # doesn't actually do anything - too fast
        self.lps_text = self.ax_lps.text(
            self.nsenselsr / 2, self.nsenselsr / 2, 'LOADING',
            color=micolors['red'][600], fontweight='bold', ha='center')
        self.rps_text = self.ax_rps.text(
            self.nsenselsr / 2, self.nsenselsr / 2, 'LOADING',
            color=micolors['red'][600], fontweight='bold', ha='center')
        self.fig.canvas.draw()

    def clear_ps_msg(self):
        self.lps_text.remove()
        self.rps_text.remove()
        self.fig.canvas.draw()

    def export_matched_data(self):
        column_names = ['time'] + [ds['name']
                                   for ds in self.digit_selectors]
        values = [self.ps_times] + self.total_force_traces
        io_tools.export_csv(
            self.trial.manually_labelled_filename, column_names, values)
        print('Exported matched profiles to {}.'.format(
            self.trial.manually_labelled_filename))

        io_tools.export_one_csv_matrix(
            self.trial.lps_map_filename, self.lps_digit_mask)
        io_tools.export_one_csv_matrix(
            self.trial.rps_map_filename, self.rps_digit_mask)
        print('Exported left and right pressure sensor masks into {} and {}, respectively.'.format(
            self.trial.lps_map_filename, self.trial.rps_map_filename))

    def load_maps(self):
        if os.path.exists(self.trial.lps_map_filename):
            self.lps_digit_mask = np.array(
                io_tools.import_one_csv_matrix(self.trial.lps_map_filename, dtype=int))
            print('Loaded left pressure sensor map from {}.'.format(
                self.trial.lps_map_filename))
        if os.path.exists(self.trial.rps_map_filename):
            self.rps_digit_mask = np.array(
                io_tools.import_one_csv_matrix(self.trial.rps_map_filename, dtype=int))
            print('Loaded right pressure sensor map from {}.'.format(
                self.trial.rps_map_filename))

    def find_square_from_events(self, event1, event2):
        for i_iso, iso in enumerate(self.isensels_offset_x):
            if abs(iso - event1.xdata) <= 0.5:
                isen_st_x = i_iso
            if abs(iso - event2.xdata) <= 0.5:
                isen_en_x = i_iso
        for i_iso, iso in enumerate(self.isensels_offset_y):
            if abs(iso - event1.ydata) <= 0.5:
                isen_st_y = i_iso
            if abs(iso - event2.ydata) <= 0.5:
                isen_en_y = i_iso
        if isen_st_x > isen_en_x:
            isen_st_x, isen_en_x = isen_en_x, isen_st_x
        if isen_st_y > isen_en_y:
            isen_st_y, isen_en_y = isen_en_y, isen_st_y
        return isen_st_x, isen_en_x, isen_st_y, isen_en_y

    def update_on_frame_change(self):
        self.update_vertical_lines()
        self.display_ps_frame()
        self.display_digit_areas()
        self.display_videos_frame()

    def on_press(self, event):
        if (event.inaxes == self.ax_tf and self.tf_highlight.contains(event)[0]) or (
                event.inaxes == self.ax_mf and self.mf_highlight.contains(event)[0]):
            self.highlight_pressevent = event
        elif event.inaxes == self.ax_digit_selector:
            for idigit, ds in enumerate(self.digit_selectors):
                if ds['patch'].contains(event)[0]:
                    self.press_selected_digit = idigit
                    break
        elif event.inaxes == self.ax_lps and self.selected_digit is not None:
            self.lps_highlight_pressevent = event
        elif event.inaxes == self.ax_rps and self.selected_digit is not None:
            self.rps_highlight_pressevent = event
        elif event.inaxes == self.ax_saveload and self.savebtn.contains(event)[0]:
            self.savebtn_pressevent = event
        elif event.inaxes == self.ax_saveload and self.loadbtn.contains(event)[0]:
            self.loadbtn_pressevent = event
        elif event.inaxes == self.ax_lps_avg:
            self.lps_avg_pressevent = event
        elif event.inaxes == self.ax_rps_avg:
            self.rps_avg_pressevent = event

    def on_release(self, event):
        if self.highlight_pressevent is not None:
            self.i_frame = self.moving_i_frame
            self.display_ps_frame()
            self.display_digit_areas()
            self.display_videos_frame()
            self.highlight_pressevent = None
        if self.press_selected_digit is not None:
            if event.inaxes == self.ax_digit_selector:
                for idigit, ds in enumerate(self.digit_selectors):
                    if ds['patch'].contains(event)[0]:
                        # if same place where pressed down
                        if idigit == self.press_selected_digit:
                            self.select_digit(idigit)
                        break
            self.press_selected_digit = None
        if self.lps_highlight_pressevent is not None:
            if event.inaxes == self.ax_lps:
                # find all sensels within the between start and end clicks
                isen_st_x, isen_en_x, isen_st_y, isen_en_y = self.find_square_from_events(
                    self.lps_highlight_pressevent, event)
                for isen_x, isen_y in product(range(isen_st_x, isen_en_x+1),
                                              range(isen_st_y, isen_en_y+1)):
                    self.lps_digit_mask[isen_y,
                                        isen_x] = self.selected_digit
                self.display_digit_areas()
                self.calculate_digit_portions()
                self.plot_digit_portions()
            self.lps_highlight_pressevent = None
        if self.rps_highlight_pressevent is not None:
            if event.inaxes == self.ax_rps:
                # find all sensels within the between start and end clicks
                isen_st_x, isen_en_x, isen_st_y, isen_en_y = self.find_square_from_events(
                    self.rps_highlight_pressevent, event)
                for isen_x, isen_y in product(range(isen_st_x, isen_en_x+1),
                                              range(isen_st_y, isen_en_y+1)):
                    self.rps_digit_mask[isen_y,
                                        isen_x] = self.selected_digit
                self.display_digit_areas()
                self.calculate_digit_portions()
                self.plot_digit_portions()
            self.rps_highlight_pressevent = None
        if self.savebtn_pressevent is not None:
            if event.inaxes == self.ax_saveload and self.savebtn.contains(event)[0]:
                self.export_matched_data()
            self.savebtn_pressevent = None
        if self.loadbtn_pressevent is not None:
            if event.inaxes == self.ax_saveload and self.loadbtn.contains(event)[0]:
                self.load_maps()
                self.display_digit_areas()
                self.calculate_digit_portions()
                self.plot_digit_portions()
            self.loadbtn_pressevent = None
        if self.lps_avg_pressevent is not None:
            if event.inaxes == self.ax_lps_avg:
                # find the time index associated with the maximum value at this element
                x = int(event.xdata)
                y = int(event.ydata)
                sensel = self.ps_matrices[LPS_NAME][:, y, x]
                senselmax = np.max(sensel)
                if senselmax > 0:
                    i_frame = tools.find_first(
                        sensel >= np.max(sensel))
                    self.i_frame = i_frame
                    self.update_on_frame_change()
            self.lps_avg_pressevent = None
        if self.rps_avg_pressevent is not None:
            if event.inaxes == self.ax_rps_avg:
                # find the time index associated with the maximum value at this element
                x = int(event.xdata)
                y = int(event.ydata)
                sensel = self.ps_matrices[RPS_NAME][:, y, x]
                senselmax = np.max(sensel)
                if senselmax > 0:
                    i_frame = tools.find_first(
                        sensel >= np.max(sensel))
                    self.i_frame = i_frame
                    self.update_on_frame_change()
            self.rps_avg_pressevent = None

    def on_move(self, event):
        if (self.highlight_pressevent is not None and
                event.inaxes == self.highlight_pressevent.inaxes):
            dx = event.xdata - self.highlight_pressevent.xdata
            new_x = self.dt_offset_times[self.i_frame] + dx
            self.moving_i_frame = tools.find_first(
                new_x <= self.dt_offset_times)
            self.update_vertical_lines(self.moving_i_frame)

    def on_key_press(self, event):
        update = False
        if self.highlight_pressevent is None:
            if event.key == 'left' and self.i_frame > 0:
                self.i_frame -= 1
                update = True
            elif event.key == 'shift+left':
                if self.i_frame > SHIFT_JUMP - 1:
                    self.i_frame -= SHIFT_JUMP
                    update = True
                elif self.i_frame > 0:
                    self.i_frame -= 1
                    update = True
            elif event.key == 'right' and self.i_frame + 1 < len(self.dt_offset_times):
                self.i_frame += 1
                update = True
            elif event.key == 'shift+right':
                if self.i_frame + SHIFT_JUMP < len(self.dt_offset_times):
                    self.i_frame += SHIFT_JUMP
                    update = True
                elif self.i_frame + 1 < len(self.dt_offset_times):
                    self.i_frame += 1
                    update = True
        if event.key == 'shift':
            self.shift = True
        if event.key == ' ' or event.key == 's' or event.key == 'ctrl+s':
            self.export_matched_data()
        if update:
            self.update_on_frame_change()

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift = False


def manual_force_labeling(trial, lps_ref_video_filename, rps_ref_video_filename, show_automatic):
    # load videos
    lps_video = cv2.VideoCapture(lps_ref_video_filename)
    fps = int(lps_video.get(cv2.CAP_PROP_FPS))
    num_frames = int(lps_video.get(cv2.CAP_PROP_FRAME_COUNT))
    rps_video = cv2.VideoCapture(rps_ref_video_filename)
    if fps != int(rps_video.get(cv2.CAP_PROP_FPS)):
        raise ValueError('Videos FPS do not match.')
    if num_frames != int(rps_video.get(cv2.CAP_PROP_FRAME_COUNT)):
        warnings.warn(
            'Total number of frames do not match. Truncating to shortest.')
        num_frames = min(num_frames, int(
            rps_video.get(cv2.CAP_PROP_FRAME_COUNT)))

    figsize = (16, 9)

    eh = ForceLabellingInterface(figsize, trial,
                                 lps_video, rps_video, fps, num_frames, show_automatic)

    # need to keep it in the workspace
    plt.show()


def manually_label_forces(server, session, trial_number, temp,
                          lps_ref_camera_name, rps_ref_camera_name, show_automatic):
    """Opens a GUI to manually assign sensels to digits.

    Arguments:
        server {str} --- Folder where the sessions are located.
        session {str} --- Session directory to use.
        trial_number {int} --- Trial to do adjustment on.
        lps_ref_camera_name {str} --- Camera name.
        rps_ref_camera_name {str} --- Camera name.
        show_automatic {bool} ---
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(session) == 0:
        session = meta_session.find_session_dirs(server)[0]

    rs('Processing session {}.'.format(session))
    server_session = os.path.join(server, session)

    if not os.path.exists(server_session):
        ValueError(
            'Session {} does not exist on the server.'.format(session))

    # load session meta
    mstruct, _, _, msession = meta_session.load_meta_information(
        server_session)

    # find trial
    trial = None
    for t in msession:
        if t.trial_number == trial_number:
            trial = t
            break
    if trial is None:
        ValueError('Could not find trial #{}.'.format(trial_number))
    if not trial.do_post_ps_files_exist():
        raise ValueError(
            'Associated processed pressure files do not exist.')

    os.makedirs(
        mstruct['manually_labelled_forces_dir'], exist_ok=True)

    lps_ref_video_filename = os.path.join(
        mstruct['videos_dir'],
        mstruct['kin_trialname_template'].format(
            trial_number=trial_number),
        lps_ref_camera_name + '.mp4')
    rps_ref_video_filename = os.path.join(
        mstruct['videos_dir'],
        mstruct['kin_trialname_template'].format(
            trial_number=trial_number),
        rps_ref_camera_name + '.mp4')

    manual_force_labeling(
        trial, lps_ref_video_filename, rps_ref_video_filename, show_automatic)
