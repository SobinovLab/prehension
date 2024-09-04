#!python3.7
import itertools
import json
import os
import random
import warnings

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tqdm

import ncams
from . import meta_session
from . import preset
from . import tools
from .materialsio_colors import materialsio_colors_rgb as micolors
from .tools import rs, ws
from .triangulate import rotation_vector

MARKERS = (
    'left_clavicle_distal', 'left_clavicle_proximal',
    'right_clavicle_proximal', 'right_clavicle_distal',
    'mid_sternum', 'bottom_sternum')
UNSELECTED_ALPHA = 0.2
SELECTED_ALPHA = 0.8
VIDEO_MARKER_SIZE = 50
ZOOM_VIDEO_MARKER_SIZE = 150
DELETE_MARKER_RADIUS = 20
VIDEO_MARKER_ALPHA = 0.5

THORAX_DOF_NAMES = ('Thorax_tra1', 'Thorax_tra2', 'Thorax_tra3',
                    'Thorax_rot1', 'Thorax_rot2', 'Thorax_rot3')


class VideoViewInterface:
    def __init__(self, figsize, trial, mstruct, default_base_spec_filename):
        self.fig = plt.figure(figsize=figsize)
        self.fig.canvas.manager.set_window_title(
            'Video View Trial #{}'.format(trial.trial_number))

        self.trial = trial
        self.mstruct = mstruct
        self.default_base_spec_filename = default_base_spec_filename
        self.videosframe = {}

        self.load_videos()
        self.i_frame = int(0.05 * self.num_frames)
        self.zoom_camera = self.cameras[0]  # zoom onto the first camera by default

        # create axes
        self.setup_video_axes()
        self.setup_zoom_video_axis()

        # add slider
        self.setup_frame_slider()

        # add buttons
        self.setup_control_buttons()

        # add marker buttons
        self.selected_marker = None
        self.setup_marker_buttons(MARKERS)
        self.create_marker_data()
        self.load_marker_data()
        self.selecting_zoom_left = False
        self.selecting_zoom_right = False

        self.display_videos_frame()

        # create callbacks
        self.selecting_camera = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('close_event', self.on_close)

        # call plot show
        plt.show()

    def load_videos(self):
        if not self.trial.do_videos_files_exist():
            raise ValueError('Trial videos missing.')

        # to sort
        # just ease of use
        self.cameras = sorted(self.trial.videos.keys())

        self.videos = {}
        self.num_frames = np.nan
        for camera in self.cameras:
            self.videos[camera] = cv2.VideoCapture(self.trial.videos[camera])
            video_nframes = int(self.videos[camera].get(cv2.CAP_PROP_FRAME_COUNT))
            if np.isnan(self.num_frames):
                self.num_frames = video_nframes
            elif self.num_frames != video_nframes:
                ws('Number of frames in videos does not match, truncating to smallest.')
                self.num_frames = min(self.num_frames, video_nframes)

    @staticmethod
    def calculate_axes_tiling(border, padding, n_rows, n_cols=None, n_tot=None):
        '''Either n_cols or n_tot need to be specified. n_cols takes priority'''
        (x_padding, y_padding) = padding
        if n_cols is None:
            n_cols = int(np.ceil(n_tot / n_rows))

        # border: <left>, <bottom>, <right>, <top>
        x_size = (1 - border[0] - border[2]) / n_cols
        x_size_int = x_size - 2 * x_padding
        y_size = (1 - border[1] - border[3]) / n_rows
        y_size_int = y_size - 2 * y_padding

        axes_pos = []
        for i_row, i_col in itertools.product(range(n_rows), range(n_cols)):
            axes_pos.append([
                border[0] + x_size * i_col + x_padding,
                border[1] + y_size * (n_rows - i_row - 1) + y_padding,
                x_size_int, y_size_int])

        return axes_pos

    def setup_video_axes(self):
        axes_pos = self.calculate_axes_tiling(
            [0.6, 0.1, 0, 0], (0.001, 0.002), 4, n_tot=len(self.videos))

        self.video_axes = {}
        for camera, axis_pos in zip(self.cameras, axes_pos):
            ax = self.fig.add_axes(axis_pos)
            ax.set_xticks([])
            ax.set_yticks([])

            ax.text(0.5, 0.92, camera,
                    color='k', ha='center', va='center', transform=ax.transAxes,
                    zorder=np.inf,
                    bbox={'boxstyle': 'round', 'fc': (1, 1, 1, 0.75), 'ec': (1, 1, 1, 0.75)})

            self.video_axes[camera] = ax

    def setup_zoom_video_axis(self):
        # <left>, <bottom>, <right>, <top>
        border = [0, 0.1, 0.4, 0.05]
        x_padding = 0.001
        y_padding = 0.002

        ax = self.fig.add_axes([
            border[0] + x_padding,
            border[1] + y_padding,
            1 - border[2] - border[0] - 2 * x_padding,
            1 - border[3] - border[1] - 2 * y_padding])
        ax.set_xticks([])
        ax.set_yticks([])

        self.zoom_video_ax = ax
        self.zoom_video_frame = None
        self.zoom_video_text = ax.text(
            0.5, 1.025, '',
            color='k', size='x-large', ha='center', va='center',
            transform=ax.transAxes)

    @staticmethod
    def display_videoframe(video, ax, cameraframe):
        video.set(cv2.CAP_PROP_POS_FRAMES, cameraframe)
        fe, frame = video.read()
        if fe is False:
            ws('Could not read the frame #{}.'.format(cameraframe))
            return
        frame_rgb = frame[..., ::-1].copy()
        return ax.imshow(frame_rgb)

    def draw_zoom(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        if self.zoom_video_frame is not None:
            self.zoom_video_frame.remove()

        # display video and title
        self.zoom_video_frame = self.display_videoframe(
            self.videos[self.zoom_camera], self.zoom_video_ax, i_frame)

        self.zoom_video_text.set_text(self.zoom_camera)

        self.fig.canvas.draw_idle()

    def display_videos_frame(self, i_frame=None):
        if i_frame is None:
            i_frame = self.i_frame
        i_frame = max(0, i_frame)
        i_frame = min(i_frame, self.num_frames-1)

        # draw side
        for camera, ax in self.video_axes.items():
            if self.videosframe.get(camera, None) is not None:
                self.videosframe[camera].remove()

            self.videosframe[camera] = self.display_videoframe(
                self.videos[camera], ax, i_frame)

        # draw zoom
        self.draw_zoom(i_frame=i_frame)

        self.fig.canvas.draw_idle()

    def update_frame_by_slider(self, i_frame):
        self.i_frame = int(i_frame)  # slider can return float
        self.display_videos_frame()

    def setup_frame_slider(self):
        self.frame_slider_ax = self.fig.add_axes([0.65, 0.05, 0.3, 0.025])
        self.frame_slider = mpl.widgets.Slider(
            ax=self.frame_slider_ax,
            label='Frame #',
            valmin=0,
            valmax=self.num_frames-1,
            valinit=self.i_frame,
            valfmt='%d',
            dragging=False,
            valstep=1
        )

        self.frame_slider.on_changed(self.update_frame_by_slider)

    def on_load_btn_press(self, event):
        self.load_marker_data()

    def on_save_btn_press(self, event):
        self.save_marker_data()

    def clear_zoom_camera_markers(self):
        for marker in self.markers:
            if self.marker_data[self.zoom_camera][marker] is not None:
                self.zoom_marker_points[marker].remove()
        self.fig.canvas.draw_idle()

    def clear_camera_markers(self, camera):
        for marker in self.markers:
            if self.marker_data[camera][marker] is not None:
                self.marker_points[camera][marker].remove()
                self.marker_data[camera][marker] = None
        self.fig.canvas.draw_idle()

    def remove_marker(self, camera, marker):
        if self.marker_data[camera][marker] is None:
            return

        self.marker_data[camera][marker] = None
        if camera == self.zoom_camera:
            self.zoom_marker_points[marker].remove()
        self.marker_points[camera][marker].remove()

        self.fig.canvas.draw_idle()

    def clear_markers(self):
        self.clear_zoom_camera_markers()
        for camera in self.cameras:
            self.clear_camera_markers(camera)
        self.update_marker_counters()
        self.fig.canvas.draw_idle()

    def on_clear_btn_press(self, event):
        self.clear_markers()

    def make_this_trial_default(self):
        with open(self.default_base_spec_filename, 'w') as f:
            json.dump({'session': self.trial.session,
                       'trial_number': self.trial.trial_number}, f, indent=4)
        rs('Made this {} session and {} trial default for calibration in {}.'.format(
            self.trial.session, self.trial.trial_number, self.default_base_spec_filename))

    def on_make_default_btn_press(self, event):
        self.make_this_trial_default()

    def check_default_for_matching(self):
        # uses input from command line, better call on closed matplotlib window
        # check existence of default file
        if not os.path.exists(self.default_base_spec_filename):
            answ = input(
                'Base session and trial are not specified for this calibration, do you'
                ' want to make this ({}: {}) the default? (yes/no) '.format(
                    self.trial.session, self.trial.trial_number))
            if answ in ('yes', 'y'):
                self.make_this_trial_default()
        else:
            # check if the default session-trial are the current
            with open(self.default_base_spec_filename, 'r') as f:
                default = json.load(f)
            if (default['session'] != self.trial.session or
                    default['trial_number'] != self.trial.trial_number):
                answ = input(
                    'A different session and trial are saved as default base session'
                    ' and trial ({}: {}), do you want to make this session and trial '
                    '({}: {}) default? (yes/no) '.format(
                        default['session'], default['trial_number'],
                        self.trial.session, self.trial.trial_number))
                if answ in ('yes', 'y'):
                    self.make_this_trial_default()

    def on_close(self, event):
        # self.check_default_for_matching() -- implementing dialogue window in matplotlib is a
        # giant pain
        pass

    def setup_control_buttons(self):
        # load save clear make default
        nbuttons = 4  # in a row
        position = [0.65, 0.01, 0.3, 0.04]

        positions = []
        for i in range(nbuttons):
            p = list(position)
            p[0] += i * p[2] / nbuttons
            p[2] /= nbuttons
            positions.append(p)

        color = micolors['grey'][200]

        self.load_btn_ax = self.fig.add_axes(positions[0])
        self.load_btn = mpl.widgets.Button(
            self.load_btn_ax, 'Load',
            color=color, hovercolor=color)
        self.load_btn.on_clicked(self.on_load_btn_press)

        self.save_btn_ax = self.fig.add_axes(positions[1])
        self.save_btn = mpl.widgets.Button(
            self.save_btn_ax, 'Save',
            color=color, hovercolor=color)
        self.save_btn.on_clicked(self.on_save_btn_press)

        self.clear_btn_ax = self.fig.add_axes(positions[2])
        self.clear_btn = mpl.widgets.Button(
            self.clear_btn_ax, 'Clear',
            color=color, hovercolor=color)
        self.clear_btn.on_clicked(self.on_clear_btn_press)

        self.make_default_btn_ax = self.fig.add_axes(positions[3])
        self.make_default_btn = mpl.widgets.Button(
            self.make_default_btn_ax, 'Make default',
            color=color, hovercolor=color)
        self.make_default_btn.on_clicked(self.on_make_default_btn_press)

    def select_marker(self, marker):
        # clicked the same one
        if marker == self.selected_marker:
            return

        # deselect previous
        if self.selected_marker is not None:
            color = list(self.marker_colors[self.selected_marker])
            color[3] = UNSELECTED_ALPHA
            self.marker_patches[self.selected_marker].set_facecolor(color)

        self.selected_marker = marker
        color = list(self.marker_colors[marker])
        color[3] = SELECTED_ALPHA
        self.marker_patches[marker].set_facecolor(color)

        self.fig.canvas.draw_idle()

    def update_marker_counters(self):
        for marker in self.markers:
            n_success = sum([int(self.marker_data[camera][marker] is not None)
                             for camera in self.cameras])
            self.marker_counters[marker].set_text('{}/{}'.format(n_success, len(self.cameras)))
            if n_success < 2:
                self.marker_counters[marker].set_color(micolors['red'][500])
            else:
                self.marker_counters[marker].set_color('k')
        self.fig.canvas.draw_idle()

    def setup_marker_buttons(self, markers):
        self.markers = markers
        self.marker_colors = {}
        self.marker_patches = {}
        self.marker_counters = {}
        self.selecting_marker = None

        # turbo
        cmap = mpl.colormaps['turbo']

        self.marker_area_ax = self.fig.add_axes(self.calculate_axes_tiling(
            [0., 0, 0.4, 0.9], (0., 0.), 1, n_cols=1)[0])
        self.marker_area_ax.set_xlim([0, 1])
        self.marker_area_ax.set_ylim([0, 1])
        self.marker_area_ax.set_xticks([])
        self.marker_area_ax.set_yticks([])
        plt.axis('off')
        axes_pos = self.calculate_axes_tiling(
            [0, 0, 0, 0], (0.01, 0.02), 2, n_tot=len(self.markers))

        for i_marker, (marker, axis_pos) in enumerate(zip(self.markers, axes_pos)):
            # choose color
            color = cmap(i_marker / (len(self.markers) - 1))
            self.marker_colors[marker] = color
            color = list(self.marker_colors[marker])
            color[3] = VIDEO_MARKER_ALPHA

            # make btn
            # note: widget buttons suck and are slow
            self.marker_area_ax.text(
                axis_pos[0] + axis_pos[2] / 2, axis_pos[1] + axis_pos[3] / 2, marker,
                color='k', ha='center', va='center', size='x-large')
            self.marker_counters[marker] = self.marker_area_ax.text(
                axis_pos[0] + axis_pos[2] * 0.98, axis_pos[1] + axis_pos[3] * 0.8, '?/?',
                color='k', ha='right', va='center', size='medium')
            self.marker_patches[marker] = mpl.patches.Rectangle(
                axis_pos[0:2], axis_pos[2], axis_pos[3],
                color=color)
            self.marker_area_ax.add_patch(self.marker_patches[marker])

    def create_marker_data(self):
        self.marker_data = {c: {m: None for m in self.markers} for c in self.cameras}
        self.marker_points = {c: {m: None for m in self.markers} for c in self.cameras}
        self.zoom_marker_points = {m: None for m in self.markers}

    def show_camera_markers(self, camera):
        for marker in self.markers:
            p = self.marker_data[camera][marker]
            if p is not None:
                color = list(self.marker_colors[marker])
                color[3] = SELECTED_ALPHA
                self.marker_points[camera][marker] = self.video_axes[camera].scatter(
                    p[0], p[1], s=VIDEO_MARKER_SIZE, c=color, zorder=np.inf)
        self.fig.canvas.draw_idle()

    def show_zoom_camera_markers(self):
        for marker in self.markers:
            p = self.marker_data[self.zoom_camera][marker]
            if p is not None:
                color = list(self.marker_colors[marker])
                color[3] = SELECTED_ALPHA
                self.zoom_marker_points[marker] = self.zoom_video_ax.scatter(
                    p[0], p[1], s=ZOOM_VIDEO_MARKER_SIZE, c=color, zorder=np.inf)
        self.fig.canvas.draw_idle()

    def show_markers(self):
        self.show_zoom_camera_markers()
        for camera in self.cameras:
            self.show_camera_markers(camera)
        self.update_marker_counters()
        self.fig.canvas.draw_idle()

    def load_marker_data(self):
        self.clear_markers()

        # actually load
        if self.trial.calib_base_marker_filename is None:
            warnings.warn('Base marker filename is None')
            return

        if not os.path.exists(self.trial.calib_base_marker_filename):
            return

        with open(self.trial.calib_base_marker_filename, 'r') as f:
            self.marker_data = json.load(f)
        self.show_markers()

    def save_marker_data(self):
        if self.trial.calib_base_marker_filename is None:
            warnings.warn('Base marker filename is None')
            return

        os.makedirs(os.path.split(self.trial.calib_base_marker_filename)[0], exist_ok=True)

        with open(self.trial.calib_base_marker_filename, 'w') as f:
            json.dump(self.marker_data, f, indent=4)

    def on_press(self, event):
        if event.inaxes in self.video_axes.values():
            self.selecting_camera = list(self.video_axes.keys())[
                list(self.video_axes.values()).index(event.inaxes)]
        elif event.inaxes == self.marker_area_ax:
            for marker, p in self.marker_patches.items():
                if p.contains(event)[0]:
                    self.selecting_marker = marker
                    break
        elif event.inaxes == self.zoom_video_ax:
            if event.button == mpl.backend_bases.MouseButton.LEFT:
                self.selecting_zoom_left = True
            elif event.button == mpl.backend_bases.MouseButton.RIGHT:
                self.selecting_zoom_right = True

    def on_release(self, event):
        if (self.selecting_camera is not None and
                event.inaxes == self.video_axes[self.selecting_camera]):
            self.clear_zoom_camera_markers()
            self.zoom_camera = self.selecting_camera
            self.draw_zoom()
            self.show_zoom_camera_markers()
        elif event.inaxes == self.marker_area_ax:
            for marker, p in self.marker_patches.items():
                if p.contains(event)[0]:
                    if self.selecting_marker == marker:
                        self.select_marker(marker)
                    break
        elif event.inaxes == self.zoom_video_ax:
            if self.selected_marker is not None and self.selecting_zoom_left:
                # add a point to storage, zoom video and sidebar videos
                zc = self.zoom_camera
                sm = self.selected_marker
                color = list(self.marker_colors[sm])
                color[3] = SELECTED_ALPHA

                # remove if the point was already there
                if self.marker_data[zc][sm] is not None:
                    self.zoom_marker_points[sm].remove()
                    self.marker_points[zc][sm].remove()
                # add to storage
                self.marker_data[zc][sm] = (event.xdata, event.ydata)
                # plot
                self.zoom_marker_points[sm] = self.zoom_video_ax.scatter(
                    event.xdata, event.ydata, s=ZOOM_VIDEO_MARKER_SIZE, c=color, zorder=np.inf)
                self.marker_points[zc][sm] = self.video_axes[zc].scatter(
                    event.xdata, event.ydata, s=VIDEO_MARKER_SIZE, c=color, zorder=np.inf)
            elif self.selecting_zoom_right:
                # if in radius of a point, delete it
                zc = self.zoom_camera
                for marker in self.markers:
                    p = self.marker_data[zc][marker]
                    if (p is not None and
                            np.sqrt((event.xdata - p[0])**2 +
                                    (event.ydata - p[1])**2) < DELETE_MARKER_RADIUS):
                        self.remove_marker(zc, marker)
            self.update_marker_counters()

        self.selecting_camera = None
        self.selecting_marker = None
        self.selecting_zoom_left = False
        self.selecting_zoom_right = False

        self.fig.canvas.draw_idle()


def label_images(default_base_spec_filename, calibration, sessions, msessions, mstructs):
    if os.path.exists(default_base_spec_filename):
        with open(default_base_spec_filename, 'r') as f:
            default = json.load(f)
        # check if within the loaded lists
        if (default['session'] not in sessions or
                meta_session.find_trial(msessions[default['session']],
                                        default['trial_number']) is None):
            default = None
    else:
        default = None

    # choose session
    if len(sessions) > 1:
        for i_s, session in enumerate(sessions):
            rs('\t{}: {}'.format(i_s, session))
        if default is not None:
            ss = input('Choose number of session to use for marking (empty or "default": {},'
                       ' "random" or "?" for random): '.format(default['session']))
        else:
            ss = input('Choose number of session to use for marking (empty, "random" or "?"'
                       ' for random): ')
        if default and ss in ('', 'default'):
            session = default['session']
        else:
            # wipe the default to load
            default = None

            if ss in ('', 'default', '?'):
                session = random.choice(sessions)
            else:
                session = sessions[int(ss)]
    else:
        session = sessions[0]
    rs('Using session {}.'.format(session))

    # choose trial
    msession = msessions[session]
    if len(msession) > 1:
        rs('{} trials available: {}'.format(
            len(msession), ', '.join([str(t.trial_number) for t in msession])))
        if default is not None:
            st = input('Choose trial number to use for marking (empty or "default": {},'
                       ' "random" or "?" for random): '.format(default['trial_number']))
        else:
            st = input('Choose trial number to use for marking (empty, "random" or "?"'
                       ' for random): ')
        if default and st in ('', 'default'):
            trial = meta_session.find_trial(msession, default['trial_number'])
        else:
            # just to stay clean
            default = None

            if st in ('', 'default', '?'):
                trial = random.choice(msession)
            else:
                trial = meta_session.find_trial(msession, int(st))
    else:
        trial = msession[0]
    rs('Using trial {}.'.format(trial.trial_number))

    # Label position of the torso on a representative trial
    vvi = VideoViewInterface((16, 9), trial, mstructs[session], default_base_spec_filename)

    # make a default
    vvi.check_default_for_matching()


def json2ncams(marker_data, cameras=None):
    # bodyparts, num_frames = 1, image_coordinates, ic_confidences
    if cameras is None:
        cameras = list(marker_data.keys())

    bodyparts = list(marker_data[cameras[0]].keys())

    image_coordinates = []
    ic_confidences = []
    for camera in cameras:
        ico = np.empty((1, 2, len(bodyparts)))
        ico.fill(np.nan)
        icc = np.zeros((1, len(bodyparts)))
        for ibp, bp in enumerate(bodyparts):
            if marker_data[camera][bp] is not None:
                ico[0, :, ibp] = marker_data[camera][bp]
                icc[0, ibp] = 1
        image_coordinates.append(ico)
        ic_confidences.append(icc)

    return bodyparts, image_coordinates, ic_confidences


def triangulate(trial, calibration, mstruct):
    # load marker data
    with open(trial.calib_base_marker_filename, 'r') as f:
        marker_data = json.load(f)

    # load calibration
    ncams_config = ncams.camera_io.yaml_to_config(
        mstruct['ncams_config'], overwrite_setup_path=True)
    # check if local extrinsic config exists and if so use it
    local_extrinsic_calibration_filename = os.path.join(
        mstruct['calibration'], 'extrinsic', 'extrinsic_calib.pickle')
    if os.path.exists(local_extrinsic_calibration_filename):
        intrinsics_config = ncams.camera_io.import_intrinsics(ncams_config)
        extrinsics_config = ncams.camera_io.import_extrinsics(
            local_extrinsic_calibration_filename)
    else:
        intrinsics_config, extrinsics_config = ncams.camera_io.load_calibrations(ncams_config)
    cameras = [str(v) for v in ncams_config['serials']]

    # transform into NCams format
    bodyparts, image_coordinates, ic_confidences = json2ncams(
        marker_data, cameras=cameras)

    # apply
    triangulated_points = ncams.reconstruction.triangulate_points(
        ncams_config, intrinsics_config, extrinsics_config,
        bodyparts, 1, image_coordinates, ic_confidences,
        threshold=0.5, method='centroid', centroid_threshold=2.5)

    # reflect(?) and rotate
    reflect = mstruct['hand'] == 'left'
    if reflect:
        marker_name_dict = ncams.utils.dic_from_csv(
            os.path.join(os.path.split(mstruct['opensim_model'])[0], 'marker_meta_reflect.csv'),
            'sDlcMarker', 'sOpenSimMarker')
    else:
        marker_name_dict = ncams.utils.dic_from_csv(
            os.path.join(os.path.split(mstruct['opensim_model'])[0], 'marker_meta.csv'),
            'sDlcMarker', 'sOpenSimMarker')
    triangulated_points = np.swapaxes(triangulated_points, 1, 2)
    marker_names = []
    for ibp, bp in enumerate(bodyparts):
        triangulated_points[:, ibp, :] = rotation_vector(triangulated_points[:, ibp, :])
        marker_names.append(marker_name_dict[bp])
    if reflect:
        triangulated_points[:, :, 0] = - triangulated_points[:, :, 0]

    # save 3D points to file
    ncams.io_utils.export_trc(
        trial.calib_base_markers_3D_filename_trc, marker_names, triangulated_points.tolist(), 50)

    # and generate IK file
    ik_xml_str = ncams.inverse_kinematics.IK_XML_STR.format(
        model_file=mstruct['opensim_model'])
    ncams.inverse_kinematics.make_ik_file(
        trial.calib_base_ik_filename, ik_xml_str, {k: 1 for k in marker_names},
        trial.calib_base_markers_3D_filename_trc, trial.calib_base_kinematic_filename, [0, 0.02])


def mark_base(server, sessions, temp, overwrite, skip_gui):
    """Manually label points on macaque torso to find its location once per calibration.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        skip_gui {bool} --- Do not launch the GUI for labeling.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    calibrations = {}
    mstructs = {}
    msessions = {}
    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, mdof, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        if os.path.exists(os.path.join(
                mstruct['calibration'], 'extrinsic', 'extrinsic_calib.pickle')):
            calibration_name = mstruct['calibration']
        else:
            calibration_name = os.path.split(mstruct['ncams_config'])[0]
        if calibration_name not in calibrations.keys():
            calibrations[calibration_name] = []
        calibrations[calibration_name].append(session)

        mstructs[session] = mstruct
        msessions[session] = msession

    rs('Found {} calibrations:'.format(len(calibrations)))
    for calibration, sessions in calibrations.items():
        rs('\t{}:'.format(calibration))
        for session in sessions:
            rs('\t\t{}'.format(session))

    # TODO add check for overwrite
    # TODO this needs to be fixed to work with session-based calibrations

    rs('Processing calibrations.')
    for calibration, sessions in calibrations.items():
        rs('Calibration {}.'.format(calibration))

        # check if default spec file exists
        default_base_spec_filename = os.path.join(calibration, 'base', 'default_{}.json'.format(
            preset.CURRENT_PRESET))

        # deal with 2d marker positions and default choices
        if not skip_gui:
            label_images(default_base_spec_filename, calibration, sessions, msessions, mstructs)

        # load default marker data
        if os.path.exists(default_base_spec_filename):
            with open(default_base_spec_filename, 'r') as f:
                default = json.load(f)
        else:
            ws('Default session-trial specification file not found. Skipping this calibration.')
            continue

        if default['session'] not in sessions:
            ws('Default session is not in the list of provided sessions,'
               ' skipping this calibration.')
            continue
        msession = msessions[default['session']]
        trial = meta_session.find_trial(msession, default['trial_number'])
        if trial is None:
            ws('Default trial is not in the list of trials, skipping this calibration.')
            continue

        if not os.path.exists(trial.calib_base_marker_filename):
            ws('Default session-trial specification file not found. Skipping this calibration.')
            continue

        # (undistort) and triangulate, create relevant IK files
        triangulate(trial, calibration, mstruct)
        rs('Triangulated.')

        # run inverse kinematics
        # opensim needs to work in Python3.8
        command = 'py execute_opensim_ik.py {} {}'.format(
            trial.calib_base_ik_filename, trial.calib_base_ik_log_filename)
        ret = os.system(command)
        # process output as error throw
        if (ret < 0):
            ws('Inverse kinematic command returned with {} error message.'.format(ret))
            continue
        rs('Completed inverse kinematics.')

        # load resulting positions
        dof_names, _, dofs = ncams.io_utils.import_mot(
            trial.calib_base_kinematic_filename)
        positions = {}
        for dof_name, dof in zip(dof_names, dofs):
            if dof_name in THORAX_DOF_NAMES:
                positions[dof_name] = dof[0]
                if mdof[dof_name]['rot']:
                    positions[dof_name] *= np.pi / 180

        # export result into the OpenSim model for each session model
        for session in sessions:
            mstruct = mstructs[session]
            ncams.inverse_kinematics.set_opensim_model_default_position(
                mstruct['opensim_model'], mstruct['opensim_model_locked_base'], positions,
                lock=True)
            rs('Created locked model {}.'.format(mstruct['opensim_model_locked_base']))
