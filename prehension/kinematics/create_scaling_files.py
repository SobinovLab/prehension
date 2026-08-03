#!python3
# -*- coding: utf-8 -*-
"""
Creates IK and Scaling files for OpenSim based on a period of trial.

Copyright (C) 2023-2024 Anton Sobinov
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
import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.ndimage
import scipy.signal

from .. import meta_session
from ..tools import logs
from ..tools import io
from ..tools import opensim_io
from . import inverse_kinematics
from ..tools.materialsio_colors import materialsio_colors as micolors
from ..tools.logs import rs, ws

SEGMENT_BODY_GROUPS = {
    'chest': ('Thorax', 'RA_clavicle', 'RA_clavphant', 'RA_scapula', 'RA_scapphant'),
    'humerus': ('RA1H_PHANT', 'RA1H_PHANT1', 'RA1H'),
    'forearm': ('RA2U', 'RA2R'),
    'hand': ('RA3L', 'RA3S', 'RA3P', 'RA3Q', 'RA3C', 'RA3T', 'RA3O', 'RA3H', 'RA4M2', 'RA4M3',
             'RA4M4', 'RA4M5'),
    'thumb': ('RA4M1_PHANT', 'RA4M1', 'RA5P1', 'RA6D1'),
    'index': ('RA5P2', 'RA6M2', 'RA7D2', ),
    'middle': ('RA5P3', 'RA6M3', 'RA7D3'),
    'ring': ('RA5P4', 'RA6M4', 'RA7D4'),
    'pinky': ('RA5P5', 'RA6M5', 'RA7D5')
}
SEGMENT_MARKER_GROUPS = {
    'chest': ('M_SternumTop', 'M_SternumBot', 'M_RScapulaAnt', 'M_RScapulaPost'),
    'humerus': ('M_RA1H_DT', 'M_RA1H_ME', 'M_RA1H_LE', 'M_RA1H_IE'),
    'forearm': ('M_RA2U_OL', 'M_RA2U_D', 'M_RA2R_M', 'M_RA2U_SP', 'M_RA2R_SP'),
    'hand': ('M_RA3M', 'M_RA4M2', 'M_RA4M3', 'M_RA4M4', 'M_RA4M5'),
    'thumb': ('M_RA4M1_B', 'M_RA4M1_H', 'M_RA5P1_H', 'M_RA6D1_H', ),
    'index': ('M_RA5P2_H', 'M_RA6M2_H', 'M_RA7D2_H', ),
    'middle': ('M_RA5P3_H', 'M_RA6M3_H', 'M_RA7D3_H', ),
    'ring': ('M_RA5P4_H', 'M_RA6M4_H', 'M_RA7D4_H', ),
    'pinky': ('M_RA5P5_H', 'M_RA6M5_H', 'M_RA7D5_H', )
}


def fill_nans(v):
    '''input v must be a vector and have at least one non-nan value'''
    # pad beginning and end with non-nan values
    non_nan_idxs = np.where(np.logical_not(np.isnan(v)))[0]
    v = np.insert(v, 0, v[non_nan_idxs[0]])
    v = np.append(v, v[non_nan_idxs[-1] + 1])

    # same to found indices
    non_nan_idxs += 1
    non_nan_idxs = np.insert(non_nan_idxs, 0, 0)
    non_nan_idxs = np.append(non_nan_idxs, len(non_nan_idxs))

    # create another array with indices of previous and next non-nan values
    # prev
    prev_nonnan_idxs = []
    lni = None
    for i, va in enumerate(v):
        if not np.isnan(va):
            lni = i
        prev_nonnan_idxs.append(lni)
    # next
    next_nonnan_idxs = []
    nni = None
    for i, va in enumerate(reversed(v)):
        if not np.isnan(va):
            nni = len(v) - i - 1
        next_nonnan_idxs.append(nni)
    next_nonnan_idxs.reverse()

    # linear interpolation for nan values
    for iv, va in enumerate(v):
        if np.isnan(va):
            v[iv] = (v[prev_nonnan_idxs[iv]] + (v[next_nonnan_idxs[iv]] - v[prev_nonnan_idxs[iv]]) *
                     (iv - prev_nonnan_idxs[iv]) / (next_nonnan_idxs[iv] - prev_nonnan_idxs[iv]))
    v = v[1:-1]
    return v


def prepare_scaling_files(trial, session, mstruct, period):
    # load trc
    bodyparts, _, times, points, rate, units = io_tools.import_trc(
        trial.markers_3D_filename_trc)
    times = np.array(times)
    points = np.array(points)

    # find ps markers, doesn't matter if left or right in this case
    marker_name_dict = io_tools.dic_from_csv(
        os.path.join(os.path.split(mstruct['opensim_model'])[0], 'marker_meta.csv'),
        'sDlcMarker', 'sOpenSimMarker')
    ps_osim_markers = []
    for poms in mstruct['ps_markers'].values():
        for pom in poms:
            ps_osim_markers.append(marker_name_dict[pom])
    non_ps_mask = np.zeros((len(bodyparts), ), dtype=bool)
    for ibp, bp in enumerate(bodyparts):
        if bp not in ps_osim_markers:
            non_ps_mask[ibp] = True

    # make a stability plot for the markers:
    #   existence of markers
    nbp_total = np.sum(non_ps_mask)
    nbp_visible = nbp_total - np.sum(np.isnan(points[:, non_ps_mask, 0]), axis=1)

    #   their 3D velocity
    bp_velocities = np.diff(points[:, non_ps_mask, :], axis=0)
    bp_velocities = np.linalg.norm(bp_velocities, axis=2) * rate
    bp_velocities = np.insert(bp_velocities, 0, bp_velocities[0, :], axis=0)

    # combined
    bp_weights = copy.deepcopy(bp_velocities)
    bp_weights /= np.nanquantile(bp_weights, 0.95, axis=0)  # normalize to 95% quantile
    bp_weights[np.logical_or(np.isnan(bp_weights), bp_weights > 1)] = 1

    # averaged
    bp_score = np.mean(bp_weights, axis=1)
    filter_window = 0.1  # s
    out_halfwindow = 0.15
    bp_score_smoothed = scipy.signal.medfilt(
        bp_score, kernel_size=int(np.ceil(rate * filter_window)))
    best_score = np.min(bp_score_smoothed)
    best_score_idx = np.where(bp_score_smoothed == best_score)[0][0]
    best_score_time = times[best_score_idx]
    bs_period = [best_score_time - out_halfwindow, best_score_time + out_halfwindow]
    if period is None or len(period) == 0:
        period = bs_period
        rs('Using best estimated period: [{}, {}]'.format(period[0], period[1]))
    else:
        rs('Using provided period: [{}, {}]'.format(period[0], period[1]))

    # display the metrics
    xn_subplots = 1
    yn_subplots = 4

    plt.figure(figsize=(16, 9))

    ax = plt.subplot(yn_subplots, xn_subplots, 1)
    ax.plot(times, nbp_visible, 'k')
    ax.set_ylim(ymin=0, ymax=nbp_total+1)
    ax.set_ylabel('# visible markers')
    tools.actual_vline(ax, period[0], color=micolors['red'][600])
    tools.actual_vline(ax, period[1], color=micolors['red'][600])

    ax2 = plt.subplot(yn_subplots, xn_subplots, 2, sharex=ax)
    ax2.plot(times, bp_velocities)
    ax2.set_ylim(ymin=0)
    ax2.set_ylabel('Speed, m/s')
    tools.actual_vline(ax2, period[0], color=micolors['red'][600])
    tools.actual_vline(ax2, period[1], color=micolors['red'][600])

    ax3 = plt.subplot(yn_subplots, xn_subplots, 3, sharex=ax)
    ax3.plot(times, bp_weights)
    ax3.set_ylim([0, 1.01])
    ax3.set_ylabel('Individual marker\nweights, nu')
    tools.actual_vline(ax3, period[0], color=micolors['red'][600])
    tools.actual_vline(ax3, period[1], color=micolors['red'][600])

    ax4 = plt.subplot(yn_subplots, xn_subplots, 4, sharex=ax)
    ax4.plot(times, bp_score, 'k')
    ax4.plot(times, bp_score_smoothed, 'k--')
    ax4.set_ylim([0, 1.01])
    ax4.set_ylabel('Marker instability\nscore, nu')
    tools.actual_vline(ax4, period[0], color=micolors['red'][600])
    tools.actual_vline(ax4, period[1], color=micolors['red'][600])
    tools.actual_vline(ax4, bs_period[0], color=micolors['purple'][600], linestyle='--')
    tools.actual_vline(ax4, bs_period[1], color=micolors['purple'][600], linestyle='--')

    ax.set_xlim([times[0], times[-1]])
    ax.set_xlabel('Time, s')

    # before taking a subset, median filter the data
    for ibp, _ in enumerate(bodyparts):
        for ixyz in range(3):
            points[:, ibp, ixyz] = scipy.signal.medfilt(
                points[:, ibp, ixyz], kernel_size=int(np.ceil(rate * filter_window)))

    # take a subset of frames based on `times'
    start_frame = tools.find_first(times >= period[0])
    end_frame = tools.find_last(times <= period[1])
    rs('Found start frame {} and end frame {}.'.format(start_frame, end_frame))
    points = points[start_frame:end_frame+1]
    # decided not to, since having a ps marker is a nice reference location
    # for pom in ps_osim_markers:
    #     ibp = bodyparts.index(pom)
    #     rs('Removing pressure sensor marker {}.'.format(pom))
    #     del bodyparts[ibp]
    #     points[:, ibp, :] = []

    # remove markers that are completely absent
    for ibp in reversed(range(len(bodyparts))):
        if np.all(np.isnan(points[:, ibp, 0])):
            rs('Removing absent marker {}.'.format(bodyparts[ibp]))
            del bodyparts[ibp]
            points = np.delete(points, ibp, 1)

    # replace the undetected with approximated positions
    for ibp, _ in enumerate(bodyparts):
        if np.any(np.isnan(points[:, ibp, 0])):
            for ixyz in range(3):
                points[:, ibp, ixyz] = fill_nans(points[:, ibp, ixyz])

    # report on how many non-PS markers left
    rs('Total {} non-pressure sensors left:'.format(
        sum([0 if bp in ps_osim_markers else 1 for bp in bodyparts]),
        ', '.join(bp for bp in bodyparts if bp not in ps_osim_markers)))
    for smg, markers in SEGMENT_MARKER_GROUPS.items():
        rs('\t{}: {}'.format(
            smg, ', '.join(m for m in markers if m in bodyparts)))
    # save trc
    io_tools.export_trc(
        trial.scaling_markers_3D_filename_trc, bodyparts, points.tolist(), rate, units=units)

    # create IK file
    marker_weights = {bp: 1 for bp in bodyparts}
    time_range = [0, 1. / rate * len(points)]
    # first time it will have to be based on unscaled model, later on the scaled one
    ik_xml_str = inverse_kinematics.IK_XML_STR.format(
        model_file='Unassigned')  # will run on the open model
    inverse_kinematics.make_ik_file(
        trial.scaling_ik_filename, ik_xml_str, marker_weights,
        trial.scaling_markers_3D_filename_trc,
        trial.scaling_kinematic_filename, time_range)

    # create SC file
    tool_name = session + '_' + os.path.basename(trial.scaling_sc_filename)[:-4] + '_scaling_tool'
    inverse_kinematics.make_sc_file(
        trial.scaling_sc_filename, tool_name, SEGMENT_BODY_GROUPS,
        trial.scaling_markers_3D_filename_trc, time_range)


def transfer_position_to_model(trial, mstruct, mdof):
    # load joint angle trace
    dof_names, _, dofs = io_tools.import_mot(trial.scaling_kinematic_filename)

    # find median posture
    median_positions = {}
    for dof_name, dof in zip(dof_names, dofs):
        median_positions[dof_name] = np.median(dof)
        if mdof[dof_name]['rot']:
            median_positions[dof_name] *= np.pi / 180

    # export (overwrite) to the opensim model
    inverse_kinematics.set_opensim_model_default_position(
        mstruct['opensim_model'], mstruct['opensim_model'], median_positions)

    rs('Transferred median position from {} into model {}.'.format(
        trial.scaling_kinematic_filename, mstruct['opensim_model']))


def create_scaling_files(preset, session, trial_number, temp, overwrite, period, transfer_position):
    """Create IK and SC files for scaling an OpenSim model.

    Arguments:
        preset {dict} --- Preset holding the raw ('default_server') and processed
            ('processed_server') server locations.
        session {str} --- Session directory to use.
        trial_number {int} --- Trial to do adjustment on.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        period {float} --- Time period in seconds to use for scaling. If empty, use best estimation.
        transfer_position {bool} --- Transfer the joint angles that have resulted from IK into the model
            that is being scaled. Different mode of operation, does not generate scaling files.
    """
    rserv = preset['default_server']
    pserv = preset['processed_server']

    logs.setup_logging(temp, sessions_dir=pserv)

    if not os.path.exists(rserv):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            rserv))

    if len(session) == 0:
        session = meta_session.find_session_dirs(rserv)[0]

    rs('Processing session {}.'.format(session))
    raw_ss = os.path.join(rserv, session)
    proc_ss = os.path.join(pserv, session)

    if not os.path.exists(raw_ss):
        ValueError('Session {} does not exist on the server.'.format(session))

    # load session meta
    mstruct, mdof, _, msession = meta_session.load_meta_information(raw_ss, proc_ss)

    # find trial
    trial = meta_session.find_trial(msession, trial_number)
    if trial is None:
        ValueError('Could not find trial #{}.'.format(trial_number))

    # check existence of necessary files
    if not trial.do_3d_files_exist():
        ValueError('Trial {} does not have 3D files to use for IK and scaling.'.format(
            trial_number))

    if transfer_position:
        # transferring average position from the IK file to a model
        if not os.path.exists(trial.scaling_kinematic_filename):
            ValueError('Scaling joint angle file for trial {} does not exists.'.format(
                trial_number))
        transfer_position_to_model(trial, mstruct, mdof)
    else:
        if trial.do_scaling_files_exist() and not overwrite:
            ws('Scaling files for trial {} already exist, aborting.'.format(trial_number))
            return

        # create out dir
        os.makedirs(mstruct['scaling_dir'], exist_ok=True)

        prepare_scaling_files(trial, session, mstruct, period)
