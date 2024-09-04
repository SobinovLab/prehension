#!python3.7
import copy
import os

import numpy as np
import tqdm
from reporting_pool import ReportingPool
from scipy.spatial.transform import Rotation as R

import ncams
from . import meta_session
from . import tools
from .tools import rs, ws

THORAX_BOUND_MARKERS = ('M_SternumTop', 'M_SternumBot')
PROXIMAL_MARKERS = ('M_SternumTop', 'M_SternumBot', 'M_RScapulaAnt', 'M_RScapulaPost')
# all ps points should be within this radius of the centroid
# calculated from ps side = 9 cm, max width = 5 cm, rounded up
PS_CENTROID_RADIUS = 80


def rotation_vector(v):
    r = R.from_euler('zyx', [0, 90, 180], degrees=True)
    return r.apply(v)


def c3f_remove_far_ps(bodyparts, triangulated_points, pressure_sensor_markers):
    psm_idxs = [bodyparts.index(bp) for bp in pressure_sensor_markers if bp in bodyparts]
    # print('found {} bps: {}'.format(
    #     len(psm_idxs), ', '.join([str(psm_idx) for psm_idx in psm_idxs])))
    psm_centroid_xs = np.median(triangulated_points[:, 0, psm_idxs], axis=1)
    psm_centroid_ys = np.median(triangulated_points[:, 1, psm_idxs], axis=1)
    psm_centroid_zs = np.median(triangulated_points[:, 2, psm_idxs], axis=1)

    for ipsm, psm_idx in enumerate(psm_idxs):
        psm_centroid_dists = np.sqrt(
            (psm_centroid_xs - triangulated_points[:, 0, psm_idx]) ** 2 +
            (psm_centroid_ys - triangulated_points[:, 1, psm_idx]) ** 2 +
            (psm_centroid_zs - triangulated_points[:, 2, psm_idx]) ** 2)
        psm_centroid_flag = psm_centroid_dists > PS_CENTROID_RADIUS
        triangulated_points[psm_centroid_flag, 0, psm_idx] = np.nan
        triangulated_points[psm_centroid_flag, 1, psm_idx] = np.nan
        triangulated_points[psm_centroid_flag, 2, psm_idx] = np.nan
    return triangulated_points


def triangulate(trial, ncams_config, intrinsics_config, extrinsics_config, threshold,
                marker_name_dict, reflect, mstruct, do_triangulate):
    '''Triangulate and export of OpenSim'''
    pressure_sensor_markers = []
    for ps_markers in mstruct['ps_markers'].values():
        pressure_sensor_markers += ps_markers

    def c3f_remove_far_ps_local(bodyparts, triangulated_points):
        return c3f_remove_far_ps(bodyparts, triangulated_points, pressure_sensor_markers)

    # Triangulate
    if do_triangulate:
        ncams.reconstruction.triangulate_csv(
            ncams_config, trial.markers_2D_dirname, intrinsics_config, extrinsics_config,
            output_csv_fname=trial.markers_3D_filename_csv, filter_2D=True, filter_3D=True,
            threshold=threshold, method='centroid', custom_3D_filter=c3f_remove_far_ps_local)

    # export for OpenSim
    marker_weights, time_range = ncams.inverse_kinematics.triangulated_to_trc(
        trial.markers_3D_filename_csv, trial.markers_3D_filename_trc, marker_name_dict,
        rotation=rotation_vector, rate=mstruct['fps'], reflect=reflect)

    # make all IK weights the same, otherwise the proximal markers overpower
    marker_weights = {k: 1 for k, v in marker_weights.items() if v > 0}

    # remove thorax-bound markers - sternum
    marker_weights_general = copy.deepcopy(marker_weights)
    for tbm in THORAX_BOUND_MARKERS:
        if tbm in marker_weights_general.keys():
            del marker_weights_general[tbm]

    # make general IK file
    ik_xml_str = ncams.inverse_kinematics.IK_XML_STR.format(
        model_file=mstruct['opensim_model_locked_base'])
    ncams.inverse_kinematics.make_ik_file(
        trial.ik_filename, ik_xml_str, marker_weights_general, trial.markers_3D_filename_trc,
        trial.pre_kinematic_filename, time_range)

    # take a subset of markers
    marker_weights = {k: v for k, v in marker_weights.items() if k in PROXIMAL_MARKERS}

    # make IK file for thorax position
    ik_xml_str = ncams.inverse_kinematics.IK_XML_STR.format(
        model_file=mstruct['opensim_model'])
    ncams.inverse_kinematics.make_ik_file(
        trial.base_ik_filename, ik_xml_str, marker_weights, trial.markers_3D_filename_trc,
        trial.base_kinematic_filename, time_range)


def run_triangulate(server, sessions, trials_sel, temp, processes, overwrite, threshold, do_triangulate):
    """Triangulates marker positions from 2D to 3D and creates inverse kinematics files.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        threshold {float} --- Threshold for likelihood of 3D points to be used for triangulation.
        do_triangulate {bool} --- If specified, triangulation itself will be skipped,
            but the supporting files will be generated.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    failed_trial_reports = []
    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {} ({}), skipping.'.format(session, repr(e)))
            continue

        # accumulate data
        trials = []
        for trial in tqdm.tqdm(msession, ncols=100, desc='Finding trials'):
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if not trial.do_2d_files_exist():
                continue
            if not overwrite and (trial.do_pre_ik_files_exist() and
                                  trial.do_pre_base_ik_files_exist()):
                continue
            trials.append(trial)

        print()
        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(trial.trial_number) for trial in trials])))

        if len(trials) == 0:
            continue

        # preload camera configs
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

        # right or left handed
        reflect = mstruct['hand'] == 'left'
        if reflect:
            marker_name_dict = ncams.utils.dic_from_csv(
                os.path.join(os.path.split(mstruct['opensim_model'])[0],
                             'marker_meta_reflect.csv'),
                'sDlcMarker', 'sOpenSimMarker')
        else:
            marker_name_dict = ncams.utils.dic_from_csv(
                os.path.join(os.path.split(mstruct['opensim_model'])[0],
                             'marker_meta.csv'),
                'sDlcMarker', 'sOpenSimMarker')

        # into these the result will go
        os.makedirs(mstruct['markers_3D_dir'], exist_ok=True)
        os.makedirs(mstruct['pre_ja_dir'], exist_ok=True)

        p_args = list(zip(*[
            trials,
            [copy.deepcopy(ncams_config) for _ in trials],
            [copy.deepcopy(intrinsics_config) for _ in trials],
            [copy.deepcopy(extrinsics_config) for _ in trials],
            [threshold for _ in trials],
            [copy.deepcopy(marker_name_dict) for _ in trials],
            [reflect for _ in trials],
            [copy.deepcopy(mstruct) for _ in trials],
            [do_triangulate for _ in trials]
        ]))

        # # test
        # triangulate(*(p_args[0]))
        # sys.exit()

        if len(p_args) > 0:
            pool = ReportingPool(triangulate, p_args, processes=processes,
                                 report_on_change=True, track_failures=True)
            pool.start()

            if len(pool.failed_i_jobs) > 0:
                print()
                ws('Failed to transform trials:')
                for v in pool.failed_i_jobs:
                    ws('\t{}: {}'.format(trials[v].trial_number, pool.error_reports[v]))
                    failed_trial_reports.append('session {} trial {} error: {}'.format(
                        session, trials[v].trial_number, pool.error_reports[v]))

    if len(failed_trial_reports) > 0:
        print()
        ws('Failed converting trials across sessions:')
        for failed_trial_report in failed_trial_reports:
            ws('\t{}'.format(failed_trial_report))
