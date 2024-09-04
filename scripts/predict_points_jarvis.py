# ================================================ #
# This file should provide functions to perform    #
# Triangulation and marker prediction              #
# Input: camera_videos                             #
# Output: markers_3D                               #
# Note: for testing use DRH
# ================================================ #

import os
import sys
import tqdm
import argparse
import datetime
import time
import csv
import glob
import copy

import torch
import cv2
import pandas as pd
import numpy as np
import joblib
from astropy.convolution import convolve, Gaussian1DKernel

import ncams
from prehension import preset
from prehension import tools
from prehension.tools import rs, ws
from prehension import meta_session

from jarvis.utils.paramClasses import Predict3DParams
# from jarvis.prediction.predict3D import predict3D
from jarvis.config.project_manager import ProjectManager
from jarvis.prediction.jarvis3D import JarvisPredictor3D
from jarvis.utils.reprojection import ReprojectionTool
import jarvis.prediction.predict3D as jp_predict3D
from jarvis.utils.skeleton import get_skeleton
import jarvis.visualization.visualization_utils as utils


def validate_pth_file(fp):
    assert os.path.isfile(fp)
    assert os.path.splitext(fp)[1] == '.pth'


def get_calibrations(mstruct):
    calib_paths = {}
    for camid in mstruct['cameras']:
        calib_path = os.path.join(mstruct['calibration'], f'calib_*_{camid}.yaml')
        calib_path = glob.glob(calib_path)
        if len(calib_path) < 1:
            raise ValueError(f'Calibration for camera {camid} in {mstruct["calibration"]} not'
                             ' found.')
        if len(calib_path) > 1:
            ws(f'Found more than one calibration file for camera {camid} in'
               f' {mstruct["calibration"]}, using first.')

        calib_paths[camid] = calib_path[0]

    # print('DEBUG found calibrations:')
    # for k, v in calib_paths.items():
    #     print(f'{k}: {v}')

    return ReprojectionTool(calib_paths=calib_paths)


def my_predict3D(params, proj_cfg, trial, reproTool, processes):
    '''Remake of jarvis.prediction.predict3D.predict3D for our needs'''
    jarvisPredictor = JarvisPredictor3D(
        proj_cfg, params.weights_center_detect, params.weights_hybridnet,
        params.trt_mode)

    # aligned with calibration order of cameras
    video_paths = [trial.videos[k] for k in reproTool.cameras]
    # create openCV video read streams
    caps, img_size = jp_predict3D.create_video_reader(params, video_paths)

    # Make the number of frames go only until the end of the SHORTEST CAP so we
    # don't have to truncate the vids manually
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    min_frame_count = min(frame_counts)
    if frame_counts.count(frame_counts[0]) != len(frame_counts):
        # Then there are varying frame lens
        ws(f'Varying frame lengths found only analyzing up to the shortest'
           f' vid (n frames = {min_frame_count}).')

    # always full video
    params.number_frames = min_frame_count

    # print('DEBUG: videos:')
    # for v in video_paths:
    #     print(f'\t{v}')
    # print(f'DEBUG output: {trial.markers_3D_filename_jarvis_csv}')

    with open(trial.markers_3D_filename_jarvis_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # if keypoint names are defined, add header to csvs
        if (len(proj_cfg.KEYPOINT_NAMES) == proj_cfg.KEYPOINTDETECT.NUM_JOINTS):
            jp_predict3D.create_header(writer, proj_cfg)

        imgs_orig = np.zeros((len(caps), img_size[1], img_size[0], 3)).astype(np.uint8)

        for frame_num in tqdm.tqdm(range(params.number_frames)):
            # load a batch of images from all cameras in parallel using joblib
            joblib.Parallel(n_jobs=processes, require='sharedmem')(
                joblib.delayed(jp_predict3D.read_images)
                (cap, slc, imgs_orig) for slc, cap in enumerate(caps))
            imgs = torch.from_numpy(imgs_orig).cuda().float().permute(0, 3, 1, 2)[:, [2, 1, 0]]/255.

            points3D_net, confidences = jarvisPredictor(
                imgs,
                reproTool.cameraMatrices.cuda(),
                reproTool.intrinsicMatrices.cuda(),
                reproTool.distortionCoefficients.cuda())

            if points3D_net is not None:
                row = []
                for point, conf in zip(points3D_net.squeeze(), confidences.squeeze().cpu().numpy()):
                    row = row + point.tolist() + [conf]
                writer.writerow(row)
            else:
                row = []
                for i in range(proj_cfg.KEYPOINTDETECT.NUM_JOINTS*4):
                    row = row + ['NaN']
                writer.writerow(row)

            if params.progress_bar is not None:
                params.progress_bar.progress((frame_num+1) / params.number_frames)

    # close the videos
    for cap in caps:
        cap.release()


def ncams_filter_3D(x, filt_width=5):
    gauss_filt = Gaussian1DKernel(stddev=filt_width/10)
    x = ncams.reconstruction._nanmedianfilt(x, filt_width)
    x = convolve(x, gauss_filt, boundary='extend', nan_treatment='interpolate')
    return x


def convert_jarvis(trial, threshold):
    df = pd.read_csv(trial.markers_3D_filename_jarvis_csv, header=[0, 1])

    # remove points with low confidence
    conf_columns = []
    for bp, coord in df.columns:
        if coord == 'confidence':
            for coord2 in ('x', 'y', 'z'):
                def clearconf(row):
                    return row[(bp, coord2)] if row[(bp, coord)] > threshold else np.nan
                df[(bp, coord2)] = df.apply(clearconf, axis=1)
            conf_columns.append((bp, coord))

    # remove confidence
    df.drop(conf_columns, axis=1, inplace=True)

    # apply ncams filtering
    for bp, coord in df.columns:
        df[(bp, coord2)] = ncams_filter_3D(df[(bp, coord2)])

    # insert first column with IDs
    df.insert(0, ('bp', 'coords'), range(len(df)))

    # save
    df.to_csv(trial.markers_3D_filename_csv, index=False)


def prepare_ik(trial, mstruct, reflect, marker_name_dict):
    # export for OpenSim
    marker_weights, time_range = ncams.inverse_kinematics.triangulated_to_trc(
        trial.markers_3D_filename_csv, trial.markers_3D_filename_trc, marker_name_dict,
        rate=mstruct['fps'], reflect=reflect)

    # make all IK weights the same, otherwise the proximal markers overpower
    marker_weights = {k: 1 for k, v in marker_weights.items() if v > 0}

    # remove thorax-bound markers - sternum
    # TODO not used yet
    marker_weights_general = copy.deepcopy(marker_weights)

    # make general IK file
    ik_xml_str = ncams.inverse_kinematics.IK_XML_STR.format(
        model_file=mstruct['opensim_model_locked_base'])
    ncams.inverse_kinematics.make_ik_file(
        trial.ik_filename, ik_xml_str, marker_weights_general, trial.markers_3D_filename_trc,
        trial.pre_kinematic_filename, time_range)


def create_video_writer(in_caps, ou_video_paths):
    outs = []
    for cap, ovp in zip(in_caps, ou_video_paths):
        frameRate = cap.get(cv2.CAP_PROP_FPS)
        img_size = [
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        outs.append(cv2.VideoWriter(
            ovp,
            cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
            frameRate,
            (img_size[0], img_size[1])))
    return outs


def my_create_videos3D(params, proj_cfg, trial, reproTool, processes):
    '''Remake of jarvis.visualization.create_videos3D.create_videos3D for our needs'''
    # aligned with calibration order of cameras
    video_paths = [trial.videos[k] for k in reproTool.cameras]
    os.makedirs(trial.jarvis_video_dir, exist_ok=True)
    ou_video_paths = [trial.jarvis_videos[k] for k in reproTool.cameras]

    # no reason to spawn more than videos
    processes = min(processes, len(video_paths))

    # create openCV video read and write streams
    caps, img_size = jp_predict3D.create_video_reader(params, video_paths)
    outs = create_video_writer(caps, ou_video_paths)

    # Make the number of frames go only until the end of the SHORTEST CAP so we
    # don't have to truncate the vids manually
    frame_counts = [int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps]
    min_frame_count = min(frame_counts)
    if frame_counts.count(frame_counts[0]) != len(frame_counts):
        # Then there are varying frame lens
        ws(f'Varying frame lengths found only analyzing up to the shortest'
           f' vid (n frames = {min_frame_count}).')

    # always full video
    params.number_frames = min_frame_count

    # load data
    colors, line_idxs = get_skeleton(proj_cfg)
    data = np.genfromtxt(trial.markers_3D_filename_jarvis_csv, delimiter=',')
    if np.isnan(data[0, 0]):
        data = data[2:]
    points3D = np.delete(data, list(range(3, data.shape[1], 4)), axis=1)
    # confidences = data[:, 3::4]

    # buffer per timepoint
    imgs_orig = np.zeros((len(caps), img_size[1], img_size[0], 3)).astype(np.uint8)

    for frame_num in tqdm.tqdm(range(params.number_frames)):
        joblib.Parallel(n_jobs=processes, require='sharedmem')(
            joblib.delayed(jp_predict3D.read_images)
            (cap, slc, imgs_orig) for slc, cap in enumerate(caps))
        points3D_net = torch.from_numpy(points3D[frame_num].reshape(-1, 3)).float()
        points3D_net = points3D_net.to('cuda:0')
        # confidence = confidences[frame_num]

        if points3D_net is not None:
            points2D = reproTool.reprojectPoint(points3D_net).cpu().numpy()

            points2D = np.array(points2D)
            for i in range(len(outs)):
                for line in line_idxs:
                    utils.draw_line(imgs_orig[i], line, points2D[:, i], img_size, colors[line[1]])
                for j, points in enumerate(points2D):
                    utils.draw_point(imgs_orig[i], points[i], img_size, colors[j])
        for i, out in enumerate(outs):
            out.write(imgs_orig[i])

    # close videos
    for out in outs:
        out.release()
    for cap in caps:
        cap.release()


def main(server, processed_server, sessions, temp, trials_sel, jarvis_proj, threshold, overwrite,
         processes, predict, transform, make_videos):
    # Jarvis saves own log
    # tools.setup_logging(temp, sessions_dir=processed_server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)
        processed_session = os.path.join(processed_server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(
                server_session, processed_session)
        except Exception as e:
            ws('Could not load meta data from session {} ({}), skipping.'.format(session, repr(e)))
            continue

        # accumulate data
        trials = []
        for trial in tqdm.tqdm(msession, ncols=100, desc='Finding trials'):
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue

            # Skip condition 1
            if (os.path.isfile(trial.markers_3D_filename_jarvis_csv) and
                    os.path.isfile(trial.markers_3D_filename_csv) and not overwrite):
                continue

            trials.append(trial)

        print()
        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(trial.trial_number) for trial in trials])))

        if len(trials) == 0:
            continue

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
        os.makedirs(mstruct['markers_3D_jarvis_dir'], exist_ok=True)
        os.makedirs(mstruct['pre_ja_dir'], exist_ok=True)

        if predict or make_videos:
            # Load project
            pm = ProjectManager()
            pm.load(jarvis_proj)
            proj_cfg = pm.get_cfg()

            # Get params
            params = Predict3DParams(jarvis_proj, '')

            # set calibration
            reproTool = get_calibrations(mstruct)

        if predict:
            for trial in tqdm.tqdm(trials, ncols=100, desc='Running prediction'):
                # Run prediction
                my_predict3D(params, proj_cfg, trial, reproTool, processes)

        if transform:
            for trial in tqdm.tqdm(trials, ncols=100, desc='Converting files'):
                # convert the file
                convert_jarvis(trial, threshold)
                prepare_ik(trial, mstruct, reflect, marker_name_dict)

        if make_videos:
            for trial in tqdm.tqdm(trials, ncols=100, desc='Running prediction'):
                # Run prediction
                my_create_videos3D(params, proj_cfg, trial, reproTool, processes)


if __name__ == "__main__":
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Runs a trained Jarvis model on videos generating 3D points and IK files.'))
    tools.add_default_kwarguments(
        parser, {
            'server': current_preset['default_server'],
            'processed_server': current_preset['processed_server']})
    tools.add_default_arguments(parser, ('sessions', 'temp', 'overwrite', 'trials', 'processes'))

    # # custom
    parser.add_argument(
        '--make_videos',
        action='store_true',
        help='Renders videos with prediction.')
    parser.add_argument(
        '--jarvis_proj',
        type=str, default=current_preset['jarvis_config_path'],
        help='Jarvis project to use.')
    parser.add_argument(
        '--threshold',
        type=float, default=0.4,
        help='Threshold for likelihood of 3D points to be used.')
    parser.add_argument(
        '--dont_predict',
        action='store_true',
        help='Do not run JARVIS on the videos.')
    parser.add_argument(
        '--dont_transform',
        action='store_true',
        help='Do not transform JARVIS files into our format and create IK files.')

    args = parser.parse_args(args=argv)
    start_time = time.time()

    main(args.server, args.processed_server, args.sessions, args.temp, args.trials,
         args.jarvis_proj, args.threshold, args.overwrite, args.processes,
         not args.dont_predict, not args.dont_transform, args.make_videos)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
