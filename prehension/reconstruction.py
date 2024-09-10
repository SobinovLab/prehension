#!python3
import csv
import warnings
from itertools import combinations

import cv2
import numpy as np
from astropy.convolution import convolve, Gaussian1DKernel
from astropy.utils.exceptions import AstropyWarning
warnings.simplefilter('ignore', category=AstropyWarning)
from scipy.spatial.distance import euclidean


def triangulate(image_coordinates, projection_matrices):
    '''
    The base triangulation function for NCams. Takes image coordinates and projection matrices from
    2+ cameras and will produce a triangulated point with the desired approach.

    Arguments:
        image_coordinates {array or list of} -- the x,y coordinates of a given marker for multiple
            cameras. The points must be in the format (1,2) if in a list or (n,2) if an array.
        projection_matrices {list} -- the projection matrices for the cameras corresponding
        to each image points input.

    Keyword Arguments:
        mode {str} -- the triangulation method to use:
            full_rank - performs SVD to find the point with the least squares error between all
                projection lines. If a threshold is given along with confidence values then only
                points above the threshold will be used.
            best_n - uses the n number of cameras with the highest confidence values for the
                triangulation. If a threshold is given then only points above the threshold will
                be considered.
            cluster - [in development] performs all combinations of triangulations and checks for
                outlying points suggesting erroneous image coordinates from one or more cameras.
                After removing the camera(s) that produce out of cluser points it then performs the
                full_rank triangulation.
        confidence_values {list or array} -- the confidence values for the points given by the
            marking system (e.g. DeepLabCut)
        threshold {float} -- the minimum confidence value to accept for triangulation.

    Output:
        u_3d {(1,3) np.array} -- the triangulated point produced.

    '''
    u_3d = np.zeros((1,3))
    u_3d.fill(np.nan)

    # Check if image coordinates are formatted properly
    if isinstance(image_coordinates, list):
        if len(image_coordinates) > 1:
            image_coordinates = np.vstack(image_coordinates)
        else:
            return u_3d

    if not np.shape(image_coordinates)[1] == 2:
        raise ValueError('ncams.reconstruction.triangulate only accepts numpy.ndarrays or lists of' +
                         'in the format (camera, [x,y])')

    num_cameras = np.shape(image_coordinates)[0]
    if num_cameras < 2: # return NaNs if insufficient points to triangulate
        return u_3d

    if num_cameras != len(projection_matrices):
        raise ValueError('Different number of coordinate pairs and projection matrices given.')

    decomp_matrix = np.empty((num_cameras*2, 4))
    for decomp_idx in range(num_cameras):
        point_mat = image_coordinates[decomp_idx]
        projection_mat = projection_matrices[decomp_idx]

        temp_decomp = np.vstack([
            [point_mat[0] * projection_mat[2, :] - projection_mat[0, :]],
            [point_mat[1] * projection_mat[2, :] - projection_mat[1, :]]])

        decomp_matrix[decomp_idx*2:decomp_idx*2 + 2, :] = temp_decomp

    Q = decomp_matrix.T.dot(decomp_matrix)
    u, _, _ = np.linalg.svd(Q)
    u = u[:, -1, np.newaxis]
    u_3d = np.transpose((u/u[-1, :])[0:-1, :])

    return u_3d


def undistort_point(distorted_points, camera_matrix, distortion_coefficient):
    return np.squeeze(cv2.undistortPoints(
        distorted_points, camera_matrix, distortion_coefficient, None,
        P=camera_matrix))


def triangulate_point(method, num_cameras, uics, iccs, projection_matrices, best_n=2,
                      centroid_threshold=2.5):
    cam_image_points = np.empty((2, num_cameras))
    cam_image_points.fill(np.nan)

    if method == 'full_rank' or method == 'centroid':
        for icam in range(num_cameras):
            cam_image_points[:, icam] = uics[icam]
    elif method == 'best_n':
        # decorate-sort-undecorate sort to find the icams for the highest likelihood
        best_likelh = [b[0] for b in sorted(
            zip(range(num_cameras), [iccs[icam].astype(np.float64) for icam in range(num_cameras)]),
            key=lambda x: x[1], reverse=True)][:best_n]
        for icam in [icam for icam in range(num_cameras) if icam in best_likelh]:
            cam_image_points[:, icam] = uics[icam]

    # Check how many cameras detected the bodypart in that frame
    cams_detecting = ~np.isnan(cam_image_points[0, :])
    cam_idx = np.where(cams_detecting)[0]
    if np.sum(cams_detecting) < 2:
        return None

    if method == 'full_rank' or method == 'best_n':
        # Create the image point and projection matrices
        tri_projection_mats, tri_image_points = [], []
        for cam in cam_idx:
            tri_image_points.append(cam_image_points[:, cam])
            tri_projection_mats.append(projection_matrices[cam])

        return triangulate(tri_image_points, tri_projection_mats)

    elif method == 'centroid':
        cam_comb_list = list(combinations(cam_idx, 2))
        num_combs = len(cam_comb_list)
        t_points = np.zeros((num_combs, 3))
        for c in range(num_combs):
            tri_projection_mats, tri_image_points = [], []
            for cam in cam_comb_list[c]:
                tri_image_points.append(cam_image_points[:, cam])
                tri_projection_mats.append(projection_matrices[cam])

            t_points[c, :] = triangulate(tri_image_points, tri_projection_mats)
        # Take the centroid of the points
        t_centroid = np.mean(t_points, axis=0)

        # Check for outliers if there are sufficient points to do so
        if num_combs > 3:
            t_cent_dist = []
            for c in range(num_combs):
                t_cent_dist.append(euclidean(t_centroid, t_points[c, :]))
            t_cent_dist = np.vstack(t_cent_dist)
            # Get z-scores for the distances from the centroid
            euclid_sd = np.std(t_cent_dist)
            euclid_threshold = euclid_sd * centroid_threshold
            dist_bool = t_cent_dist < euclid_threshold

            if np.sum(dist_bool) < num_combs:  # Recalculate the centroid
                cent_idx = np.where(dist_bool)[0]
                t_points_filt = t_points[cent_idx, :]
                t_centroid = np.mean(t_points_filt, axis=0)

        return t_centroid


def triangulate_points(
        ncams_config, intrinsics_config, extrinsics_config,
        bodyparts, num_frames, image_coordinates, ic_confidences,
        threshold=0.9, method='full_rank', best_n=2,
        centroid_threshold=2.5, undistorted_data=False,
        filter_3D=False, custom_3D_filter=None):
    # check if configs are not None
    if intrinsics_config is None:
        raise ValueError('No intrinsic configuration provided.')
    if extrinsics_config is None:
        raise ValueError('No extrinsics configuration provided.')

    camera_matrices = intrinsics_config['camera_matrices']
    if not undistorted_data:
        distortion_coefficients = intrinsics_config['distortion_coefficients']

    world_locations = extrinsics_config['world_locations']
    world_orientations = extrinsics_config['world_orientations']

    if method not in ('full_rank', 'best_n', 'centroid'):
        raise ValueError('"{}" is not an accepted method. '
                         'Please use "full_rank", "best_n", or "centroid".'.format(method))

    num_cameras = len(ncams_config['serials'])
    num_bodyparts = len(bodyparts)

    if not undistorted_data:  # Undistort points
        undistorted_image_coordinates = []
        # for each camera
        for cam_image_coordinates, camera_matrix, distortion_coefficient in zip(
                image_coordinates, camera_matrices, distortion_coefficients):
            undistorted_csv_array = np.empty(cam_image_coordinates.shape)
            undistorted_csv_array.fill(np.nan)
            for bp in range(num_bodyparts):
                undistorted_csv_array[:, :, bp] = undistort_point(
                    cam_image_coordinates[:, :, bp],
                    camera_matrix, distortion_coefficient)

            undistorted_image_coordinates.append(undistorted_csv_array)

    else:
        undistorted_image_coordinates = image_coordinates

    # Triangulation
    # Make the projection matrices
    projection_matrices = []
    for icam in range(num_cameras):
        projection_matrices.append(make_projection_matrix(
            camera_matrices[icam], world_orientations[icam], world_locations[icam]))

    # Triangulate the points
    triangulated_points = np.empty((num_frames, 3, len(bodyparts)))
    triangulated_points.fill(np.nan)

    for iframe in range(num_frames):
        for bodypart in range(len(bodyparts)):
            triangulated_point = triangulate_point(
                method, num_cameras,
                [undistorted_image_coordinates[icam][iframe, :, bodypart]
                 for icam in range(num_cameras)],
                [ic_confidences[icam][iframe, bodypart] for icam in range(num_cameras)],
                projection_matrices,
                best_n=best_n, centroid_threshold=centroid_threshold)

            if triangulated_point is not None:
                triangulated_points[iframe, :, bodypart] = triangulated_point

    if filter_3D:
        triangulated_points = process_points(triangulated_points, '3D', threshold=threshold)

    if custom_3D_filter is not None:
        triangulated_points = custom_3D_filter(bodyparts, triangulated_points)

    return triangulated_points


def make_projection_matrix(camera_matrix, world_orientation, world_location):
    '''Makes a projection matrix from camera calibration and pose estimation info.

    Arguments:
        camera_matrix {np.array} -- camera calibration matrix for the camera.
        world_orientation {np.array} -- world orientation of the camera.
        world_location {np.array} -- world location of the camera.

    Output:
        projection_matrix {np.array} -- projection matrix of the camera
    '''
    # Make matrix if necessary
    if world_orientation.shape == (3, 1) or world_orientation.shape == (1, 3):
        world_orientation = cv2.Rodrigues(world_orientation)[0]  # Convert to matrix

    if world_location.shape == (1, 3):  # Format
        world_location = np.transpose(world_location)

    projection_matrix = np.matmul(camera_matrix, np.hstack((world_orientation, world_location)))

    return projection_matrix


def process_points(path_or_array, csv_type, filt_width=5, threshold=0.9, filtering=True):
    '''Formats and processes CSVs or numpy arrays as necessary for further usage.
       Uses median and gaussian filters to both smooth and interpolate points.
       Will only interpolate when fewer missing values are present than the gaussian width.

    Arguments:
        path_or_array {str} -- path of the triangulated csv or a numpy array (2 or 3D).
        csv_type {str} -- indicator of whether or not the array is '2D' or '3D'
    Keyword Arguments:
        filt_width {int} -- how wide the filters should be. (default: 5)
        threshold {float} -- confidence threshold to filter 2D DLC data by (default: 0.9).
        filtering {bool} -- whether or not to perform median and gaussian filters (default: True).
    Outputs if csv_type == '2D' is a tuple:
        processed_point_array {ndarray([num frame,num axes,num bodypart])}
        formatted_confidence_values {ndarray([num frame,num bodypart])}
    Output if csv_type == '3D':
        processed_point_array {ndarray([num frame,num axes,num bodypart])}
    '''
    # Check if the input is an array or a path
    if type(path_or_array) == str: # Assume it's DLC output or CSV output from triangulation
        # Load in the CSV
        with open(path_or_array, 'r') as f:
            csv_reader = csv.reader(f)
            # Check if the csv is a DLC output or a triangulation output
            csv_row = next(csv_reader)
            if csv_type == '2D':
                csv_row = next(csv_reader)

            # Get the names of the bodyparts for storage
            bodyparts = []
            for i, bp in enumerate(csv_row):
                if (i-1)%3 == 0:
                    bodyparts.append(bp)
            num_bodyparts = len(bodyparts)

            next(csv_reader) # Skip the 'xyz/xyc title row'
            point_array = []
            for row in csv_reader:
                point_array.append([[] for _ in range(3)])
                for ibp in range(num_bodyparts):
                    point_array[-1][0].append(float(row[1+ibp*3]))
                    point_array[-1][1].append(float(row[2+ibp*3]))
                    point_array[-1][2].append(float(row[3+ibp*3]))

        point_array = np.array(point_array)

    elif type(path_or_array) == np.ndarray: # Assume it's ncam working array
        if len(path_or_array.shape) == 2: # Flat CSV format
            num_bodyparts = int(path_or_array.shape[1]/3)
            n_frames = int(path_or_array.shape[0])
            point_array = []
            for f in range(n_frames):
                row = path_or_array[f,:]
                point_array.append([[] for _ in range(3)])
                for ibp in range(num_bodyparts):
                    point_array[-1][0].append(float(row[ibp*3]))
                    point_array[-1][1].append(float(row[1+ibp*3]))
                    point_array[-1][2].append(float(row[2+ibp*3]))

            point_array = np.array(point_array)

        elif len(path_or_array.shape) == 3: # Already formatted
            point_array = path_or_array
            num_bodyparts = int(path_or_array.shape[2])
            n_frames = int(path_or_array.shape[0])

    else:
        raise ValueError('Incompatible type given to "path_or_array". Must be "str" or "ndarray".')

    # Threshold filtering
    if csv_type == '2D':
        thresholded_point_array = np.empty((point_array.shape[0], 2, point_array.shape[2]))
        thresholded_point_array.fill(np.nan)
        formatted_confidence_values = np.empty((point_array.shape[0], point_array.shape[2]))
        formatted_confidence_values.fill(np.nan)
        for ibp in range(num_bodyparts):
            c_vals = np.squeeze(point_array[:,2,ibp])
            c_idx = np.where(c_vals > threshold)[0]
            ibp_vals = np.squeeze(point_array[:,:2,ibp])
            thresholded_point_array[c_idx,:,ibp] = ibp_vals[c_idx,:]
            formatted_confidence_values[c_idx,ibp] = c_vals[c_idx]

        point_array = thresholded_point_array

    # gaussian and median filtering
    if filtering:
        num_axes = point_array.shape[1]
        # Smooth each bodypart along each axis
        processed_point_array = np.empty(point_array.shape)
        processed_point_array.fill(np.nan)
        gauss_filt = Gaussian1DKernel(stddev=filt_width/10)
        for ibp in range(num_bodyparts):
            for a in range(num_axes):
                ibp_a = np.squeeze(point_array[:,a,ibp])
                # Apply median filter
                ibp_a = _nanmedianfilt(ibp_a, filt_width)
                # Apply gaussian filter
                ibp_a_gauss = convolve(ibp_a, gauss_filt, boundary='extend', nan_treatment='interpolate')
                processed_point_array[:,a,ibp] = ibp_a_gauss
    else:
        processed_point_array = point_array

    # return
    if csv_type == '2D':
        return processed_point_array, formatted_confidence_values
    else:
        return processed_point_array


def _nanmedianfilt(input_vector, kernel_width):
    '''Median filter that ignores nan values'''
    if kernel_width % 2 == 0:
        kernel_width = kernel_width + 1

    kernel_offset = int((kernel_width-1)/2)

    output_vector = np.empty(input_vector.shape)
    output_vector.fill(np.nan)

    init_idx = int(np.ceil(kernel_width/2))
    term_idx = int(len(input_vector) - np.ceil(kernel_width/2))

    output_vector[:init_idx] = input_vector[:init_idx]
    output_vector[term_idx:] = input_vector[term_idx:]
    for idx in np.arange(init_idx, term_idx):
        vals_to_filt = input_vector[idx-kernel_offset:idx+kernel_offset+1]
        num_nans = sum(np.isnan(vals_to_filt))
        if num_nans < kernel_offset:
            output_vector[idx] = np.nanmedian(vals_to_filt)

    return output_vector
