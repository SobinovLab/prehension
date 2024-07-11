#!python3
import os
import re
import sys
import warnings
import logging
import datetime
import time
import numpy as np
import scipy
import scipy.signal
import scipy.interpolate
import matplotlib.pyplot as plt
import uuid
import atexit
import shutil
from . import io_tools
import pdb

# matching geom or body
DIGITS = {
    'thumb': {'c': 'pink', 'exp': lambda v: re.search('[RL]A[0-9][MPD]1.*', v)},
    'index': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A[0-9][MPD]2.*', v)},
    'middle': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A[0-9][MPD]3.*', v)},
    'ring': {'c': 'lime', 'exp': lambda v: re.search('[RL]A[0-9][MPD]4.*', v)},
    'pinky': {'c': 'brown', 'exp': lambda v: re.search('[RL]A[0-9][MPD]5.*', v)},
    'None': {'c': 'blue grey', 'exp': lambda v: True}
}
UNCLAIMED_NAME = 'None'
UNCLAIMED_INDEX = list(DIGITS.keys()).index(UNCLAIMED_NAME)

SEGMENTS = {
    'thumb_mc': {'c': 'pink', 'exp': lambda v: re.search('[RL]A4M1.*', v)},
    'thumb_pp': {'c': 'pink', 'exp': lambda v: re.search('[RL]A5P1.*', v)},
    'thumb_dp': {'c': 'pink', 'exp': lambda v: re.search('[RL]A6D1.*', v)},

    'index_mc': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A4M2.*', v)},
    'index_pp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A5P2.*', v)},
    'index_mp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A6M2.*', v)},
    'index_dp': {'c': 'indigo', 'exp': lambda v: re.search('[RL]A7D2.*', v)},

    'middle_mc': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A4M3.*', v)},
    'middle_pp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A5P3.*', v)},
    'middle_mp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A6M3.*', v)},
    'middle_dp': {'c': 'cyan', 'exp': lambda v: re.search('[RL]A7D3.*', v)},

    'ring_mc': {'c': 'lime', 'exp': lambda v: re.search('[RL]A4M4.*', v)},
    'ring_pp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A5P4.*', v)},
    'ring_mp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A6M4.*', v)},
    'ring_dp': {'c': 'lime', 'exp': lambda v: re.search('[RL]A7D4.*', v)},

    'pinky_mc': {'c': 'brown', 'exp': lambda v: re.search('[RL]A4M5.*', v)},
    'pinky_pp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A5P5.*', v)},
    'pinky_mp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A6M5.*', v)},
    'pinky_dp': {'c': 'brown', 'exp': lambda v: re.search('[RL]A7D5.*', v)}
}

LPS_NAME = 'medial_sensor'
RPS_NAME = 'lateral_sensor'


# @https://stackoverflow.com/questions/12523586/python-format-size-application-converting-b-to-kb-mb-gb-tb
def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)
    else:
        return '{0:.2f} TB'.format(B / TB)


def print_copy(*args, **kwargs):
    """Simulates shutil.copy, but only prints out the targets."""
    print('{} -> {}'.format(args[0], args[1]))


class PrintCopyAccumulateSize():
    """Calculates the sum of sizes of files passed as  """

    def __init__(self, dry_run, verbose):
        self.size = 0
        self.dry_run = dry_run
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose > 0:
            print_copy(*args, **kwargs)
        self.size += os.path.getsize(args[0])
        if not self.dry_run:
            shutil.copy2(*args, **kwargs)

    def __str__(self):
        return humanbytes(self.size)


def copytree(src, dst, symlinks=False, ignore=None, copy_function=shutil.copy2,
             ignore_dangling_symlinks=False, make_dir=True, dir_exist_ok=False,
             overwrite_existing=None, verbose=False):
    """Recursively copy a directory tree.

    ADAPTED from shutil.py function. Changes:
        option to turn off make directory. Useful when only walking the tree with reports instead
            of copy_functions. If actually trying to copy the files, will most likely throw an
            error.
        option to make directory exist_ok true
        a callable option that asks what to do when the file already exists at the destination.
            Return True to overwrite, False to skip.
        verbose prints current source directory.

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied. If the file pointed by the symlink doesn't
    exist, an exception will be added in the list of errors raised in
    an Error exception at the end of the copy process.

    You can set the optional ignore_dangling_symlinks flag to true if you
    want to silence this exception. Notice that this has no effect on
    platforms that don't support os.symlink.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    The optional copy_function argument is a callable that will be used
    to copy each file. It will be called with the source path and the
    destination path as arguments. By default, copy2() is used, but any
    function that supports the same signature (like copy()) can be used.

    """
    if verbose:
        print(src)
    names = sorted(os.listdir(src))
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    if make_dir:
        os.makedirs(dst, exist_ok=dir_exist_ok)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if os.path.islink(srcname):
                linkto = os.readlink(srcname)
                if symlinks:
                    # We can't just leave it to `copy_function` because legacy
                    # code with a custom `copy_function` may rely on copytree
                    # doing the right thing.
                    os.symlink(linkto, dstname)
                    shutil.copystat(srcname, dstname, follow_symlinks=not symlinks)
                else:
                    # ignore dangling symlink if the flag is on
                    if not os.path.exists(linkto) and ignore_dangling_symlinks:
                        continue
                    # otherwise let the copy occurs. copy2 will raise an error
                    if os.path.isdir(srcname):
                        copytree(
                            srcname, dstname,
                            symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                            ignore_dangling_symlinks=ignore_dangling_symlinks, make_dir=make_dir,
                            dir_exist_ok=dir_exist_ok, overwrite_existing=overwrite_existing,
                            verbose=verbose)
                    else:
                        if (overwrite_existing is None or not os.path.exists(dstname) or
                                overwrite_existing(srcname, dstname)):
                            copy_function(srcname, dstname)
            elif os.path.isdir(srcname):
                copytree(
                    srcname, dstname,
                    symlinks=symlinks, ignore=ignore, copy_function=copy_function,
                    ignore_dangling_symlinks=ignore_dangling_symlinks, make_dir=make_dir,
                    dir_exist_ok=dir_exist_ok, overwrite_existing=overwrite_existing,
                    verbose=verbose)
            else:
                if (overwrite_existing is None or not os.path.exists(dstname) or
                        overwrite_existing(srcname, dstname)):
                    # Will raise a SpecialFileError for unsupported file types
                    copy_function(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except OSError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        # Copying file access times may fail on Windows
        if getattr(why, 'winerror', None) is None:
            errors.append((src, dst, str(why)))
    if errors:
        raise shutil.Error(errors)
    return dst


def overwrite_existing_always(srcname, dstname):
    return True


def overwrite_existing_never(srcname, dstname):
    return False


def overwrite_existing_new(srcname, dstname):
    return os.path.getmtime(srcname) > os.path.getmtime(dstname)


def overwrite_existing_new_box(srcname, dstname):
    # BOX does not save sub-second creation time
    return os.path.getmtime(srcname) - os.path.getmtime(dstname) >= 1


def copy_folder_contents(src_dir, target_dir, dir_names=[], file_names=[], suppress_warnings=False,
                         dry_run=False, copy_function=None, box=False, overwrite=False):
    start_time = time.time()
    if copy_function is None:
        local_copy_function = True
        copy_function = PrintCopyAccumulateSize(dry_run, int(dry_run))
    else:
        local_copy_function = False

    if overwrite:
        oex = overwrite_existing_always
    elif box:
        oex = overwrite_existing_new_box
    else:
        oex = overwrite_existing_new

    for dir_name in dir_names:
        src = os.path.join(src_dir, dir_name)
        dst = os.path.join(target_dir, dir_name)
        if os.path.isdir(src):
            copytree(src, dst, overwrite_existing=oex, dir_exist_ok=True,
                     copy_function=copy_function)
        else:
            if not suppress_warnings:
                ws(f'Warning: could not find expected directory ({src}) to upload')

    for fname in file_names:
        f_src = os.path.join(src_dir, fname)
        f_dst = os.path.join(target_dir, fname)

        if not os.path.isfile(f_src):
            if not suppress_warnings:
                ws(f'Warning: could not find expected file ({f_src}) to upload')
            continue

        # Check if the destination file exists
        if not os.path.isfile(f_dst) or oex(f_src, f_dst):
            # Destination file does not exist, copy the source file
            copy_function(f_src, f_dst)

    timedelta = str(datetime.timedelta(seconds=time.time() - start_time))
    # if shutil copy, the report won't be meaningful
    if local_copy_function:
        rs('Found {} files for copying during {}.'.format(
            copy_function, timedelta))


def savefig(dirname, filename):
    if dirname is None:
        return
    os.makedirs(dirname, exist_ok=True)

    plt.savefig(os.path.join(dirname, filename + '.png'))
    plt.savefig(os.path.join(dirname, filename + '.pdf'))


def actual_vline(ax, x, **kwargs):
    ymin, ymax = ax.get_ylim()
    ax.vlines(x, ymin, ymax, **kwargs)
    ax.set_ylim(ymin, ymax)


def downsample_at_timeseries(times, data, times_new):
    '''times_new has to have a much lower frequency. Neither have to be uniform.
    times_new cannot be wider than times'''
    times = np.array(times)
    data = np.array(data)
    times_new = np.array(times_new)

    data_new = []
    diff_times_new_hvd = np.diff(times_new) / 2
    t_froms = np.insert(times_new[1:] - diff_times_new_hvd, 0, times_new[0])
    t_tos = np.insert(times_new[:-1] + diff_times_new_hvd, len(diff_times_new_hvd), times_new[-1])
    for t_from, t_to in zip(t_froms, t_tos):
        slc = np.logical_and(times >= t_from, times < t_to)
        if sum(slc) == 0:
            warnings.warn('downsample_at_timeseries: No corresponding interval found.')
            data_new.append(np.zeros(np.shape(data)[1:]))
        else:
            data_new.append(np.median(data[slc], axis=0))

    # # testing
    # plt.figure()
    # plt.plot(times, reduce_force_matrices(data), 'k')
    # plt.plot(times_new, reduce_force_matrices(data_new), 'r')
    # plt.show()

    return np.array(data_new)


# TODO instead of decimate use time-based median filter bc sensor times are not consistent
def downsample(ps_times, data, ja_period):
    '''Downsamples pressure sensor data to joint angle frequency'''
    ps_period = np.median(np.diff(ps_times))

    # downsample
    q = int(round(ja_period / ps_period))
    data = scipy.signal.decimate(data, q, axis=0, ftype='fir')

    ps_times_new = np.arange(ps_times[0], ps_times[-1], ja_period)

    # HACK sometimes there is an off-by-one error for decimate requiring q to be an integer
    if len(ps_times_new) > len(data):
        data = np.append(data, [data[-1]], axis=0)

    return ps_times_new, data


def get_slice_to_time_base(tmin, n_times, times):
    # tmin must be within [times[0], times[1]] period
    # otherwise it will throw an exception
    start = next(x for x, val in enumerate(times) if val >= tmin)
    return slice(start, start + n_times)


def enforce_rom(dof, rng):
    dof[dof < rng[0]] = rng[0]
    dof[dof > rng[1]] = rng[1]
    return dof


def reduce_force_matrices(matrices, reduction=np.sum):
    '''Consider replacing with np.sum(matrices, axis=(1, 2)) on numpy versions >= 1.7.0'''
    red = []
    for matrix in matrices:
        red.append(reduction(matrix))
    return red


# @https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
def find_first(x):
    idx = x.view(bool).argmax() // x.itemsize
    return idx if x[idx] else -1


def find_last(x):
    ff = find_first(np.flip(x))
    if ff == -1:
        return -1
    return len(x) - ff - 1


def logging_atexit_function(logging_filename, destination_path, start_time):
    end_time = datetime.datetime.now()
    end_time_str = end_time.strftime('%Y.%m.%d-%H:%M:%S')
    time_passed_str = (end_time - start_time)
    logging.info(f'\n\tTimestamp: {end_time_str}\n'
                 f'\tTime passed: {time_passed_str}')
    shutil.copy(logging_filename, destination_path)


def setup_logging(temp, sessions_dir=None):
    '''
    Log filename, all arguments, and time
    '''
    os.makedirs(temp, exist_ok=True)
    # Move creation of random hash to this function
    random_hash = str(uuid.uuid4().hex)
    # Exectuting file name (for example: create_meta.py)
    arg_stump = ' '.join(sys.argv)
    exec_fname = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    # exec_fname = os.path.splitext(os.path.split(sys.argv[0])[1])[0]
    timestamp = datetime.date.today().strftime('%Y.%m.%d')
    start_time = datetime.datetime.now()
    timestamp_long = start_time.strftime('%Y.%m.%d-%H:%M:%S')
    logging_filename = os.path.join(temp, f'{timestamp}_{exec_fname}_{random_hash}.log')
    logging.basicConfig(filename=logging_filename, level=logging.INFO)
    logging.info('')

    # Define the cleanup action as upload to sessions_log dir
    if sessions_dir is not None:
        destination_path = os.path.join(sessions_dir, 'logs')
        os.makedirs(destination_path, exist_ok=True)
        assert os.path.exists(logging_filename)
        assert os.path.exists(destination_path)

        def laf():
            logging_atexit_function(logging_filename, destination_path, start_time)

        atexit.register(laf)

    logging.info(f'\n\tLog path: {logging_filename}\n'
                 f'\tScript call: {arg_stump}\n'
                 f'\tTimestamp: {timestamp_long}')
    print(f'{logging_filename} created.')


def rs(s):
    print(s)
    logging.info(s)


def ws(s):
    warnings.warn(s, stacklevel=2)
    logging.warning(s)


def add_default_arguments(parser, arguments):
    '''Possible arguments:
        sessions
        session -- singular
        trials
        trial -- singular required argument
        temp
        processes
        overwrite
        dry_run
    '''
    if isinstance(arguments, str):
        arguments = (arguments, )
    if 'sessions' in arguments:
        parser.add_argument(
            '--sessions',
            type=str, default=[], nargs='*', metavar='SESSION',
            help='List of directories for processing. If empty, find all unprocessed directories. '
            'Empty by default.')

    if 'trials' in arguments:
        parser.add_argument(
            '--trials',
            type=int, default=[], nargs='*', metavar='TRIAL_NUMBER',
            help='List of trials for processing. If empty, find all unprocessed trials. '
            'Empty by default.')

    temp_folder = os.path.join('C:\\', 'tmp')
    if 'temp' in arguments:
        parser.add_argument(
            '--temp',
            type=str, default=temp_folder,
            help='Folder for local temporary storage. Default: {}'.format(temp_folder))

    processes = int(round(os.cpu_count()*1.4))
    if 'processes' in arguments:
        parser.add_argument(
            '--processes',
            type=int, default=processes,
            help='Number of parallel processes in the pool. Defaults to {}.'.format(processes))

    # affecting data
    if 'overwrite' in arguments:
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrites the created files if they exist.')

    if 'dry_run' in arguments:
        parser.add_argument(
            '--dry_run', '--check',
            action='store_true',
            help='Check which trials have not been converted. Does not create data.')

    if 'session' in arguments:
        parser.add_argument(
            'session',
            type=str,
            help='Session directory to use. Required argument.')

    if 'trial' in arguments:
        parser.add_argument(
            'trial',
            type=int,
            help='Trial to do adjustment on. Required argument.')

    if 'make_plots' in arguments:
        parser.add_argument(
            '--make_plots',
            action='store_true',
            help='Makes some inspection figures. Run with --processes 1.')

    if 'debug' in arguments:
        parser.add_argument(
            '--debug',
            action='store_true',
            help='Run script in debug mode'
        )


def add_default_kwargument(parser, k, v):
    if k == 'server':
        parser.add_argument(
            '--server',
            type=str, default=v,
            help='Folder where the sessions are located. Default: {}'.format(v))
        return

    if k == 'processed_server':
        parser.add_argument(
            '--processed_server',
            type=str, default=v,
            help='Folder where the processed data from sessions are located. Default: {}'.format(v))
        return

    raise ValueError('Unknown keyword argument for parser: {}'.format(k))


def add_default_kwarguments(parser, kwarguments):
    '''Possible kwargs:
        server
    '''
    for k, v in kwarguments.items():
        add_default_kwargument(parser, k, v)


def get_matched_contact_frame_mask(exp, mcf, frame_size):
    frame_mask = np.zeros(frame_size, dtype=bool)
    for k, v in mcf.items():
        if exp(k):
            for ve in v:
                frame_mask[ve[0]][ve[1]] = True
    return frame_mask


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


def load_maps(trial):
    if os.path.exists(trial.lps_map_filename):
        lps_digit_mask = np.array(
            io_tools.import_one_csv_matrix(trial.lps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    if os.path.exists(trial.rps_map_filename):
        rps_digit_mask = np.array(
            io_tools.import_one_csv_matrix(trial.rps_map_filename, dtype=int))
    else:
        raise ValueError('Map does not exist.')
    return lps_digit_mask, rps_digit_mask


def load_forces(mstruct, trial):

    ps_matrices = {}
    matched_contacts = {}

    for ps_name in mstruct['ps_dic'].keys():
        ps_times, ps_matrices[ps_name] = io_tools.import_matrices(
            trial.get_post_ps_filenames()[ps_name])
        matched_contacts[ps_name] = io_tools.import_matched_contacts(
            trial.matched_contacts_filenames[ps_name])

    # some basic force parameters
    dts = np.diff(ps_times)
    trial.dt = np.median(dts)
    # for varied dts, for each time point:
    trial.dts = (np.concatenate(([0], dts)) + np.concatenate((dts, [0]))) / 2
    trial.total_force = (np.sum(ps_matrices['medial_sensor'], axis=(1, 2)) +
                         np.sum(ps_matrices['lateral_sensor'], axis=(1, 2)))
    trial.max_total_force = np.max(trial.total_force)
    trial.summed_force = np.sum(trial.total_force)
    trial.summed_impulse = np.sum(trial.total_force * trial.dts)
    trial.active_period_flag = trial.total_force >= (0.05 * trial.max_total_force)

    # export
    trial.ps_times = ps_times
    trial.ps_matrices = ps_matrices
    trial.matched_contacts = matched_contacts

    # pool all distances
    trial.pooled_distances = []
    for mc_fl, mc_fr in zip(matched_contacts['medial_sensor'], matched_contacts['lateral_sensor']):
        # print(sum([mc for mc in mc_fl.values()], []))
        # print(sum([mc for mc in mc_fr.values()], []))
        trial.pooled_distances.append([mc[2] for mc in sum([mc for mc in mc_fl.values()], [])] +
                                      [mc[2] for mc in sum([mc for mc in mc_fr.values()], [])])

    # load maps
    manual_digit_maps = {}
    # lps_digit_mask, rps_digit_mask  # rigidly set for ps_names
    (manual_digit_maps['medial_sensor'],
     manual_digit_maps['lateral_sensor']) = load_maps(trial)

    # find mask-based difference between manual and automatic labels
    mask_based_diff_per_sensor = {}
    unclaimed_force = {}
    for ps_name in mstruct['ps_dic'].keys():
        manual_digit_map = manual_digit_maps[ps_name]
        mask_based_diff_per_sensor[ps_name] = []
        unclaimed_force[ps_name] = []
        for i_frame in range(len(ps_times)):
            # build auto mask
            auto_mask = (len(DIGITS) - 1) * np.ones(np.shape(manual_digit_map))
            for i_digit, d in enumerate(DIGITS.values()):
                if i_digit == len(DIGITS) - 1:
                    break
                digit_auto_mask = get_matched_contact_frame_mask(
                    d['exp'], matched_contacts[ps_name][i_frame],
                    np.shape(manual_digit_map))
                auto_mask[digit_auto_mask] = i_digit

            # diff mask
            diff_mask = np.not_equal(auto_mask, manual_digit_map)
            ps_matrix_frame = ps_matrices[ps_name][i_frame]
            mask_based_diff_per_sensor[ps_name].append(np.sum(np.abs(ps_matrix_frame[diff_mask])))

            # manual unclaimed mask
            unclaimed_mask = manual_digit_map == (len(DIGITS) - 1)
            unclaimed_force[ps_name].append(np.sum(np.abs(ps_matrix_frame[unclaimed_mask])))

        mask_based_diff_per_sensor[ps_name] = np.array(mask_based_diff_per_sensor[ps_name])
        unclaimed_force[ps_name] = np.array(unclaimed_force[ps_name])
    # sum across sensors
    trial.mask_based_diff = (mask_based_diff_per_sensor['medial_sensor'] +
                             mask_based_diff_per_sensor['lateral_sensor'])
    trial.unclaimed_force = (unclaimed_force['medial_sensor'] +
                             unclaimed_force['lateral_sensor'])



def get_summed_force_data(tsm1_file, tsm2_file, verbose=False):

    # ---- ----- ---- ---- #

    # Handle different start/end times
    # load time and ps data for each file
    ps_times1, ps_matrices1 = io_tools.import_tsm_matrix(tsm1_file) #io_tools.import_matrices(tsm1_file)
    ps_times2, ps_matrices2 = io_tools.import_tsm_matrix(tsm2_file) #io_tools.import_matrices(tsm2_file)

    # Get the sums at each timestep
    ps_sum1 = np.sum(ps_matrices1, axis=(1, 2))
    ps_sum2 = np.sum(ps_matrices2, axis=(1, 2))

    # MEM ISSUE FIX
    ps_matrices1 = None
    ps_matrices2 = None

    # Sanity check that time and pressure data are the same size
    if ps_times1.size != ps_sum1.size:
        raise ValueError("Size of tsm1 time and pressure data not equal")
    if ps_times2.size != ps_sum2.size:
        raise ValueError("Size of tsm2 time and pressure data not equal")

    # find smallest common time range
    tmin = max([ps_times1[0], ps_times2[0]])
    tmax = min([ps_times1[-1], ps_times2[-1]])

    # trim both datasets to that range
    valid_idx1 = (ps_times1 >= tmin) & (ps_times1 <= tmax)
    ps_times1 = np.array(ps_times1[valid_idx1])
    ps_sum1 = np.array(ps_sum1[valid_idx1])

    valid_idx2 = (ps_times2 >= tmin) & (ps_times2 <= tmax)
    ps_times2 = np.array(ps_times2[valid_idx2])
    ps_sum2 = np.array(ps_sum2[valid_idx2])

    # Take union of times to get all common times
    U_times = np.union1d(ps_times1, ps_times2)
    max_size = max(ps_times1.size, ps_times2.size)
    added_times_pct = abs(U_times.size - max_size) / max_size
    if added_times_pct > 0.05:
        if verbose:
            ws(f"Times union size is greater 5% of the max \
            ps_times array: pct diff = ({added_times_pct})")

    # Interp the missing sums
    ps_sum1_fill = np.interp(U_times, ps_times1, ps_sum1)
    ps_sum2_fill = np.interp(U_times, ps_times2, ps_sum2)

    if ps_sum1_fill.size != ps_sum2_fill.size:
        raise ValueError(
            f"Len of ps sum 1 and 2 not equal "
            f"({ps_sum1_fill.size} / {ps_sum2_fill.size})"
        )

    # Left/Right force sums
    fss_total = np.sum([ps_sum1_fill, ps_sum2_fill], axis=0)

    return (
        U_times,
        fss_total
    )
