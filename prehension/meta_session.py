#!python3
import os
import sys
import re
import glob
import warnings
import json
import ncams


SENSOR_SERIAL1 = '00110-2743'
SENSOR_SERIAL2 = '00110-2746'


def find_session_dirs(dirname):
    sessions = []
    session_re = re.compile('[0-9]{4}_[0-9]{2}_[0-9]{2}.*')
    # process the directory
    for d in os.listdir(dirname):
        if session_re.fullmatch(d) is not None:
            sessions.append(d)
    return sessions


def get_default_meta_structure():
    '''Returns a default structure for a session.

    Recommend filling it with relevant:
        ps_dic
        ps_markers
    session-specific fields:
        opensim_model_locked_base
        mujoco_model_sensorized
        ncams_config

    CHECK:
        hand (right or left)

    The following can be filled in automatically using `fill_meta_structure()`:
        auto_log
        manual_log
        opensim_model_locked_base
        mujoco_model_sensorized
        cameras

    These are filled by preset in create_meta:
        ps_dic
        ps_markers
        hand
        fps
    '''
    return {
        'auto_log': '',
        'manual_log': '',
        'videos_dir': 'camera_videos',
        'images_dir': 'cameras',
        'cameras': {
            # 19194005: 'cam19194005',  # now handled by fill_meta_structure
            # 19194008: 'cam19194008',
            # 19194009: 'cam19194009',
            # 19335177: 'cam19335177',
            # 19340298: 'cam19340298',
            # 19340300: 'cam19340300',
            # 19340396: 'cam19340396',
            # 20050811: 'cam20050811',
        },
        'markers_2D_dir': 'markers_2D',
        'markers_2D_video_dir': 'markers_2D_videos',
        'markers_3D_dir': 'markers_3D',
        'pre_ja_dir': 'joint_angles',
        'post_ja_dir': 'aligned_joint_angles',
        'raw_ps_dir': 'sensors',
        'transformed_ps_dir': 'transformed_sensors',
        'pre_ps_dir': 'filtered_sensors',
        'post_ps_dir': 'aligned_sensors',
        'matched_contacts_dir': 'matched_contacts',
        'manually_labelled_forces_dir': 'manually_labelled',
        'scaling_dir': 'scaling',
        'digit_forces_dir': 'digit_forces',
        'segment_forces_dir': 'segment_forces',
        'ps_dic': {},
        'ps_trialname_template': '{ps_serial}_{trial_number}',
        'kin_trialname_template': 'trial{trial_number}',
        'kin_adjustment_suffix': '_adjust',
        'ps_log_filename': 'sensors/trial_log.csv',
        'opensim_model': '../opensim_models/RightArmAndHand_NoMuscles_Scaled.osim',
        'opensim_model_locked_base': '',  # session-specific
        'mujoco_model': '../mujoco_models/RightArmAndHand_NoMuscles_Scaled.xml',
        'mujoco_model_sensorized': '',  # session-specific
        'ncams_config': '',  # session-specific
        'calibration': 'calibration',  # local calibration directory
        'hand': 'right',
        'fps': 50,  # fill in
        'ps_markers': {
            'medial_sensor': ('o_sensor_tb', 'o_sensor_tf', 'o_sensor_bb', 'o_sensor_bf'),
            'lateral_sensor': ('b_sensor_tb', 'b_sensor_tf', 'b_sensor_bb', 'b_sensor_bf')
        },
        'object_def_columns': (
            'pos_translation_z(mm)', 'pos_tilt(deg)', 'pos_aperture(mm)')
    }


def fill_meta_structure(mstruct, dirname, session, log_rel_dir='behavior'):
    '''If the meta structure dictionary has empty auto_log and manual_log, script searches for them.
    Replacement is done in place.

    Also fills mujoco and opensim models, and identifies cameras
    '''
    if len(mstruct['auto_log']) == 0:
        # search automatically
        auto_log = glob.glob(os.path.join(
            dirname, log_rel_dir, 'session_*.csv'))
        if len(auto_log) > 1:
            # sort them
            def order(v):
                if re.match('.*\([0-9]+\).csv$', v) is not None:
                    return int(re.findall('\([0-9]+\)', v)[-1][1:-1])
                return -1

            auto_log.sort(key=order)

            warnings.warn('Several session log filenames found: {}'.format(auto_log))
        elif len(auto_log) == 0:
            raise ValueError('Could not find auto log session filenames in {}.'.format(dirname))

        mstruct['auto_log'] = [os.path.join(log_rel_dir, os.path.basename(al)) for al in auto_log]

    if len(mstruct['manual_log']) == 0:
        # search automatically
        manual_log = glob.glob(os.path.join(
            dirname, log_rel_dir, '*Daily experiment log - trials*.csv'))
        if len(manual_log) == 0:
            manual_log = glob.glob(os.path.join(
                dirname, log_rel_dir, 'Manual_*.csv'))
        if len(manual_log) > 1:
            warnings.warn(
                'Too many manual session log filenames found: {}. Using first one.'.format(
                    manual_log))
            mstruct['manual_log'] = os.path.join(log_rel_dir, os.path.basename(manual_log[0]))
        elif len(manual_log) == 0:
            warnings.warn('Could not find manual session log filenames in {}.'.format(dirname))
        else:
            mstruct['manual_log'] = os.path.join(log_rel_dir, os.path.basename(manual_log[0]))

    mstruct['opensim_model_locked_base'] = '{}_locked_{}.osim'.format(
        mstruct['opensim_model'][:-5], session)
    mstruct['mujoco_model_sensorized'] = '{}_Tessellated_{}.xml'.format(
        mstruct['mujoco_model'][:-4], session)

    # identify cameras
    if os.path.exists(os.path.join(dirname, mstruct['videos_dir'])):
        # TODO suboptimal
        _cameras = glob.glob(os.path.join(dirname, mstruct['videos_dir'], 'trial*', 'cam*.mp4'))
        _cameras_dict = {}
        for _c in _cameras:
            camera = os.path.split(_c)[1]
            try:
                serial = int(camera[3:-4])
            except Exception as e:
                continue
            _cameras_dict[serial] = camera[:-4]
    else:
        _cameras = glob.glob(os.path.join(dirname, mstruct['images_dir'], 'cam*'))
        _cameras_dict = {}
        for _c in _cameras:
            camera = os.path.split(_c)[1]
            try:
                serial = int(camera[3:])
            except Exception as e:
                continue
            _cameras_dict[serial] = camera
    mstruct['cameras'] = _cameras_dict


def normpath(dirname, p):
    if len(p) == 0:
        return None
    return os.path.normpath(os.path.join(dirname, p))


def import_meta_structure(dirname):
    filename = os.path.join(dirname, 'meta_structure.json')
    with open(filename, 'r') as f:
        mstruct = json.load(f)

    # resolve relative paths
    paths_to_resolve = (
        'auto_log', 'manual_log',
        'videos_dir', 'images_dir', 'markers_2D_dir', 'markers_2D_video_dir', 'markers_3D_dir',
        'pre_ja_dir', 'post_ja_dir', 'ps_log_filename',
        'raw_ps_dir', 'transformed_ps_dir', 'pre_ps_dir', 'post_ps_dir',
        'matched_contacts_dir', 'manually_labelled_forces_dir', 'scaling_dir',
        'opensim_model', 'opensim_model_locked_base',
        'mujoco_model', 'mujoco_model_sensorized',
        'calibration',
        'digit_forces_dir', 'segment_forces_dir'
    )
    for ptr in paths_to_resolve:
        if ptr not in mstruct.keys():
            continue
        # in case one has multiple elements
        if isinstance(mstruct[ptr], (list, tuple)):
            mstruct[ptr] = [normpath(dirname, p) for p in mstruct[ptr]]
        else:
            mstruct[ptr] = normpath(dirname, mstruct[ptr])

    ###############################################################################################
    # HACK FOR saving videos
    ###############################################################################################
    # should only work for *_camera_copy presets
    # mojito lhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Spring_2021',
        'Recording_sessions', 'Mojito')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'MojitoLeftHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    # pimms rhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Pimms',
        'RightHem_Recordings')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'PimmsRightHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    # mojito rhem
    default_server = os.path.join(
        r'\\BENSMAIA-LAB', 'LabSharing', 'Stereognosis', 'Data', 'Mojito',
        'RightHem_Recordings')
    new_default_server = os.path.join(
        r'\\192.170.210.120', 'Data', 'ProjectFolders', 'Prehension', 'MojitoRightHemisphere',
        'sessions')
    if default_server in mstruct['videos_dir']:
        mstruct['videos_dir'] = mstruct['videos_dir'].replace(default_server, new_default_server)

    return mstruct


def import_meta_object(dirname):
    filename = os.path.join(dirname, 'meta_object.csv')
    column_names, values = ncams.io_utils.import_csv(filename)
    object_ids = values[column_names.index('id')]
    object_def_columns = [v for v in column_names if v != 'id']

    # make a dictionary of all objects
    answ = {}
    for i_object, object_id in enumerate(object_ids):
        answ[object_id] = {'def': {}}
        for odc in object_def_columns:
            answ[object_id]['def'][odc] = values[column_names.index(odc)][i_object]
        answ[object_id]['sstr'] = ' '.join(
            str(v) for v in answ[object_id]['def'].values())
        answ[object_id]['str'] = ', '.join(
            '{}: {}'.format(k, v) for k, v in answ[object_id]['def'].items())
    return answ


def import_meta_dof(dirname):
    filename = os.path.join(dirname, 'meta_dof.csv')
    column_names, values = ncams.io_utils.import_csv(filename)

    i_dofname = column_names.index('dof_name')
    i_rmin = column_names.index('range_min')
    i_rmax = column_names.index('range_max')
    i_rot = column_names.index('rotation')

    mdof = {name: {'range': [rmin, rmax],
                   'rot': rot
                   }
            for name, rmin, rmax, rot
            in zip(values[i_dofname], values[i_rmin], values[i_rmax], values[i_rot])}
    return mdof


def import_manual_log(filename):  # mstruct['manual_log']
    column_names, values = ncams.io_utils.import_csv(filename, cast=str)
    mlog = {int(trial_number): code.split(',')
            for trial_number, code in zip(values[column_names.index('Trial')],
                                          values[column_names.index('Code')])}
    return mlog


class TrialInfo():
    """Contains all relevant information about a trial, including all filenames"""
    def __init__(self, session, trial_number, object_id, success, other_info=None):
        self.session = session
        self.trial_number = int(trial_number)
        self.object_id = int(object_id)
        self.success = int(success)

        # process additional parameters
        if other_info is not None:
            for k, v in other_info.items():
                setattr(self, k, v)

    def generate_filenames(self, mstruct, dirname):
        trial_name = mstruct['kin_trialname_template'].format(trial_number=self.trial_number)
        self.trial_name = trial_name

        # recorded images from all cameras
        self.images_dirnames = {
            k: os.path.join(mstruct['images_dir'], v, trial_name)
            for k, v in mstruct['cameras'].items()
        }
        self.images_logs = {
            k: os.path.join(mstruct['images_dir'], v, trial_name, v + '.csv')
            for k, v in mstruct['cameras'].items()
        }
        # recorded videos
        self.videos = {
            k: os.path.join(mstruct['videos_dir'], trial_name, v + '.mp4')
            for k, v in mstruct['cameras'].items()
        }
        self.videos_logs = {
            k: os.path.join(mstruct['videos_dir'], trial_name, v + '.csv')
            for k, v in mstruct['cameras'].items()
        }

        # directory with 2D labelled CSVs
        self.markers_2D_dirname = os.path.join(
            mstruct['markers_2D_dir'], trial_name)
        self.dlc_filemasks = {
            k: os.path.join(self.markers_2D_dirname, v + '*')
            for k, v in mstruct['cameras'].items()
        }
        self.markers_2D_filemasks = {
            k: os.path.join(self.markers_2D_dirname, v + '*.csv')
            for k, v in mstruct['cameras'].items()
        }
        self.markers_2D_marker_video_filemasks = {
            k: os.path.join(self.markers_2D_dirname, v + '*_labeled.mp4')
            for k, v in mstruct['cameras'].items()
        }

        # scaling files all go together
        self.scaling_markers_3D_filename_trc = os.path.join(
            mstruct['scaling_dir'], trial_name + '.trc')
        self.scaling_ik_filename = os.path.join(
            mstruct['scaling_dir'], trial_name + '_IK.xml')
        self.scaling_sc_filename = os.path.join(
            mstruct['scaling_dir'], trial_name + '_SC.xml')
        self.scaling_kinematic_filename = os.path.join(
            mstruct['scaling_dir'], trial_name + '.mot')

        # add .csv or .trc depending on use
        self.markers_3D_filename_csv = os.path.join(
            mstruct['markers_3D_dir'], trial_name + '.csv')
        self.markers_3D_filename_trc = os.path.join(
            mstruct['markers_3D_dir'], trial_name + '.trc')
        self.ik_filename = os.path.join(
            mstruct['pre_ja_dir'], trial_name + '_IK.xml')
        self.base_ik_filename = os.path.join(
            mstruct['pre_ja_dir'], trial_name + '_base_IK.xml')
        self.ik_log_filename = os.path.join(
            mstruct['pre_ja_dir'], trial_name + '.log')

        # before filtering and aligning
        self.pre_kinematic_filename = os.path.join(
            mstruct['pre_ja_dir'], trial_name + '.mot')
        self.base_kinematic_filename = os.path.join(
            mstruct['pre_ja_dir'], trial_name + '_base.mot')
        # needs .mot or .csv
        self.post_kinematic_filename_mot = os.path.join(
            mstruct['post_ja_dir'], trial_name + '.mot')
        self.post_kinematic_filename_csv = os.path.join(
            mstruct['post_ja_dir'], trial_name + '.csv')

        # created by MuJoCo adjustment program
        self.adjustment_kinematic_filename = os.path.join(
            mstruct['post_ja_dir'],
            trial_name + mstruct['kin_adjustment_suffix'] + '.csv')

        # pressure sensor data
        # new format - relying on TSM
        self.raw_ps_filenames = {}
        self.transformed_ps_filenames = {}
        self.transformed_ps_csv_filenames = {}
        self.filtered_ps_filenames = {}
        self.aligned_ps_filenames = {}

        # the following are deprecated and should be avoided
        self.pre_ps_filenames = {}
        self.post_ps_filenames = {}
        self.pre_ps_tsm_filenames = {}
        self.post_ps_tsm_filenames = {}

        self.matched_contacts_filenames = {}


        for ps_name, ps_serial in mstruct['ps_dic'].items():
            ps_trial_name = mstruct['ps_trialname_template'].format(
                trial_number=self.trial_number, ps_serial=ps_serial)

            self.raw_ps_filenames[ps_name] = os.path.join(
                mstruct['raw_ps_dir'], ps_trial_name + '.fsx')
            self.transformed_ps_filenames[ps_name] = os.path.join(
                mstruct['transformed_ps_dir'], ps_trial_name + '.tsm')
            self.transformed_ps_csv_filenames[ps_name] = os.path.join(
                mstruct['transformed_ps_dir'], ps_trial_name + '.csv')
            self.filtered_ps_filenames[ps_name] = os.path.join(
                mstruct['pre_ps_dir'], ps_trial_name + '.tsm')
            self.aligned_ps_filenames[ps_name] = os.path.join(
                mstruct['post_ps_dir'], ps_trial_name + '.tsm')

            self.pre_ps_filenames[ps_name] = os.path.join(
                mstruct['pre_ps_dir'], ps_trial_name + '.csv')
            self.post_ps_filenames[ps_name] = os.path.join(
                mstruct['post_ps_dir'], ps_trial_name + '.csv')
            self.pre_ps_tsm_filenames[ps_name] = os.path.join(
                mstruct['pre_ps_dir'], ps_trial_name + '.tsm')
            self.post_ps_tsm_filenames[ps_name] = os.path.join(
                mstruct['post_ps_dir'], ps_trial_name + '.tsm')

            self.matched_contacts_filenames[ps_name] = os.path.join(
                mstruct['matched_contacts_dir'], ps_trial_name + '.csv')

        # manually labeled forces
        self.manually_labelled_filename = os.path.join(
            mstruct['manually_labelled_forces_dir'], trial_name + '.csv')
        self.lps_map_filename = os.path.join(
            mstruct['manually_labelled_forces_dir'], trial_name + '_lps.csv')
        self.rps_map_filename = os.path.join(
            mstruct['manually_labelled_forces_dir'], trial_name + '_rps.csv')

        # markers for the thorax estimate
        # if there is a local session calibration, use that, otherwise with NCams
        extrinsic_calibration_filename = os.path.join(
            mstruct['calibration'], 'extrinsic', 'extrinsic_calib.pickle')
        if os.path.exists(extrinsic_calibration_filename):
            calibration_dir = mstruct['calibration']
        elif mstruct['ncams_config'] is not None and len(mstruct['ncams_config']) > 0:
            calibration_dir = os.path.split(mstruct['ncams_config'])[0]
        else:
            calibration_dir = None
        if calibration_dir is not None:
            self.calib_base_marker_filename = os.path.join(
                calibration_dir, 'base', self.session, trial_name + '.json')
            self.calib_base_markers_3D_filename_trc = os.path.join(
                calibration_dir, 'base', self.session, trial_name + '.trc')
            self.calib_base_ik_filename = os.path.join(
                calibration_dir, 'base', self.session, trial_name + '_IK.xml')
            self.calib_base_ik_log_filename = os.path.join(
                calibration_dir, 'base', self.session, trial_name + '.log')
            self.calib_base_kinematic_filename = os.path.join(
                calibration_dir, 'base', self.session, trial_name + '.mot')
        else:
            self.calib_base_marker_filename = None
            self.calib_base_markers_3D_filename_trc = None
            self.calib_base_ik_filename = None
            self.calib_base_ik_log_filename = None
            self.calib_base_kinematic_filename = None

        # digit forces - compiled from matched_contacts
        self.digit_forces_filename = os.path.join(
            os.path.join(mstruct['digit_forces_dir'], trial_name + '.csv'))
        self.segment_forces_filename = os.path.join(
            os.path.join(mstruct['segment_forces_dir'], trial_name + '.csv'))

    # IMAGES
    def do_images_dirs_files_exist(self):
        for d in self.images_dirnames.values():
            if not os.path.exists(d):
                return False
        for f in self.images_logs.values():
            if not os.path.exists(f):
                return False
        return True

    # VIDEOS
    def do_videos_files_exist(self):
        for d in self.videos.values():
            if not os.path.exists(d):
                return False
        for f in self.videos_logs.values():
            if not os.path.exists(f):
                return False
        return True

    # DLC files
    @staticmethod
    def get_dlc_filenames(filemask):
        candidates = glob.glob(filemask)
        return candidates

    def get_dlc_filenames_all(self):
        return {
            k: TrialInfo.get_dlc_filenames(v)
            for k, v in self.dlc_filemasks.items()
        }

    def do_dlc_files_exist(self):
        return not any([len(v) == 0 for v in self.get_dlc_filenames_all().values()])

    def remove_dlc_files(self):
        '''Will clean dlc marker files'''
        for fnames in self.get_dlc_filenames_all().values():
            for fname in fnames:
                os.remove(fname)

    # 2D MARKERS
    @staticmethod
    def get_2d_filename(filemask):
        candidates = glob.glob(filemask)
        if len(candidates) == 0:
            return None
        return candidates[0]  # what if there are more than one?

    def get_2d_filenames(self):
        return {
            k: TrialInfo.get_2d_filename(v)
            for k, v in self.markers_2D_filemasks.items()
        }

    def do_2d_files_exist(self):
        return None not in self.get_2d_filenames().values()

    def remove_2d_files_all(self):
        '''Will clean anything that looks like a 2D marker file'''
        for filemask in self.markers_2D_filemasks.values():
            for candidate in glob.glob(filemask):
                os.remove(candidate)

    def remove_2d_files(self):
        '''Will clean 2D marker files'''
        for fname in self.get_2d_filenames().values():
            os.remove(fname)

    # videos with 2D markers
    @staticmethod
    def get_2d_marker_video_filename(filemask):
        candidates = glob.glob(filemask)
        if len(candidates) == 0:
            return None
        return candidates[0]  # what if there are more than one?

    def get_2d_marker_video_filenames(self):
        return {
            k: TrialInfo.get_2d_marker_video_filename(v)
            for k, v in self.markers_2D_marker_video_filemasks.items()
        }

    def do_2d_marker_video_files_exist(self):
        return None not in self.get_2d_marker_video_filenames().values()

    def remove_2d_marker_video_files(self):
        '''Will clean 2D marker files'''
        for fname in self.get_2d_marker_video_filenames().values():
            os.remove(fname)

    # 3D MARKERS
    def do_3d_files_exist(self):
        if not os.path.exists(self.markers_3D_filename_csv):
            return False
        if not os.path.exists(self.markers_3D_filename_trc):
            return False
        return True

    def do_pre_ik_files_exist(self):
        if not os.path.exists(self.markers_3D_filename_trc):
            return False
        if not os.path.exists(self.ik_filename):
            return False
        return True

    # JOINT ANGLES
    def does_post_ik_file_exists(self):
        if not os.path.exists(self.pre_kinematic_filename):
            return False
        return True

    def do_pre_base_ik_files_exist(self):
        if not os.path.exists(self.markers_3D_filename_trc):
            return False
        if not os.path.exists(self.base_ik_filename):
            return False
        return True

    # PROCESSED JOINT ANGLES
    def does_post_base_ik_file_exists(self):
        if not os.path.exists(self.base_kinematic_filename):
            return False
        return True

    def does_pre_kin_file_exist(self):
        return self.does_post_ik_file_exists()

    def does_post_kin_file_exist(self):
        if not os.path.exists(self.post_kinematic_filename_mot):
            return False
        if not os.path.exists(self.post_kinematic_filename_csv):
            return False
        return True

    # PRESSURE
    def do_pre_ps_files_exist(self):
        answ = True
        for filename in self.filtered_ps_filenames.values():
            if not os.path.exists(filename):
                answ = False
                break
        return answ

    def get_pre_ps_filenames(self):
        return self.filtered_ps_filenames

    # PREPROCESSED DATA
    def do_all_pre_files_exist(self):
        return self.does_pre_kin_file_exist() and self.do_pre_ps_files_exist()

    # PROCESSED PRESSURE
    def do_post_ps_files_exist(self):
        answ = True
        # for filename, filename_tsm in zip(self.post_ps_filenames.values(),
        #                                   self.post_ps_tsm_filenames.values()):
        #     if not os.path.exists(filename) and not os.path.exists(filename_tsm):
        #         answ = False
        #         break
        for filename in self.aligned_ps_filenames.values():
            if not os.path.exists(filename):
                answ = False
                break
        return answ

    def get_post_ps_filenames(self):
        # # prioritize TSM
        # post_ps_filenames = {}
        # for ps_name in self.post_ps_filenames.keys():
        #     if os.path.exists(self.post_ps_tsm_filenames[ps_name]):
        #         post_ps_filenames[ps_name] = self.post_ps_tsm_filenames[ps_name]
        #     else:
        #         post_ps_filenames[ps_name] = self.post_ps_filenames[ps_name]
        return self.aligned_ps_filenames

    # DATA POST-PROCESSING
    def do_all_post_files_exist(self):
        return self.does_post_kin_file_exist() and self.do_post_ps_files_exist()

    # AUTOMATICALLY MATCHED CONTACTS
    def do_matched_contacts_files_exist(self):
        answ = True
        for filename in self.matched_contacts_filenames.values():
            if not os.path.exists(filename):
                answ = False
                break
        return answ

    # MANUALLY LABELLED FORCES
    def does_manually_labelled_file_exists(self):
        if not os.path.exists(self.manually_labelled_filename):
            return False
        return True

    # SCALING FILES 3D->JA
    def do_scaling_files_exist(self):
        if not os.path.exists(self.scaling_markers_3D_filename_trc):
            return False
        if not os.path.exists(self.scaling_ik_filename):
            return False
        if not os.path.exists(self.scaling_sc_filename):
            return False
        return True

    def does_digit_force_file_exist(self):
        if os.path.exists(self.digit_forces_filename):
            return True
        return False

    def does_segment_force_file_exist(self):
        if os.path.exists(self.segment_forces_filename):
            return True
        return False


def _column_pop(k, column_names, values):
    i_k = column_names.index(k)
    answ = values[i_k]
    del column_names[i_k]
    del values[i_k]
    return answ


def load_meta_information(dirname, trial_subset=None, only_successful_trials=False,
                          check_manual_log=False, session=None):
    # find the session name if it was None
    if session is None:
        session = os.path.basename(dirname)

    mstruct = import_meta_structure(dirname)
    mdof = import_meta_dof(dirname)
    mobject = import_meta_object(dirname)
    #TODO extract session here and pass to trial instead of in each trialinfo

    # meta session
    meta_session_filename = os.path.join(dirname, 'meta_session.csv')
    column_names, values = ncams.io_utils.import_csv(meta_session_filename)

    # essential trial parameters
    trial_numbers = _column_pop('trial_number', column_names, values)
    successs = _column_pop('success', column_names, values)
    object_ids = _column_pop('object_id', column_names, values)

    # load manual log
    if check_manual_log:
        if mstruct['manual_log'] is None:
            warnings.warn('No manual log specified in session structure, cannot check it.')
            check_manual_log = False
        else:
            try:
                mlog = import_manual_log(mstruct['manual_log'])
                # total fail or multigrasp or multireach or ?
                mlog_failed_numbers = ['0', '2', '3', '4', '5', '?', '14']
            except Exception as e:
                warnings.warn('Could not load manual log.')
                check_manual_log = False

    msession = []
    for i_trial, (trial_number, success, object_id) in enumerate(zip(
            trial_numbers, successs, object_ids)):
        if only_successful_trials and not success:
            continue
        if only_successful_trials and check_manual_log:
            if any([mfn in mlog[trial_number] for mfn in mlog_failed_numbers]):
                continue
        msession.append(TrialInfo(session, trial_number, object_id, success,
                                  other_info={k: v[i_trial] for k, v in zip(column_names, values)}))
        msession[-1].generate_filenames(mstruct, dirname)

    return mstruct, mdof, mobject, msession


def import_adjustment_trials(dirname):
    if not os.path.exists(os.path.join(dirname, 'adjustment_files.csv')):
        return {}

    column_names, values = ncams.io_utils.import_csv(os.path.join(dirname, 'adjustment_files.csv'))

    trial_numbers = [int(v) for v in values[column_names.index('trial_number')]]
    adjustment_trials = [int(v) for v in values[column_names.index('adjustment_trial')]]

    return {k: v for k, v in zip(trial_numbers, adjustment_trials)}


def find_trial(msession, trial_number):
    trial = None
    for t in msession:
        if t.trial_number == trial_number:
            trial = t
            break
    return trial


def get_trial_log_info(mstruct, trial_number, column_names):
    if not isinstance(column_names, (list, tuple)):
        column_names = [column_names]

    sy_column_names, sy_data = ncams.utils.import_csv(mstruct['auto_log'][0])
    # sy_data = np.array(sy_data).transpose()

    # TODO check if the trial not in the list
    row = sy_data[sy_column_names.index('trial_num')].index(trial_number)

    column_ids = [sy_column_names.index(cn) for cn in column_names]

    return [sy_data[ci][row] for ci in column_ids]
