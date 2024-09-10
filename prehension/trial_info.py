#!python3
import glob
import os


class TrialInfo():
    """Contains all relevant information about a trial, including all filenames"""
    def __init__(self, session, trial_number, object_id, success, other_info=None):
        self.session = session
        self.trial_number = int(trial_number)
        self.object_id = int(object_id)
        self.success = int(success)

        # process additional parameters
        self.other_info = other_info
        if other_info is not None:
            for k, v in other_info.items():
                setattr(self, k, v)


    def generate_filenames(self, mstruct):
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
        if mstruct['videos_dir'] != mstruct['images_dir']:
            # old style structure - different folders for images and videos
            self.videos = {
                k: form_cam_fname(mstruct['videos_dir'], trial_name, v, '.mp4')
                for k, v in mstruct['cameras'].items()}

            self.videos_logs = {
                k: form_cam_fname(mstruct['videos_dir'], trial_name, v, '.csv')
                for k, v in mstruct['cameras'].items()}
        else:
            # inverted_dir_structure
            self.videos = {
                k: form_cam_inverted_fname(mstruct['videos_dir'], trial_name, v, '.mp4')
                for k, v in mstruct['cameras'].items()}
            self.videos_logs = {
                k: form_cam_inverted_fname(mstruct['videos_dir'], trial_name, v, '.csv')
                for k, v in mstruct['cameras'].items()}

        self.jarvis_video_dir = os.path.join(mstruct['jarvis_video_dir'], trial_name)
        self.jarvis_videos = {
            k: os.path.join(self.jarvis_video_dir, v + '.mp4')
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
        self.markers_3D_filename_jarvis_csv = os.path.join(
            mstruct['markers_3D_jarvis_dir'], trial_name + '.csv')
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
            mstruct['digit_forces_dir'], trial_name + '.csv')
        self.segment_forces_filename = os.path.join(
            mstruct['segment_forces_dir'], trial_name + '.csv')

        self.mujoco_video = os.path.join(
            mstruct['mujoco_videos_dir'], trial_name + '.mp4')

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


def form_cam_fname(vid_dir, trial_name, cam_id, ext):
    fn1 = os.path.join(vid_dir, trial_name, cam_id + '.avi')
    if os.path.exists(fn1):
        return fn1
    return os.path.join(vid_dir, trial_name, cam_id + ext)


def form_cam_inverted_fname(vid_dir, trial_name, cam_id, ext):
    fn1 = os.path.join(vid_dir, cam_id, trial_name, cam_id + '.avi')
    if os.path.exists(fn1):
        return fn1
    return os.path.join(vid_dir, cam_id, trial_name, cam_id + ext)
