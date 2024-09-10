#!python3.8
import os
import xml.etree.ElementTree as ET

import tqdm
from reporting_pool import ReportingPool

from . import meta_session
from . import tools
from .tools import rs, ws

# Default accuracy of 1e-5 does not produce precise enough results for hand and finger movements.
IK_XML_STR = '''\
<?xml version="1.0" encoding="UTF-8" ?>
<OpenSimDocument Version="40000">
    <InverseKinematicsTool>
        <!--Directory used for writing results.-->
        <results_directory>./</results_directory>
        <!--Directory for input files-->
        <input_directory />
        <!--Name of the model file (.osim) to use for inverse kinematics.-->
        <model_file>{model_file}</model_file>
        <!--A positive scalar that weights the relative importance of satisfying constraints. A weighting of 'Infinity' (the default) results in the constraints being strictly enforced. Otherwise, the weighted-squared constraint errors are appended to the cost function.-->
        <constraint_weight>Inf</constraint_weight>
        <!--The accuracy of the solution in absolute terms. Default is 1e-5. It determines the number of significant digits to which the solution can be trusted.-->
        <accuracy>1.0000000000000001e-06</accuracy>
        <adaptiveAccuracy>true</adaptiveAccuracy>
        <ignoreConvergenceErrors>true</ignoreConvergenceErrors>
        <!--Markers and coordinates to be considered (tasks) and their weightings. The sum of weighted-squared task errors composes the cost function.-->
        <IKTaskSet>
            <objects />
            <groups />
        </IKTaskSet>
        <!--TRC file (.trc) containing the time history of observations of marker positions obtained during a motion capture experiment. Markers in this file that have a corresponding task and model marker are included.-->
        <marker_file>Unassigned</marker_file>
        <!--The name of the storage (.sto or .mot) file containing the time history of coordinate observations. Coordinate values from this file are included if there is a corresponding model coordinate and task. -->
        <coordinate_file>Unassigned</coordinate_file>
        <!--The desired time range over which inverse kinematics is solved. The closest start and final times from the provided observations are used to specify the actual time range to be processed.-->
        <time_range> 0 1</time_range>
        <!--Flag (true or false) indicating whether or not to report marker errors from the inverse kinematics solution.-->
        <report_errors>true</report_errors>
        <!--Name of the resulting inverse kinematics motion (.mot) file.-->
        <output_motion_file>out_inv_kin.mot</output_motion_file>
        <!--Flag indicating whether or not to report model marker locations. Note, model marker locations are expressed in Ground.-->
        <report_marker_locations>false</report_marker_locations>
    </InverseKinematicsTool>
</OpenSimDocument>
'''


def make_ik_file(filename, ik_xml_str, marker_weights, trc_file, ik_out_mot_file, time_range,
                 verbose=0):
    if ik_xml_str is None:
        ik_xml_str = IK_XML_STR.format(model_file="Unassigned")
    if verbose > 0:
        print('Making IK file {}'.format(filename))

    # check the basic elements
    root = ET.fromstring(ik_xml_str)
    if root.tag != 'OpenSimDocument':
        raise ValueError('Wrong structure of the IK string. OpenSimDocument is not present at '
                         'top-level.')

    ikt = root.find('InverseKinematicsTool')
    if ikt is None:
        raise ValueError('Wrong structure of the IK string. InverseKinematicsTool is not present.')

    # add the IK task and objects (markers) if missing
    ikts = _add_xml_element(ikt, 'IKTaskSet')
    _add_xml_element(ikts, 'groups')
    iktso = _add_xml_element(ikts, 'objects', text='\n' + ' '*16, tail='\n' + ' '*12)

    # add each marker with weights
    for marker_name, marker_weight in marker_weights.items():
        mare = _add_xml_element(iktso, 'IKMarkerTask',
                                text='\n' + ' '*20, tail='\n' + ' '*16,
                                unique=False)
        mare.set('name', marker_name)

        _add_xml_element(mare, 'weight', text=str(marker_weight), tail='\n' + ' '*20)

        if marker_weight < 1e-8:
            _add_xml_element(mare, 'apply', text='false', tail='\n' + ' '*16)
        else:
            _add_xml_element(mare, 'apply', text='true', tail='\n' + ' '*16)

    # other elements
    _add_xml_element(ikt, 'time_range', text='{} {}'.format(time_range[0], time_range[1]))
    _add_xml_element(ikt, 'marker_file', text=trc_file)
    _add_xml_element(ikt, 'output_motion_file', text=ik_out_mot_file)

    tree = ET.ElementTree(element=root)
    tree.write(filename, encoding='UTF-8', xml_declaration=True)


def _add_xml_element(parent, name, text=None, tail=None, unique=True):
    '''Adds an element to the parent. If unique, checks if such element already exists in the parent
    and does not add it if it does. Otherwise always adds.
    '''
    if unique:
        el = parent.find(name)
    if not unique or el is None:
        el = ET.Element(name)
        parent.append(el)
    if text is not None:
        el.text = text
    if tail is not None:
        el.tail = tail
    return el


def set_opensim_model_default_position(osim_model_in, osim_model_ou, positions, lock=False):
    tree = ET.parse(osim_model_in)
    root = tree.getroot()

    for dof_name, position in positions.items():
        coordinate = root.find(".//Coordinate[@name='{}']".format(dof_name))
        c_defval = coordinate.find("default_value")
        c_defval.text = str(position)
        if lock:
            c_locked = coordinate.find("locked")
            c_locked.text = 'true'

    tree.write(osim_model_ou, encoding='UTF-8', xml_declaration=True)


def run_ik_f(ik_file, log_file):
    # if log file exists, remove it
    if os.path.exists(log_file):
        os.remove(log_file)

    # needed for independent logging
    import opensim
    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(log_file)
    opensim.Logger.setLevelString('warn')
    task = opensim.tools.InverseKinematicsTool(ik_file)

    task.run()


def inverse_kinematics(server, sessions, trials_sel, temp, processes, overwrite, base):
    """Runs the inverse kinematics OpenSim tool.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        processes {int} --- Number of parallel processes in the pool.
        overwrite {bool} --- Overwrites the created files if they exist.
        base {bool} --- Runs inverse kinematics on the most proximal markers
            that can be used to estimate the default static thorax position.
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
            _, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        trials = []
        ik_files = []
        log_files = []
        for trial in msession:
            if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
                continue
            if base:
                if not trial.do_pre_base_ik_files_exist():
                    continue
                if not overwrite and trial.does_post_base_ik_file_exists():
                    continue
                ik_files.append(trial.base_ik_filename)
            else:
                if not trial.do_pre_ik_files_exist():
                    continue
                if not overwrite and trial.does_post_ik_file_exists():
                    continue
                ik_files.append(trial.ik_filename)
            log_files.append(trial.ik_log_filename)
            trials.append(trial)

        rs('Found {} trials: {}'.format(
            len(trials), ', '.join([str(trial.trial_number) for trial in trials])))

        p_args = list(zip(*[ik_files, log_files]))

        if len(p_args) > 0:
            pool = ReportingPool(run_ik_f, p_args, processes=processes,
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
