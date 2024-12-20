#!python3
# -*- coding: utf-8 -*-
"""
Provides utilities for session visualization such as session status and exising files within a
given session.

If your command line is not displaying the tick marks, change the default font to something like
Cascadia Code.


Copyright (C) 2019-2024 Anton Sobinov, Caleb Raman
https://github.com/BensmaiaLab/prehension

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
import os
import sys

from colorama import Fore, Style

YES = f'{Fore.GREEN}✓{Style.RESET_ALL}'
NO = f'{Fore.RED}x{Style.RESET_ALL}'


####################################### Callable classes to evaluate stages of processing
class SessionProcessedBase():
    '''
    Children have to set
        header
        _type
    and implement either
        _eval_trial (if _type == per_trial)
        _eval_per_session (if _type == per_session_one_symbol or per_session_num_trials)
    '''
    header = 'UNDEFINED'
    _type = 'NOT IMPLEMENTED'
    max_trial_symbols = 3  # maximum number of symbols needed to describe all trials in a session

    def __init__(self):
        self._select_type_methods()
        self._skip_the_rest = False

    def _select_type_methods(self):
        # pyflakes complains about match
        match self._type:
            case 'per_trial':
                self.header_1 = self._header_1_per_trial
                self.header_2 = self._header_2_per_trial
                self.width = self._width_per_trial
                self.eval = self._eval_per_trial
            case 'per_session_one_symbol':
                self.header_1 = self._header_1_per_session
                self.header_2 = self._header_2_per_session
                self.width = self._width_one_symbol
                self.eval = self._eval_per_session
            case 'per_session_num_trials':
                self.header_1 = self._header_1_per_session
                self.header_2 = self._header_2_per_session
                self.width = self._width_num_trials
                self.eval = self._eval_per_session
            case _:
                raise ValueError('Wrong type set:', self._type)

    def skip_the_rest(self):
        '''If returns True, the rest of SessionProcesses are skipped for this session. The
        variable should be set during the session call.

        Almost never should be set to True, except for checking for critical steps like having
        meta files.
        '''
        return self._skip_the_rest

    # CALLS and EVALS
    def _eval_trial(self, trial):
        """Return
            -1 if previous does not exist,
            0 if exists, but child does not,
            1 if child exists,
            2 if child was made after the parents  TODO(AS)
        """
        raise NotImplementedError()
        return -1

    def eval(self, sw, preset):
        '''Returns a string evaluation of 1 session.

        If general, returns general answer, otherwise - stats per trial
        '''
        raise NotImplementedError()
        return ''

    def _eval_per_trial(self, sw, preset):
        numsucces = sum([trial.success for trial in sw.msession])
        reports = [self._eval_trial(t) for t in sw.msession]
        resp_neg = sum([v == -1 for v in reports])
        resp_zer = sum([v == 0 for v in reports])
        resp_pos = sum([v > 0 for v in reports])
        resp_cor = sum([v == 2 for v in reports])
        color_pos = _color_resp_trials(resp_pos, numsucces)
        color_cor = _color_resp_trials(resp_cor, numsucces)
        return (
            f'{resp_neg:>{self.max_trial_symbols}}' +
            f'{resp_zer:>{self.max_trial_symbols+1}}' +
            f'{color_pos}{resp_pos:>{self.max_trial_symbols+1}}{Style.RESET_ALL}' +
            f'{color_cor}{resp_cor:>{self.max_trial_symbols+1}}{Style.RESET_ALL}')

    def _eval_per_session(self, sw, preset):
        raise NotImplementedError()
        return ''

    # HEADERS FORMATTING
    def header_1(self):
        raise NotImplementedError()
        return ''

    def _header_1_per_trial(self):
        return f'{self.header:^{self.width()}}'

    def _header_1_per_session(self):
        return f"{self.header[:self.width()]:^{self.width()}}"

    def header_2(self):
        raise NotImplementedError()
        return ''

    def _header_2_per_trial(self):
        return (
            f'{"NEG":>{self.max_trial_symbols}}' +
            f'{"ZER":>{self.max_trial_symbols+1}}' +
            f'{"POS":>{self.max_trial_symbols+1}}' +
            f'{"COR":>{self.max_trial_symbols+1}}')

    def _header_2_per_session(self):
        return ' ' * self.width()

    # WIDTH
    def width(self):
        """Width of one column"""
        raise NotImplementedError()
        return 0

    def _width_per_trial(self):
        return (self.max_trial_symbols + 1) * 4 - 1

    def _width_one_symbol(self):
        return 1

    def _width_num_trials(self):
        return self.max_trial_symbols


# ----------------- per trial classes
class SpMarkers2D(SessionProcessedBase):
    header = 'Markers 2D'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_videos_files_exist():
            return -1
        if not trial.do_2d_files_exist():
            return 0
        latest_pre_file = max(trial.video_file_times())
        oldest_pos_file = min(trial.m_2d_file_times())
        if latest_pre_file > oldest_pos_file:
            return 1
        return 2



class SpMarkers3D(SessionProcessedBase):
    header = 'Markers 3D'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_2d_files_exist():
            return -1
        if not trial.do_3d_files_exist():
            return 0
        latest_pre_file = max(trial.m_2d_file_times())
        oldest_pos_file = min(trial.m_3d_files_times())
        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpJointAngles(SessionProcessedBase):
    header = 'Raw JA'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_pre_ik_files_exist():
            return -1
        if not trial.does_post_ik_file_exists():
            return 0

        latest_pre_file = max(trial.pre_ik_times())
        oldest_pos_file = trial.post_ik_file_time()

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpFilteredSensors(SessionProcessedBase):
    header = 'Filtered PS'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_transformed_ps_files_exist():
            print(trial.transformed_ps_filenames.values())
            sys.exit()
            return -1
        if not trial.do_pre_ps_files_exist():
            return 0

        latest_pre_file = max(trial.transformed_ps_files_times())
        oldest_pos_file = min(trial.pre_ps_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpAlignedData(SessionProcessedBase):
    header = 'Aligned Data'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_all_pre_files_exist():
            return -1
        if not trial.do_all_post_files_exist():
            return 0

        latest_pre_file = max(trial.all_pre_files_times())
        oldest_pos_file = min(trial.all_post_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpMatchedContacts(SessionProcessedBase):
    header = 'Matched Conts'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_all_post_files_exist():
            return -1
        if not trial.do_matched_contacts_files_exist():
            return 0

        latest_pre_file = max(trial.all_post_files_times())
        oldest_pos_file = min(trial.matched_contacts_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpExportedForces(SessionProcessedBase):
    header = 'Exp Forces'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        if not trial.do_matched_contacts_files_exist():
            return -1
        if not trial.does_digit_force_file_exist() or not trial.does_segment_force_file_exist():
            return 0

        latest_pre_file = max(trial.matched_contacts_files_times())
        oldest_pos_file = min(trial.digit_force_file_time(), trial.segment_force_file_time())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class SpTorques(SessionProcessedBase):
    header = 'Torques'
    _type = 'per_trial'

    def __init__(self):
        super().__init__()

    def _eval_trial(self, trial):
        # not implemented
        return -1


# ----------------- per session classes
class SpMetaSession(SessionProcessedBase):
    header = 'M'
    _type = 'per_session_one_symbol'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        if sw.has_meta:
            self._skip_the_rest = False
            return YES
        else:
            self._skip_the_rest = True
            return NO


class SpNumTrialsSession(SessionProcessedBase):
    header = 'NUM'
    _type = 'per_session_num_trials'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        numtrials = len(sw.msession)
        return f'{numtrials:>{self.width()}}'


class SpNumSuccessfulTrialsSession(SessionProcessedBase):
    header = 'SUC'
    _type = 'per_session_num_trials'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        numsucces = sum([trial.success for trial in sw.msession])
        return f'{numsucces:>{self.width()}}'


class SpOpensimModel(SessionProcessedBase):
    header = 'O'
    _type = 'per_session_one_symbol'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        if os.path.exists(sw.mstruct['opensim_model_locked_base']):
            return YES
        else:
            return NO


class SpMujocoModel(SessionProcessedBase):
    header = 'M'
    _type = 'per_session_one_symbol'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        if os.path.exists(sw.mstruct['mujoco_model_sensorized']):
            return YES
        else:
            return NO


class SpGoodSession(SessionProcessedBase):
    header = 'G'
    _type = 'per_session_one_symbol'

    def __init__(self):
        super().__init__()

    def _eval_per_session(self, sw, preset):
        if 'good_sessions' in preset.keys() and sw.sess_name in preset['good_sessions']:
            return YES
        else:
            return NO


DEFAULT_EVALUATORS = (
    SpMetaSession(), SpGoodSession(), SpNumTrialsSession(), SpNumSuccessfulTrialsSession(),
    SpMarkers2D(), SpMarkers3D(), SpOpensimModel(), SpJointAngles(),
    SpFilteredSensors(), SpAlignedData(), SpMujocoModel(),
    SpMatchedContacts(), SpExportedForces(),
    SpTorques())

# TODO:
# Add Jarvis variant
# Add check for OpenSim models
# Add check for MuJoCo models







####################################### Processing processing
def _color_resp_trials(resp, numtrials):
    if resp >= numtrials:
        return Fore.GREEN
    if resp == 0:
        return Fore.RED
    return Fore.YELLOW


def report_sessions_processing_status(session_wrappers, preset, evaluators=None, verbose=0):
    """
    Keyword Arguments:
        evaluators (list of callable): each element should implement SessionProcessedBase class
            methods. If None, uses DEFAULT_EVALUATORS.
        verbose (int): if >0, will report additional multi-line error messages.

    Total report structure
    session_name|NUM|TRIAL_EVALUATOR1|TRIAL_EVALUATOR2...
        M: mstruct, mdof, mobject, msession presence
        NUM: total number of trials from meta_session

    Trial evaluator report structure:
        |TRIAL_EVALUATOR|
        |NEG ZER POS COR|
        NEG: parent file(s) do not exist
        ZER: parent file(s) exist, but children do not
        POS: children file(s) exist
        COR: children file(s) exist and are newer than parents
    """
    # TODO switch list based on preset
    if evaluators is None:
        evaluators = DEFAULT_EVALUATORS

    session_field_width = max([len(sw.sess_name) for sw in session_wrappers])
    # first line
    print(' '*session_field_width, end='')
    for e in evaluators:
        print(f'|{e.header_1()}', end='')

    # second line
    print()
    print(' '*session_field_width, end='')
    for e in evaluators:
        print(f'|{e.header_2()}', end='')

    for sw in session_wrappers:
        print()
        # session name
        print(f'{sw.sess_name:{session_field_width}}', end='')

        for e in evaluators:
            print(f'|{e.eval(sw, preset)}', end='')
            if e.skip_the_rest():
                break
    print()
