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

from colorama import Fore, Style


####################################### Callable classes to evaluate stages of processing
class TrialProcessedBase():
    """docstring for TrialProcBase"""
    header = 'UNDEFINED'  # 12 symbols or less. Shortest column 3 symbols

    def __call__(self, trial):
        """Return
            -1 if previous does not exist,
            0 if exists, but child does not,
            1 if child exists,
            2 if child was made after the parents  TODO(AS)
        """
        return -1

    def width(self):
        return 3  # max(len(self.header), 3)


class TpMarkers2D(TrialProcessedBase):
    header = 'Markers 2D'

    def __call__(self, trial):
        if not trial.do_videos_files_exist():
            return -1
        if not trial.do_2d_files_exist():
            return 0
        latest_pre_file = max(trial.video_file_times())
        oldest_pos_file = min(trial.m_2d_file_times())
        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpMarkers3D(TrialProcessedBase):
    header = 'Markers 3D'

    def __call__(self, trial):
        if not trial.do_2d_files_exist():
            return -1
        if not trial.do_3d_files_exist():
            return 0
        latest_pre_file = max(trial.m_2d_file_times())
        oldest_pos_file = min(trial.m_3d_files_times())
        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpJointAngles(TrialProcessedBase):
    header = 'Raw JA'

    def __call__(self, trial):
        if not trial.do_pre_ik_files_exist():
            return -1
        if not trial.does_post_ik_file_exists():
            return 0

        latest_pre_file = max(trial.pre_ik_times())
        oldest_pos_file = trial.post_ik_file_time()

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpFilteredSensors(TrialProcessedBase):
    header = 'Filtered PS'

    def __call__(self, trial):
        if not trial.do_transformed_ps_files_exist():
            return -1
        if not trial.do_pre_ps_files_exist():
            return 0

        latest_pre_file = max(trial.transformed_ps_files_times())
        oldest_pos_file = min(trial.pre_ps_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpAlignedData(TrialProcessedBase):
    header = 'Aligned Data'

    def __call__(self, trial):
        if not trial.do_all_pre_files_exist():
            return -1
        if not trial.do_all_post_files_exist():
            return 0

        latest_pre_file = max(trial.all_pre_files_times())
        oldest_pos_file = min(trial.all_post_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpMatchedContacts(TrialProcessedBase):
    header = 'Matched Conts'

    def __call__(self, trial):
        if not trial.do_all_post_files_exist():
            return -1
        if not trial.do_matched_contacts_files_exist():
            return 0

        latest_pre_file = max(trial.all_post_files_times())
        oldest_pos_file = min(trial.matched_contacts_files_times())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpExportedForces(TrialProcessedBase):
    header = 'Exp Forces'

    def __call__(self, trial):
        if not trial.do_matched_contacts_files_exist():
            return -1
        if not trial.does_digit_force_file_exist() or not trial.does_segment_force_file_exist():
            return 0

        latest_pre_file = max(trial.matched_contacts_files_times())
        oldest_pos_file = min(trial.digit_force_file_time(), trial.segment_force_file_time())

        if latest_pre_file > oldest_pos_file:
            return 1
        return 2


class TpTorques(TrialProcessedBase):
    header = 'Torques'

    def __call__(self, trial):
        # not implemented
        return -1


DEFAULT_TRIAL_EVALUATORS = (
    TpMarkers2D(), TpMarkers3D(), TpJointAngles(),
    TpFilteredSensors(), TpAlignedData(),
    TpMatchedContacts(), TpExportedForces(),
    TpTorques())


####################################### Processing processing
def _color_resp_trials(resp, numtrials):
    if resp >= numtrials:
        return Fore.GREEN
    if resp == 0:
        return Fore.RED
    return Fore.YELLOW


def report_sessions_processing_status(session_wrappers, trial_evaluators=None, verbose=0):
    """
    Keyword Arguments:
        trial_evaluators (list of callable): each element should implement TrialProcessedBase class
            methods. If None, uses DEFAULT_TRIAL_EVALUATORS.
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
    if trial_evaluators is None:
        trial_evaluators = DEFAULT_TRIAL_EVALUATORS

    session_field_width = max([len(sw.sess_name) for sw in session_wrappers])
    # first several global reports
    print(' '*session_field_width + '|M|NUM|SUC', end='')

    # then trial evaluators
    for te in trial_evaluators:
        width = te.width()
        width_tot = (width + 1) * 4 - 1
        print('|' + f'{te.header:^{width_tot}}', end='')

    print()
    print(' '*session_field_width + '| |   |   ', end='')
    for te in trial_evaluators:
        width = te.width()
        print('|' +
              f'{"NEG":>{width}}' +
              f'{"ZER":>{width+1}}' +
              f'{"POS":>{width+1}}' +
              f'{"COR":>{width+1}}',
              end='')

    for sw in session_wrappers:
        print()
        # session-wide reports
        print(f'{sw.sess_name:{session_field_width}}', end='')
        if sw.has_meta:
            print(f'|{Fore.GREEN}✓{Style.RESET_ALL}', end='')
        else:
            print(f'|{Fore.RED}x{Style.RESET_ALL}', end='')
            if verbose > 0:
                print(sw.load_meta_exception)
            continue

        numtrials = len(sw.msession)
        numsucces = sum([trial.success for trial in sw.msession])
        print(f'|{numtrials:>3}|{numsucces:>3}', end='')

        # per-trial evals
        for te in trial_evaluators:
            width = te.width()
            reports = [te(t) for t in sw.msession]
            resp_neg = sum([v == -1 for v in reports])
            resp_zer = sum([v == 0 for v in reports])
            resp_pos = sum([v > 0 for v in reports])
            resp_cor = sum([v == 2 for v in reports])
            color_pos = _color_resp_trials(resp_pos, numsucces)
            color_cor = _color_resp_trials(resp_cor, numsucces)
            print('|' +
                  f'{resp_neg:>{width}}' +
                  f'{resp_zer:>{width+1}}' +
                  f'{color_pos}{resp_pos:>{width+1}}{Style.RESET_ALL}' +
                  f'{color_cor}{resp_cor:>{width+1}}{Style.RESET_ALL}',
                  end='')
    print()
