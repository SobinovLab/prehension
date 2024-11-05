#!python3
# -*- coding: utf-8 -*-
"""
Provides utilities for session visualization such as session status and exising files within a
given session

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


from colorama import Fore, Style
import os
import glob
import shutil


EXPECTED_PLOT_NAMES = [
    'AvgCondSuccessMatrix.png',
    'CondSuccessMatrix.png',
    'ForceTrace_from_success_grasp_start.png',
    'ForceTrace_from_ttl_to_reward.png',
    'Performance.png',
    'PerformanceLast10Days.png'
]


def display_session_info(experimental_server_wrappers, training_server_wrappers, clean, last_n):


    # Function to format and color the session status
    def format_status(session_path):
        if session_path and os.path.exists(session_path):
            return f"{Fore.GREEN}{os.path.basename(session_path)}{Style.RESET_ALL}"
        else:
            return f"{Fore.RED}{'MISSING'}{Style.RESET_ALL}"

    # Function to check if specific folders (behavior, sensors) exist in sw.raw_ss
    def check_folders(session_path):
        behavior_exists = os.path.exists(os.path.join(session_path, 'behavior'))
        sensors_exists = os.path.exists(os.path.join(session_path, 'sensors'))

        behavior_symbol = f" {Fore.GREEN}✔{Style.RESET_ALL} " if behavior_exists else f" {Fore.RED}✘{Style.RESET_ALL} "
        sensors_symbol = f" {Fore.GREEN}✔{Style.RESET_ALL} " if sensors_exists else f" {Fore.RED}✘{Style.RESET_ALL} "

        return f"{behavior_symbol}{sensors_symbol}"

    # Function to check if specific directories (filtered_sensors, prehension_plots, transformed_sensors) exist in sw.proc_ss
    def check_proc_folders(proc_path):
        filtered_sensors = os.path.exists(os.path.join(proc_path, "filtered_sensors"))
        prehension_plots_exist_bool = [
            os.path.exists(os.path.join(proc_path, "prehension_plots", fname)) for fname in EXPECTED_PLOT_NAMES
        ]
        transformed_sensors = os.path.exists(os.path.join(proc_path, "transformed_sensors"))

        filtered_symbol = (
            f" {Fore.GREEN}✔{Style.RESET_ALL} "
            if filtered_sensors
            else f" {Fore.RED}✘{Style.RESET_ALL} "
        )

        if all(prehension_plots_exist_bool):
            prehension_symbol = f" {Fore.GREEN}{sum(prehension_plots_exist_bool)}/{len(EXPECTED_PLOT_NAMES)}{Style.RESET_ALL} "
        elif any(prehension_plots_exist_bool):
            prehension_symbol = f" {Fore.YELLOW}{sum(prehension_plots_exist_bool)}/{len(EXPECTED_PLOT_NAMES)}{Style.RESET_ALL} "
        else:
            prehension_symbol = f" {Fore.RED}{sum(prehension_plots_exist_bool)}/{len(EXPECTED_PLOT_NAMES)}{Style.RESET_ALL} "

        transformed_symbol = (
            f" {Fore.GREEN}✔{Style.RESET_ALL} "
            if transformed_sensors
            else f" {Fore.RED}✘{Style.RESET_ALL} "
        )

        return f"{filtered_symbol}{prehension_symbol}{transformed_symbol}"

    # Function to check for timepoints.csv in sw.proc_ss
    def check_timepoints(proc_path):
        timepoints_exist = os.path.exists(os.path.join(proc_path, 'timepoints.csv'))
        return f" {Fore.GREEN}✔{Style.RESET_ALL} " if timepoints_exist else f" {Fore.RED}✘{Style.RESET_ALL} "

    # Function to check and format the has_meta property
    def format_meta(has_meta):
        return f" {Fore.GREEN}✔{Style.RESET_ALL} " if has_meta else f" {Fore.RED}✘{Style.RESET_ALL} "

    def can_delete(sw, delete=False):

        if not sw.log_full:
            return "no log found"
        logname_full = sw.log_full
        logname = os.path.basename(logname_full)
        matching_logs_other = set(glob.glob(os.path.join(
            os.path.dirname(os.path.dirname(sw.raw_ss)),
            '**', logname), recursive=True)) - set([logname_full, ])
        duplicates = len(matching_logs_other) > 0

        mark_for_delete = not os.path.exists(os.path.join(sw.raw_ss, 'sensors')) and duplicates

        msg = ""
        if delete:
            if mark_for_delete:
                shutil.rmtree(sw.raw_ss)
                shutil.rmtree(sw.proc_ss)
                msg='D!'
        return f" {Fore.RED}DELETE{msg}{Style.RESET_ALL} " if mark_for_delete else "KEEP"


    # Helper function to format the table rows
    def format_row(raw_status, folder_status_raw, proc_status, folder_status_proc, timepoints_status, meta_status, can_delete):
        return (f"{raw_status:<25} {folder_status_raw:<40} {proc_status:<25} {folder_status_proc:<25} "
                f"{timepoints_status:<7} {meta_status:<7} {can_delete:<7}")

    # Print a nicely formatted table for experimental sessions
    def print_header(header_text):
        print(
            f"{Fore.CYAN}~~~~~~~~~~~ {header_text} ~~~~~~~~~~~{Style.RESET_ALL}")
        print(f"{'raw session':<16} {'(behavior/sensors)':<21} {'processed session':<22} "
              f"{'(filtered/plots/transformed/timepoints/meta)':<20} {'cleanup status':<7}")
        print('-' * 135)

    print_header('EXPERIMENTAL SESSIONS')
    sorted_exp_sessions = sorted(experimental_server_wrappers, key=lambda x: (x.datetime, x.set_number))
    if last_n > 0:
        sorted_exp_sessions = sorted_exp_sessions[-last_n:]
    for sw in sorted_exp_sessions:
        raw_status = format_status(sw.raw_ss)
        proc_status = format_status(sw.proc_ss)
        folder_status_raw = check_folders(sw.raw_ss)
        folder_status_proc = check_proc_folders(sw.proc_ss)
        timepoints_status = check_timepoints(sw.proc_ss)
        meta_status = format_meta(sw.has_meta)
        if clean:
            can_delete_status = can_delete(sw, delete=True)
        else:
            can_delete_status = "----"
        print(format_row(raw_status, folder_status_raw, proc_status, folder_status_proc, timepoints_status, meta_status, can_delete_status))

    print()  # Line break between sections

    print_header('TRAINING SESSIONS')
    sorted_training_sessions = sorted(training_server_wrappers, key=lambda x: (x.datetime, x.set_number))
    if last_n > 0:
        sorted_training_sessions = sorted_training_sessions[-last_n:]
    for sw in sorted_training_sessions:
        raw_status = format_status(sw.raw_ss)
        proc_status = format_status(sw.proc_ss)
        folder_status_raw = check_folders(sw.raw_ss)
        folder_status_proc = check_proc_folders(sw.proc_ss)
        timepoints_status = check_timepoints(sw.proc_ss)
        meta_status = format_meta(sw.has_meta)
        if clean:
            can_delete_status = can_delete(sw, delete=True)
        else:
            can_delete_status = "----"
        print(format_row(raw_status, folder_status_raw, proc_status,
              folder_status_proc, timepoints_status, meta_status, can_delete_status))
