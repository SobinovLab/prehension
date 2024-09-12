#!python3.8
# -*- coding: utf-8 -*-
"""
Execute an OpenSim inverse kinematic routine. Used by mark_base.

TODO Check if OpenSim can run in Python3.11, or use a compiled exe file.

Copyright (C) 2019-2024 Anton Sobinov
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
import opensim


def run_ik_f(ik_file, log_file):
    # if log file exists, remove it
    if os.path.exists(log_file):
        os.remove(log_file)

    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(log_file)
    opensim.Logger.setLevelString('warn')
    task = opensim.tools.InverseKinematicsTool(ik_file)

    task.run()


if __name__ == '__main__':
    run_ik_f(sys.argv[1], sys.argv[2])
