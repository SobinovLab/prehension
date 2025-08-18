#!python3
# -*- coding: utf-8 -*-
"""
Creates IK and Scaling files for OpenSim based on a period of trial.

Copyright (C) 2023-2024 Anton Sobinov
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
import xml.etree.ElementTree as ET

from ..tools import io


def get_body_masses(osim_filename, verbose=False):
    tree = ET.parse(osim_filename)
    root = tree.getroot()

    body_masses = {}
    for body_e in root.findall('.//Body'):
        body_name = body_e.attrib['name']

        mass = float(body_e.find('mass').text.strip())

        body_masses[body_name] = mass

        if verbose:
            print('Body {}: {} kg'.format(body_name, mass))

    return body_masses


def export_body_masses(osim_filename, o_filename, verbose=False):
    body_masses = get_body_masses(osim_filename, verbose=verbose)

    io.dic_to_csv(o_filename, body_masses, column_names=['body', 'mass kg'])
