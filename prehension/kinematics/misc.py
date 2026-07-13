#!python3
# -*- coding: utf-8 -*-
"""
Creates IK and Scaling files for OpenSim based on a period of trial.

Copyright (C) 2023-2024 Anton Sobinov
https://github.com/SobinovLab/prehension

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
from warnings import warn

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


def scale_segment_masses(parent_osim_filename, child_osim_filename, o_osim_filename):
    p_tree = ET.parse(parent_osim_filename)
    p_root = p_tree.getroot()

    c_tree = ET.parse(child_osim_filename)
    c_root = c_tree.getroot()

    for p_body_e in p_root.findall('.//Body'):
        body_name = p_body_e.attrib['name']

        # find mass
        mass = float(p_body_e.find('mass').text.strip())
        if mass <= 0:
            continue
        # find mass center and scale
        p_mass_center_e = p_body_e.find('mass_center')
        p_mass_center = [float(v) for v in p_mass_center_e.text.strip().split()]
        p_geom_e = p_body_e.find('attached_geometry')
        p_mesh_e = p_geom_e.find('Mesh')
        if p_mesh_e is None:
            warn(f'Could not find mesh for body {body_name}.')
            continue
        p_scale_factors_e = p_mesh_e.find('scale_factors')
        p_scale = [float(v) for v in p_scale_factors_e.text.strip().split()]

        # find the child body
        c_body_e = c_root.find(f'.//Body[@name="{body_name}"]')
        if c_body_e is None:
            warn(f'Could not find body {body_name}.')
            continue

        # find mass, scale and mass center
        c_mass_e = c_body_e.find('mass')
        c_geom_e = c_body_e.find('attached_geometry')
        c_mesh_e = c_geom_e.find('Mesh')
        if c_mesh_e is None:
            warn(f'Could not find mesh for body {body_name}.')
            continue
        c_scale_factors_e = c_mesh_e.find('scale_factors')
        c_scale = [float(v) for v in c_scale_factors_e.text.strip().split()]
        c_mass_center_e = c_body_e.find('mass_center')

        # scale things to new proportions
        x_prop = c_scale[0] / p_scale[0]
        y_prop = c_scale[1] / p_scale[1]
        z_prop = c_scale[2] / p_scale[2]
        new_mass = mass * x_prop * y_prop * z_prop
        new_mass_center = p_mass_center
        new_mass_center[0] *= x_prop
        new_mass_center[1] *= y_prop
        new_mass_center[2] *= z_prop

        c_mass_e.text = str(new_mass)
        c_mass_center_e.text = ' '.join([str(v) for v in new_mass_center])

    # export the model
    c_tree.write(o_osim_filename, encoding='UTF-8', xml_declaration=True)
