#!python3
# -*- coding: utf-8 -*-
"""
Transforms an OpenSim model into a MuJoCo one. Once per monkey.

Copyright (C) 2019-2025 Anton Sobinov
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
import warnings

import O2MConverter

from .. import meta_session


def convert_osim_model(osim_model, geometry_folder, output_folder):
    converter4 = O2MConverter.Converter4()

    # custom scene parameters
    converter4.mujoco_dic['worldbody']['geom']['@pos'] = '0 0 -0.5'  # floor
    # nicer light
    converter4.mujoco_dic["worldbody"]['body']["light"] = {
        "@cutoff": "50", "@diffuse": ".7 .7 .7", "@dir": "-1.5 -1 -1", "@directional": "false",
        "@exponent": "30", "@pos": "1.5 1 1", "@specular": ".9 .9 .9"
    }
    # quality and diffuse light
    converter4.mujoco_dic["visual"] = {
        "map": {"@fogstart": "6", "@fogend": "10"},
        "quality": {"@shadowsize": "2048"},
        'headlight': {'@diffuse': ".6 .6 .6", '@specular': "0 0 0"}
    }
    # skybox
    converter4.mujoco_dic["asset"]["texture"].append(
        {'@builtin': "gradient", '@height': "100",
         '@rgb1': "0.93 0.93 0.93", '@rgb2': "0.259 0.259 0.259",
         '@type': "skybox", '@width': "100"}
    )
    # ground texture
    converter4.mujoco_dic["asset"]["texture"][0] = {
        "@name": "texplane", "@type": "2d", "@builtin": "checker", "@rgb1": "0.631 0.533 0.498",
        "@rgb2": "0.471 0.565 0.612", "@width": "100", "@height": "100"
    }
    # ground material
    converter4.mujoco_dic["asset"]["material"][0] = {
        "@name": "MatPlane", "@reflectance": "0.1", "@texture": "texplane",
        "@texrepeat": "10 10", "@texuniform": "true"
    }
    # geom texture
    converter4.mujoco_dic["asset"]["texture"][1] = {
        "@name": "texgeom", "@type": "cube", "@builtin": "flat",
        "@width": "100", "@height": "100", "@rgb1": "0.816 0.706 0.663", "@rgb2": "0.816 0.706 0.663",
        "@mark": "cross", "@markrgb": "1 1 1",
    }
    # default coloring
    converter4.mujoco_dic["default"]["geom"] = {
        "@contype": "1", "@conaffinity": "1", "@condim": "3",
        "@margin": "0.001",
        "@solref": ".02 1", "@solimp": ".8 .8 .01",
        "@material": "geom"}

    converter4.convert(osim_model, output_folder, geometry_folder, False)


def transform_osim_model(rserv, pserv, session, overwrite):
    """Generates a MuJoCo model from an OpenSim model.

    Arguments:
        server {str} --- Folder where the sessions are located.
        session {str} --- Session directory to use.
        temp {str} --- Folder for local temporary storage.
    """
    if not os.path.exists(rserv):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            rserv))

    if len(session) == 0:
        session = meta_session.find_session_dirs(rserv)[0]

    print('Processing session {}.'.format(session))
    raw_ss = os.path.join(rserv, session)
    proc_ss = os.path.join(pserv, session)

    if not os.path.exists(raw_ss):
        ValueError('Session {} does not exist on the server.'.format(session))

    # load session meta structure
    mstruct, _, _, msession = meta_session.load_meta_information(raw_ss, proc_ss)

    if not overwrite and os.path.exists(mstruct['mujoco_model']):
        warnings.warn('MuJoCo model exists, not converting. Specify overwrite key to overwrite.')
        return

    convert_osim_model(
        mstruct['opensim_model'],
        os.path.join(os.path.split(mstruct['opensim_model'])[0], 'Geometry'),
        os.path.split(mstruct['mujoco_model'])[0])
