#!python3.7
import os
import sys
import inspect
import time
import datetime
import argparse
import warnings

import O2MConverter

# include local library functions - TB included in NCams
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from common import preset
from common import meta_session
from common import tools


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


def main(server, session, overwrite):
    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(
            server))

    if len(session) == 0:
        session = meta_session.find_session_dirs(server)[0]

    print('Processing session {}.'.format(session))
    server_session = os.path.join(server, session)

    if not os.path.exists(server_session):
        ValueError('Session {} does not exist on the server.'.format(session))

    # load session meta structure
    mstruct = meta_session.import_meta_structure(server_session)

    if not overwrite and os.path.exists(mstruct['mujoco_model']):
        warnings.warn('MuJoCo model exists, not converting. Specify overwrite key to overwrite.')
        return

    convert_osim_model(
        mstruct['opensim_model'],
        os.path.join(os.path.split(mstruct['opensim_model'])[0], 'Geometry'),
        os.path.split(mstruct['mujoco_model'])[0])


if __name__ == '__main__':
    current_preset_name, current_preset, argv = preset.process_args_for_preset()

    parser = argparse.ArgumentParser(
        description=('Generates a MuJoCo model from an OpenSim model.'))
    tools.add_default_arguments(
        parser, ('session', 'overwrite'))
    tools.add_default_kwarguments(
        parser, {'server': current_preset['default_server']})

    args = parser.parse_args(args=argv)

    start_time = time.time()
    main(args.server, args.session, args.overwrite)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
