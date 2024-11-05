#!python3.7
import itertools
import os
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import tqdm

from .. import io_tools
from .. import meta_session
from .. import tools
from ..tools import rs, ws


def make_trial_mask(filename):
    # load pressure sensors
    ps_times, ps_matrices = io_tools.import_matrices(filename)
    ps_matrices = np.array(ps_matrices)

    ps_mask = np.sum(np.abs(ps_matrices), axis=0, dtype=bool)

    return ps_mask


def make_session_mask(mstruct, msession, trials_sel, mask_filenames, overwrite):
    # accumulate data
    trials = []
    for trial in msession:
        if len(trials_sel) != 0 and trial.trial_number not in trials_sel:
            continue
        if not trial.do_post_ps_files_exist():
            continue
        trials.append(trial)

    rs('Found {} trials: {}'.format(len(trials), ', '.join([str(t.trial_number) for t in trials])))

    # go through each trial and accumulate inclusive matrices
    ps_masks = dict.fromkeys(mstruct['ps_dic'])
    for trial in tqdm.tqdm(trials, ncols=100, desc='Trials'):
        for ps_name in mstruct['ps_dic']:
            try:
                ps_mask = make_trial_mask(trial.get_post_ps_filenames()[ps_name])
            except Exception as e:
                ws(
                    'File {} corrupted for import: {}'.format(
                        trial.get_post_ps_filenames()[ps_name], e
                    )
                )
                raise e
            if ps_masks[ps_name] is None:
                ps_masks[ps_name] = ps_mask
            else:
                ps_masks[ps_name] = np.logical_or(ps_masks[ps_name], ps_mask)

    print()
    for ps_name, ps_mask in ps_masks.items():
        rs('Pressure sensor {} has {} active sensels.'.format(ps_name, np.sum(ps_mask)))
        o_filename = mask_filenames[ps_name]
        if overwrite or not os.path.exists(o_filename):
            io_tools.export_one_csv_matrix(o_filename, ps_mask.astype(int))
        else:
            ws('Pressure sensor {} mask file {} already exists.'.format(ps_name, o_filename))


def make_segment(name, pos, size, rgba, ntab, joints=None, geomname=None):
    segment = ET.Element('body')
    segment.set('name', name)
    segment.set('pos', ' '.join(str(i) for i in pos))
    segment.text = '\n' + ' ' * (2 * (ntab + 2))
    segment.tail = '\n' + ' ' * (2 * ntab)

    geom = ET.Element('geom')
    segment.append(geom)
    geom.tail = '\n' + ' ' * (2 * (ntab + 2))
    geom.set('type', 'box')
    geom.set('size', ' '.join(str(i) for i in size))
    geom.set('rgba', ' '.join(str(i) for i in rgba))
    if geomname is not None:
        geom.set('name', geomname)

    if joints is None:
        joints = []
    if len(joints) != 0 and not isinstance(joints[0], (list, tuple)):
        joints = [joints]

    for joint in joints:
        joint_name, joint_type, joint_axis, joint_range = joint
        dof = ET.Element('joint')
        segment.append(dof)
        dof.set('name', joint_name)
        dof.set('axis', ' '.join(str(i) for i in joint_axis))
        dof.set('pos', '0 0 {}'.format(size[2]))
        dof.set('range', ' '.join(str(i) for i in joint_range))
        dof.set('type', joint_type)
        dof.tail = '\n' + ' ' * (2 * (ntab + 2))

    segment.findall('*')[-1].tail = '\n' + ' ' * (2 * (ntab + 1))

    return segment


def make_contact_pair(geom1, geom2, tail, margin=0, gap=0):
    contact = ET.Element('pair')
    contact.set('geom1', geom1)
    contact.set('geom2', geom2)
    if margin is not None:
        contact.set('margin', str(margin))
    if gap is not None:
        contact.set('gap', str(gap))
    contact.tail = tail
    return contact


def add_pair_contacts(geom, geom_list, margin=0.03, gap=0.03, tail='\n' + ' ' * 2 * 2):
    '''Adds pairs of contacts between geom and the geom list that calculate distance when not
    contacting

    Both are names or lists of geoms

    margin and gap SHOULD be equal

    Arguments:
        geom {str} -- [description]
        geom_list {list of str} -- [description]
    Returns:
        {list of ET.Elements} -- list of elements that pair contact b/w geom and all relatives
    '''
    contacts = []
    for g in geom_list:
        contacts.append(make_contact_pair(geom, g, tail, margin=margin, gap=gap))

    return contacts


def tessellate_sensor(
    wb,
    name,
    hand_geomnames,
    sense_distance,
    n=(44, 44),
    sense_box_rgba=(0.2, 0.2, 0.2, 1),
    mask=None,
    leave_edges=False,
    inverse_x=False,
    inverse_y=False,
    switch_xy=False):
    att = wb.find(".//body[@name='{}']".format(name))
    if att is None:
        raise ValueError('Did not find the pressure sensor {}.'.format(name))

    # remove the attached geom
    att_geom = att.find("./geom")
    if att_geom is not None:
        att.remove(att_geom)

    # TODO(future) pull from asset/mesh/block.stl
    # although that is hard and feels like a waste of time
    sz = (0.0838, 0.0838)
    width = 0.006
    padding = 1e-4

    # mjc expects half-size
    sz = (sz[0] / 2, sz[1] / 2)
    width = width / 2

    # add base geom if mask is provided
    if mask is not None:
        att_box_rgba = list(sense_box_rgba)
        att_box_rgba[3] = 0.4
        att_box_size = list(sz) + [width * 0.9]
        geom = ET.Element('geom')
        geom.tail = '\n' + ' ' * (2 * (2 + 2))
        geom.set('type', 'box')
        geom.set('size', ' '.join(str(i) for i in att_box_size))
        geom.set('rgba', ' '.join(str(i) for i in att_box_rgba))
        geom.set('name', name + '_base_geom')
        att.append(geom)

    sense_box_name_fmt = name + '_{ix:02d}_{iy:02d}'

    box_ext_size = (sz[0] / n[0], sz[1] / n[1])
    box_int_size = (sz[0] / n[0] - padding, sz[1] / n[1] - padding, width)

    contacts = []
    n_sensels = 0
    for ix, iy in itertools.product(range(n[0]), range(n[1])):
        if mask is None or mask[ix][iy] or (leave_edges and (ix in (0, n[0]) or iy in (0, n[1]))):
            sense_box_name = sense_box_name_fmt.format(ix=ix, iy=iy)
            sense_box_geomname = sense_box_name + '_geom'
            x = (ix + 0.5) * box_ext_size[0] * 2 - sz[0]
            y = (iy + 0.5) * box_ext_size[1] * 2 - sz[1]
            if switch_xy:
                x, y = y, x
            if inverse_x:
                x = -x
            if inverse_y:
                y = -y
            sense_box = make_segment(
                sense_box_name,
                [x, y, 0],
                box_int_size,
                sense_box_rgba,
                3,
                geomname=sense_box_geomname,
            )
            att.findall('*')[-1].tail = '\n' + ' ' * (2 * 4)
            att.append(sense_box)
            contacts += add_pair_contacts(
                sense_box_geomname, hand_geomnames, margin=sense_distance, gap=sense_distance
            )
            n_sensels += 1
    rs('Created {} sensels for {}.'.format(n_sensels, name))

    return contacts


def tessellate_sensors(
    mjc_model,
    out_model,
    sense_distance,
    nconmax=100000,
    njmax=10000,
    nstack=3000000,
    left_ps='LPS',
    right_ps='RPS',
    hand_parent_bname='RA3C',
    left_color=(0.984, 0.549, 0.0, 1),
    right_color=(0.129, 0.588, 0.952, 1),
    left_ps_mask_filename=None,
    right_ps_mask_filename=None,
    geoms_contact_floor=('Thorax_thorax', 'LPS_base_geom', 'RPS_base_geom')):
    tree = ET.parse(mjc_model)
    root = tree.getroot()
    if root.tag != 'mujoco':
        raise ValueError(
            'Wrong structure of the MuJoCo model file.' ' Tag mujoco is not present at top level.'
        )

    # collision detection only between defined pairs
    option = root.find('option')
    if option is None:
        option = ET.Element('option')
        root.append(option)
    option.set('collision', 'predefined')  # previously known as 'pair'

    # increase simulation buffers
    size = root.find('size')
    if size is None:
        size = ET.Element('size')
        root.append(size)
    size.set('nconmax', str(nconmax))
    size.set('njmax', str(njmax))
    size.set('nstack', str(nstack))

    wb = root.find('worldbody')
    if wb is None:
        raise ValueError('Could not find worldbody.')

    # find hand geoms
    hand_parent = wb.find(".//body[@name='{}']".format(hand_parent_bname))
    if hand_parent is None:
        raise ValueError('Could not find hand parent body {}.'.format(hand_parent_bname))
    hand_geoms = hand_parent.findall('.//geom')
    hand_geomnames = []
    for hand_geom in hand_geoms:
        hand_geomname = hand_geom.get('name')
        if hand_geomname is not None:
            hand_geomnames.append(hand_geomname)
    rs('Found {} hand geometries: {}.'.format(len(hand_geomnames), ', '.join(hand_geomnames)))

    # load masks of sensels
    if left_ps_mask_filename is None:
        left_ps_mask = None
    else:
        left_ps_mask = np.array(io_tools.import_one_csv_matrix(left_ps_mask_filename)).astype(bool)

    if right_ps_mask_filename is None:
        right_ps_mask = None
    else:
        right_ps_mask = np.array(io_tools.import_one_csv_matrix(right_ps_mask_filename)).astype(
            bool
        )

    contacts = tessellate_sensor(
        wb,
        left_ps,
        hand_geomnames,
        sense_distance,
        sense_box_rgba=left_color,
        mask=left_ps_mask,
        switch_xy=True,
        inverse_x=True,
    )
    contacts += tessellate_sensor(
        wb,
        right_ps,
        hand_geomnames,
        sense_distance,
        sense_box_rgba=right_color,
        mask=right_ps_mask,
        switch_xy=True,
        inverse_x=True,
    )

    # add contact bw geoms and the floor
    # find floor
    if wb.find("geom[@name='floor']") is not None:
        for geomname in geoms_contact_floor:
            if wb.find(".//geom[@name='{}']".format(geomname)) is not None:
                contacts += add_pair_contacts(geomname, ['floor'], gap=None, margin=None)
    else:
        warnings.warn('Could not find floor geom.')

    # add contacts
    if len(contacts) > 0:
        contact = root.find('contact')
        if contact is None:
            contact = ET.Element('contact')
            contact.tail = '\n'
            root.findall('*')[-1].tail = '\n' + ' ' * 2
            root.append(contact)
        # contact padding
        contact_c = contact.findall('*')
        if contact_c is None or len(contact_c) == 0:
            contact.text = '\n' + ' ' * (2 * 2)
        else:
            contact_c[-1].tail = '\n' + ' ' * (2 * 2)
        contact.extend(contacts)
        contact[-1].tail = '\n' + ' ' * 2

    tree.write(out_model, encoding='UTF-8', xml_declaration=True)


def prepare_mujoco_model(server, sessions, trials_sel, temp, overwrite, make_mask, tessellate,
                         sense_distance):
    """Generates a mask of pressure sensors matrix that highlights activated sensels
    and tessellates model sensors based on it.

    Arguments:
        server {str} --- Folder where the sessions are located.
        sessions {list of str} --- List of directories for processing. If empty, find all
            unprocessed directories.
        trials_sel {list of str} --- List of trials for processing. If empty, find all
            unprocessed trials.
        temp {str} --- Folder for local temporary storage.
        overwrite {bool} --- Overwrites the created files if they exist.
        make_mask {bool} --- Converts.
        tessellate {bool} --- Tessellates the pressure sensors into sensels.
        sense_distance {float} --- Distance between geom centers for "contact" calculation.
            Larger values slow down the execution, but low values are too short for relatively large
            bending bones like metacarpals and large muscle areas like thenar eminence. In meters.
    """
    tools.setup_logging(temp, sessions_dir=server)

    if not os.path.exists(server):
        raise ValueError('Server directory {} does not exist or is inaccessible.'.format(server))

    if len(sessions) == 0:
        sessions = meta_session.find_session_dirs(server)

    if len(trials_sel) > 0 and len(sessions) > 1:
        ws('A subset of trials was selected, only the first session will be used.')
        sessions = sessions[:1]

    # sort
    sessions.sort()
    rs('Found {} sessions: {}'.format(len(sessions), ', '.join(sessions)))

    for session in tqdm.tqdm(sessions, ncols=100, desc='Sessions'):
        print()
        rs('Processing session {}.'.format(session))
        server_session = os.path.join(server, session)

        if not os.path.exists(server_session):
            ws('Session {} does not exist on the server.'.format(session))
            continue

        # load session meta
        try:
            mstruct, _, _, msession = meta_session.load_meta_information(server_session)
        except Exception as e:
            ws('Could not load meta data from session {}, skipping.'.format(session))
            ws('Error message: {}'.format(e))
            continue

        # since it is solely used in this script, no need to have it in mstruct
        mask_filenames = {
            ps_name: os.path.join(server_session, 'ps_{}_mask.csv'.format(ps_name))
            for ps_name in mstruct['ps_dic'].keys()
        }

        if make_mask:
            make_session_mask(mstruct, msession, trials_sel, mask_filenames, overwrite)

        if tessellate:
            left_ps_mask_filename = mask_filenames['medial_sensor']
            right_ps_mask_filename = mask_filenames['lateral_sensor']
            tessellate_sensors(
                mstruct['mujoco_model'],
                mstruct['mujoco_model_sensorized'],
                sense_distance,
                left_ps_mask_filename=left_ps_mask_filename,
                right_ps_mask_filename=right_ps_mask_filename,
            )
