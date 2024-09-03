#!python3
# -*- coding: utf-8 -*-
import csv
import xml.etree.ElementTree as ET


def import_mot(fname):
    '''Import OpenSim motion file into a python structure.

    Arguments:
        fname {str} -- motion file.

    Returns a list:
        dof_names {list of str} -- names of DOFs.
        times {list of numbers} -- time series.
        dofs {list} -- each item corresponds to values for that DOF for each frame.
            dofs[iDOF][iTime]
    '''
    with open(fname, 'r') as f:
        rdr = csv.reader(f, dialect='excel-tab')

        l = next(rdr)
        while len(l) < 1 or not l[0].strip().lower() == 'time':
            l = next(rdr)

        dof_names = [i.strip() for i in l[1:]]

        times = []
        dofs = [[] for _ in dof_names]

        for li in rdr:
            times.append(float(li[0]))
            for idof, vdof in enumerate(li[1:]):
                dofs[idof].append(float(vdof))
    return dof_names, times, dofs


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
