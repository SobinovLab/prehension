from ..io_tools import import_csv


def load_roms(filename, dof_names=None):
    column_names, values = import_csv(filename)

    i_dofname = column_names.index('dof_name')
    i_rmin = column_names.index('range_min')
    i_rmax = column_names.index('range_max')
    if 'rotation' in column_names:
        i_rot = column_names.index('rotation')
    else:
        i_rot = -1

    if dof_names is None:
        ranges = [[rmin, rmax] for rmin, rmax in zip(values[i_rmin], values[i_rmax])]
        if i_rot >= 0:
            return values[i_dofname], ranges, values[i_rot]
        else:
            return values[i_dofname], ranges

    ranges = []
    rots = []
    for dof_name in dof_names:
        i_dof = values[i_dofname].index(dof_name)
        ranges.append([values[i_rmin][i_dof], values[i_rmax][i_dof]])
        if i_rot >= 0:
            rots.append(values[i_rot][i_dof])
    if i_rot >= 0:
        return ranges, rots
    else:
        return ranges
