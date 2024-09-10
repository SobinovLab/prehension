#!python3

import csv
import pylab
import scipy.signal


def import_dat(fname):
    with open(fname, 'r') as f:
        rdr = csv.reader(f)

        l = next(rdr)
        sensor_names = [i.strip() for i in l[1:]]

        times = []
        sensor_vals = [[] for _ in sensor_names]

        for li in rdr:
            times.append(float(li[0]))
            for isens, sensorval in enumerate(li[1:]):
                sensor_vals[isens].append(float(sensorval))

        # first record is fucky - before MJC initialized?
        for isens in range(len(sensor_names)):
            sensor_vals[isens][0] = 0.

    return sensor_names, times, sensor_vals

def main():
    filename = './touchsensors.csv'
    sensor_names, times, sensor_vals = import_dat(filename)

    finger_groups = [
        ['Stouch_RA5C1', 'Stouch_RA6D1'],
        ['Stouch_RA4P2', 'Stouch_RA5C2', 'Stouch_RA6D2'],
        ['Stouch_RA4P3', 'Stouch_RA5C3', 'Stouch_RA6D3'],
        ['Stouch_RA4P4', 'Stouch_RA5C4', 'Stouch_RA6D4'],
        ['Stouch_RA4P5', 'Stouch_RA5C5', 'Stouch_RA6D5']]
    finger_group_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    finger_groups_idxs = []
    for finger_group in finger_groups:
        finger_groups_idxs.append([])
        for sensor in finger_group:
            finger_groups_idxs[-1].append(sensor_names.index(sensor))

    pylab.figure()
    for igroup, groupname in enumerate(finger_group_names):
        sensvals = sensor_vals[finger_groups_idxs[igroup][0]]
        for isens in finger_groups_idxs[igroup][1:]:
            sensvals = [v + vs for v, vs in zip(sensvals, sensor_vals[isens])]

        sensvals = scipy.ndimage.median_filter(sensvals, size=5, mode='mirror')
        sensvals = scipy.signal.wiener(sensvals, mysize=9)

        pylab.plot(times, sensvals, label=groupname)

    # for isens, sensor_val in enumerate(sensor_vals):
    #     pylab.plot(times, sensor_val, label=sensor_names[isens])

    pylab.xlim([min(times), max(times)])
    pylab.ylim([0, 12])
    pylab.xlabel('Time, s')
    pylab.ylabel('Force, N')
    pylab.legend()



if __name__ == '__main__':
    main()

    pylab.show()
