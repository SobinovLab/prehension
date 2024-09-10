#!python3.7
import csv
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage

import ncams


def main():
    filenames = {
        'touchsensors_compliance_0.002.csv': '2',
        'touchsensors_compliance_0.00225.csv': '2.25',
        'touchsensors_compliance_0.0025.csv': '2.5',
        'touchsensors_compliance_0.00275.csv': '2.75',
        'touchsensors_compliance_0.003.csv': '3',
        'touchsensors_compliance_0.00325.csv': '3.25',
        'touchsensors_compliance_0.0035.csv': '3.5',
        'touchsensors_compliance_0.00375.csv': '3.75',
        'touchsensors_compliance_0.004.csv': '4',
        'touchsensors_compliance_0.005.csv': '5',
        'touchsensors_compliance_0.007.csv': '7',
        'touchsensors_compliance_0.009.csv': '9',
    }
    only_sensor = 'Stouch_RA6D2'

    colors = {}
    cmap = plt.get_cmap('plasma')

    times_acc = {}
    data_acc = {}

    for ifile, filename in enumerate(filenames.keys()):
        column_names, data = ncams.utils.import_csv(filename)
        times_acc[filename] = data[0]
        if np.any(np.diff(times_acc[filename]) < 0):
            trm = np.where(np.diff(times_acc[filename]) < 0)[0][0]
            times_acc[filename] = times_acc[filename][:trm]
        else:
            trm = None
        colors[filename] = cmap(ifile / len(filenames))

        for cn, dat in zip(column_names[1:], data[1:]):
            if cn not in data_acc.keys():
                data_acc[cn] = {}
            if trm is None:
                data_acc[cn][filename] = dat
            else:
                data_acc[cn][filename] = dat[:trm]

    if only_sensor is not None:
        data_acc = {only_sensor: data_acc[only_sensor]}

    n_subplots = len(data_acc.keys())
    xn_subplots = int(np.ceil(np.sqrt(n_subplots)))
    yn_subplots = int(np.ceil(n_subplots / xn_subplots))

    plt.figure(figsize=(16, 9))
    plt.suptitle('Raw values')
    for isensor, (sensor, dics) in enumerate(data_acc.items()):
        plt.subplot(xn_subplots, yn_subplots, isensor + 1)
        for filename, val in dics.items():
            v = np.array(val)
            v[v > 100] = 0
            v = scipy.ndimage.gaussian_filter1d(v, 3, mode='reflect')
            plt.plot(times_acc[filename], v, label=filenames[filename], c=colors[filename])
        plt.xlabel('Time, s')
        plt.ylabel(sensor + ', N')
        plt.legend()

    norm_period = [0.5, 0.75]
    zero_time_hreshold = 1e-3
    plt.figure(figsize=(16, 9))
    plt.suptitle('Normalized to {} and contact at 0'.format(norm_period))
    for isensor, (sensor, dics) in enumerate(data_acc.items()):
        plt.subplot(xn_subplots, yn_subplots, isensor + 1)
        for filename, val in dics.items():
            v = np.array(val)
            v[v > 100] = 0
            v = scipy.ndimage.gaussian_filter1d(v, 5, mode='reflect')
            t = np.array(times_acc[filename])
            # normalization
            norm_value = np.mean(v[(t > norm_period[0]) & (t < norm_period[1])])
            # offset
            strt = t[np.where(v > zero_time_hreshold)[0][0]]
            t = t - strt
            plt.plot(t, v / norm_value, label=filenames[filename], c=colors[filename])
        plt.xlabel('Time, s')
        plt.ylabel(sensor + ', au')
        plt.legend()


if __name__ == '__main__':
    main()

    plt.show()
