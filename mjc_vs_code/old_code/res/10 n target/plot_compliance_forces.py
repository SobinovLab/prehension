#!python3.7
import csv
import os

import numpy as np
import matplotlib.pyplot as plt

import ncams


def main():
    filenames = {
        'touchsensors_compliance_10.csv': '10',
        'touchsensors_compliance_9.csv': '9',
        'touchsensors_compliance_8.csv': '8',
        'touchsensors_compliance_7.csv': '7',
        'touchsensors_compliance_6.csv': '6',
        'touchsensors_compliance_5.csv': '5',
        'touchsensors_compliance_4.csv': '4',
    }
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

    n_subplots = len(data_acc.keys())
    xn_subplots = int(np.ceil(np.sqrt(n_subplots)))
    yn_subplots = int(np.ceil(n_subplots / xn_subplots))

    plt.figure(figsize=(16, 9))
    for isensor, (sensor, dics) in enumerate(data_acc.items()):
        plt.subplot(xn_subplots, yn_subplots, isensor + 1)
        for filename, val in dics.items():
            v = np.array(val)
            v[v > 100] = 0
            plt.plot(times_acc[filename], v, label=filenames[filename], c=colors[filename])
        plt.xlabel('Time, s')
        plt.ylabel(sensor + ', N')
        plt.legend()


if __name__ == '__main__':
    main()

    plt.show()
