#!python3.8
import os
import sys
import opensim

def run_ik_f(ik_file, log_file):
    # if log file exists, remove it
    if os.path.exists(log_file):
        os.remove(log_file)

    opensim.Logger.removeFileSink()
    opensim.Logger.addFileSink(log_file)
    opensim.Logger.setLevelString('warn')
    task = opensim.tools.InverseKinematicsTool(ik_file)

    task.run()


if __name__ == '__main__':
    run_ik_f(sys.argv[1], sys.argv[2])
