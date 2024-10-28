# This is a helper script to do the following preprocessing steps
# 1. create meta_data
# 2  preprocess pressure sensors
# 3. filter pressure sensors

#!python3.11

import prehension
import os

PRESET = 'daiquiri_right_hemisphere_training_k1'

# Look for prehension scripots directory
basedir = os.path.dirname(prehension.__path__[0])
scriptsdir = os.path.join(basedir, 'prehension_scripts')

assert os.path.exists(scriptsdir), f'Scripts directory not found: {scriptsdir}'
os.chdir(scriptsdir)

cmds = [
    rf"py -3.11 create_meta.py {PRESET}",
    rf"py -3.11 pressure_sensors\preprocess_pressure_sensors.py {PRESET}",
    rf"py -3.11 pressure_sensors\filter_pressure_sensors.py {PRESET}",
]

for cmd in cmds:
    os.system(cmd)
