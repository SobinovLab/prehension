#!python3.12
# Non-pytest test script
# Simply run and verify output

import os
import inspect
import sys

# Hack to add prehension to path
# For production, package will be installed using setup.py
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
print(f'Adding {parentdir} to path')
sys.path.insert(0, parentdir)

from prehension import io_tools


def main():
    pass


if __name__ == "__main__":
    main()