#!python3
# -*- coding: utf-8 -*-
"""
Train Jarvis. UNTESTED

Copyright (C) 2024 Caleb Raman, Rashi Bhatt
https://github.com/BensmaiaLab/prehension

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
# Fxns for training jarvis

# 1. Run any of the three models simultaneously
# 2. Run one model on all three body parts
# models_body_2run = [(bodypart, model) .....]
# 3. User inputs a model name --> yaml --> modify yaml (either user or in code based on inputs???)
# 4. Run in parallel
# 5. User needs to see: loss, acy, epoch

import argparse
import copy
import datetime
import sys
import time

from prehension.kinematics.train_jarvis import train_jarvis


# Custom type to parse list of tuples
def tuple_list(s):
    try:
        tuples = []
        for pair in s.split(','):
            values = pair.split(':')
            if len(values) != 2:
                raise argparse.ArgumentTypeError(
                    "Tuples must be in the format (x:y)")
            tuples.append((str(values[0]), str(values[1])))
        return tuples
    except ValueError:
        raise argparse.ArgumentTypeError("Unable to parse tuples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Trains Jarvis (pronounced \'Yar-vuh-s\' in BULK."))

    parser.add_argument('--combos', type=tuple_list,
                        help='List of project:model tuples to train')
    parser.add_argument('--epochs', default=50, type=int, required=False,
                        help="Num epochs for training")
    parser.add_argument('--verbose', action='store_true')
    # parser.add_argument(
    #     '--bodyparts',
    #     type=list, default=[],
    #     help='Bodyparts to train')

    argv = copy.deepcopy(sys.argv)[1:]
    args = parser.parse_args(args=argv)

    start_time = time.time()
    train_jarvis(args.combos, args.epochs, args.verbose)

    print('Program took {}.'.format(
        datetime.timedelta(seconds=time.time() - start_time)))
