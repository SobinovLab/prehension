#!python3
# -*- coding: utf-8 -*-
"""
Untested.

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

# 1. Run any of the three models simultainiously
# 2. Run one model on all three body parts
# models_body_2run = [(bodypart, model) .....]
# 3. User inputs a model name --> yaml --> modify yaml (either user or in code based on inputs???)
# 4. Run in parallel
# 5. User needs to see: loss, acy, epoch

import concurrent.futures
import os

import jarvis.train_interface as train_interface
import torch
from jarvis.config.project_manager import ProjectManager


# CODE from:
# JARVIS-HybridNet/jarvis/ui/interactive_cli/train_cli.py
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic


def validate_pth_file(fp):
    assert os.path.isfile(fp)
    assert fp.split(".")[-1] == 'pth'


def train_hybridnet(proj_name, weights_pretrain='None',
                    keypoint_weights_pth=None, HN_weights_pth=None,
                    num_epochs=50, training_mode='all', bar_position=0, bar_desc=''):

    projectManager = ProjectManager()

    if not projectManager.load(proj_name):
        print(f"Could not load Project {proj_name}!")
        return

    if weights_pretrain == 'None':
        if keypoint_weights_pth == None:
            weights_keypoint_detect = 'latest'
        else:
            validate_pth_file(keypoint_weights_pth)
            weights_keypoint_detect = keypoint_weights_pth

        if weights_keypoint_detect == "":
            weights_keypoint_detect = None

            validate_pth_file(HN_weights_pth)
            weights_hybridnet = HN_weights_pth
            if weights_hybridnet == "":
                weights_hybridnet = None
        else:
            weights_hybridnet = None

    else:
        weights_keypoint_detect = None
        weights_hybridnet = weights_pretrain

    assert training_mode in ['3D_only', 'last_layers', 'bifpn', 'all']
    mode = training_mode
    if mode == '3D_only':
        finetune = False
    else:
        finetune = True
    assert torch.cuda.device_count() > 0
    train_interface.train_hybridnet(proj_name, num_epochs,
                                    weights_keypoint_detect, weights_hybridnet,
                                    mode, finetune, bar_position=bar_position, bar_desc=bar_desc)
    print('Training finished! Your HybridNet is '
          'ready for prediction, have fun :)')


def train_keypoint_detect(proj_name, weights='None', num_epochs=50, bar_position=0, bar_desc=''):
    projectManager = ProjectManager()
    if not projectManager.load(proj_name):
        print(f"Could not load Project {proj_name}!")
        return
    assert torch.cuda.device_count() > 0
    train_interface.train_efficienttrack('KeypointDetect', proj_name,
                                         num_epochs, weights, bar_position=bar_position, bar_desc=bar_desc)
    print('{Training finished! Your KeypointDetect network is '
          'ready for prediction, have fun :)')


def train_center_detect(proj_name, weights='None', num_epochs=50, bar_position=0, bar_desc=''):
    projectManager = ProjectManager()
    if not projectManager.load(proj_name):
        print(f"Could not load Project {proj_name}!")
        return
    assert torch.cuda.device_count() > 0
    train_interface.train_efficienttrack('CenterDetect', proj_name,
                                         num_epochs, weights, bar_position=bar_position, bar_desc=bar_desc)
    print('Training finished! Your CenterDetect network is '
          'ready for prediction, have fun :)')


def train(project, model, num_epochs, verbose, id):
    bar_desc = f"{project} | {model}"

    if model == 'cd':
        train_center_detect(
            project, num_epochs=num_epochs, bar_position=id, bar_desc=bar_desc)

    elif model == 'hn':
        train_hybridnet(project, num_epochs=num_epochs,
                        bar_position=id, bar_desc=bar_desc)

    elif model == 'kd':
        train_keypoint_detect(
            project, num_epochs=num_epochs, bar_position=id, bar_desc=bar_desc)

    else:
        print(
            f'unrecognized training model name {model} (choose from: hn, cd, kd)')


def train_jarvis(combos, epochs, verbose):
    """Trains Jarvis (pronounced \'Yar-vuh-s\' in BULK.

    Arguments:
        combos {list of tuple} --- List of project:model tuples to train
        epochs {int} --- Num epochs for training
        verbose {bool} ---
    """
    n_proc = len(combos)
    projects, models = tuple(zip(*combos))

    # Special case: serial allows for debugging
    if n_proc == 1:
        print("single-thread mode:")
        train(combos[0][0], combos[0][1], epochs, verbose, 0)

    p_args = list(zip(*[
        projects,
        models,
        [epochs, ] * n_proc,
        [verbose, ] * n_proc,
        [i for i in range(n_proc)]
    ]))

    if len(p_args) > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            futures = {executor.submit(
                train, *args): args for args in p_args}
            for future in concurrent.futures.as_completed(futures):
                args = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'Exception occurred: {exc}')
                    # Handle exception here if needed
                    pass
