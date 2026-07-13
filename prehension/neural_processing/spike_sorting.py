#!python3
# -*- coding: utf-8 -*-
"""
Spike sorting (shared).  Runs cfg.sorter_name on the preprocessed recording.

Copyright (C) 2026 Anton Sobinov
https://github.com/SobinovLab/prehension

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
from . import config
from ..tools.logs import rs, ws


def run_spike_sorting(cfg):
    """Run the configured sorter on the preprocessed recording."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Sorting with {}.'.format(cfg.sorter_name))

    try:
        installed = si.installed_sorters()
        print('Installed sorters:', installed)
        if cfg.sorter_name not in installed:
            ws('{!r} not in installed_sorters().'.format(cfg.sorter_name))
    except Exception as e:
        ws('Could not query installed sorters: {}'.format(e))

    rec_pre = config.load_preprocessed(cfg)
    sorter_folder = cfg.subfolders()['sorter']
    rs('Running -> {}'.format(sorter_folder))

    sorting = si.run_sorter(sorter_name=cfg.sorter_name, recording=rec_pre,
                            folder=sorter_folder, remove_existing_folder=True,
                            verbose=True, raise_error=True)
    print(sorting)
    rs('Found {} units.'.format(len(sorting.get_unit_ids())))
    return sorting
