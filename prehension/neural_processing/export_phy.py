#!python3
# -*- coding: utf-8 -*-
"""
Phy export and visual report (optional / diagnostic).

export_to_phy --- Phy template-gui dataset (all units) for manual curation.
export_report --- QC images from the curated analyzer, or the full one with
    all_units=True (-> reports/all_units_report/).

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
import os

from . import config
from ..tools.logs import rs, ws


def export_to_phy(cfg):
    """Export a Phy template-gui dataset from the (full) analyzer."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Exporting to Phy.')
    analyzer = config.load_analyzer(cfg)
    phy_folder = cfg.subfolders()['phy']
    si.export_to_phy(sorting_analyzer=analyzer, output_folder=phy_folder,
                     remove_if_exists=True, **cfg.job_kwargs)
    rs('Exported Phy dataset -> {}'.format(phy_folder))
    print('Curate with:\n  cd {}\n  phy template-gui params.py'.format(phy_folder))


def export_report(cfg, all_units=False):
    """Export QC images; all_units=True reports on every unit (full analyzer)."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Exporting report.')
    full = config.load_analyzer(cfg)

    curated_folder = cfg.subfolders()['analyzer_curated']
    if all_units:
        report_analyzer = full
        report_folder = os.path.join(cfg.work_folder, 'reports', 'all_units_report')
        print('all_units: report covers ALL units.')
    elif os.path.exists(curated_folder):
        report_analyzer = si.load_sorting_analyzer(str(curated_folder))
        report_folder = cfg.subfolders()['report']
        print('Using curated analyzer for the report.')
    else:
        report_analyzer = full
        report_folder = cfg.subfolders()['report']
        print('No curated analyzer; using full analyzer for the report.')

    try:
        si.export_report(report_analyzer, output_folder=report_folder,
                         remove_if_exists=True, format='png', **cfg.job_kwargs)
        rs('Exported report ({} units) -> {}'.format(
            report_analyzer.get_num_units(), report_folder))
    except Exception as e:
        ws('export_report failed: {}'.format(e))
