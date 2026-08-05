#!python3
# -*- coding: utf-8 -*-
"""
Postprocessing (optional / diagnostic):
    build_sorting_analyzer --- analyzer + extensions needed for metrics & phy.
    compute_quality_metrics --- quality_metrics.csv (diagnostic).
    curate_by_quality --- quality triage -> analyzer_curated (for nwb_units=curated).

None of this is required for the NWB unless curated units are requested.

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
import shutil

from . import config
from ..tools import io
from ..tools.logs import rs, ws


def build_sorting_analyzer(cfg):
    """Create the analyzer and compute the extensions used by metrics and phy."""
    import spikeinterface.full as si

    cfg.ensure_work_folder()
    rs('Building the sorting analyzer.')

    sorting = config.load_sorting(cfg)
    rec_pre = config.load_preprocessed(cfg)

    folder = cfg.subfolders()['analyzer']
    if os.path.exists(folder):
        shutil.rmtree(folder)

    analyzer = si.create_sorting_analyzer(sorting=sorting, recording=rec_pre,
                                          sparse=cfg.sparse, format='binary_folder',
                                          folder=folder)
    jk = cfg.job_kwargs
    analyzer.compute('random_spikes', method='uniform', max_spikes_per_unit=500)
    analyzer.compute('waveforms', ms_before=1.5, ms_after=2.0, **jk)
    analyzer.compute('templates', operators=['average', 'median', 'std'])
    analyzer.compute('noise_levels')
    analyzer.compute('correlograms')
    analyzer.compute('unit_locations')
    analyzer.compute('spike_amplitudes', **jk)
    analyzer.compute('template_similarity')
    analyzer.compute('principal_components', n_components=5,
                     mode='by_channel_local', **jk)
    rs('Analyzer saved -> {}'.format(folder))
    return analyzer


def compute_quality_metrics(cfg):
    """Compute standard quality metrics and save them to quality_metrics.csv."""
    cfg.ensure_work_folder()
    rs('Computing quality metrics.')
    analyzer = config.load_analyzer(cfg)
    metrics = analyzer.compute('quality_metrics', metric_names=[
        'firing_rate', 'presence_ratio', 'snr', 'isi_violation',
        'amplitude_cutoff']).get_data()
    out = os.path.join(cfg.work_folder, 'quality_metrics.csv')
    metrics.to_csv(out)
    rs('Saved -> {}'.format(out))
    return metrics


def curate_by_quality(cfg):
    """Quality triage -> analyzer_curated (subset of units).  Returns kept ids."""
    cfg.ensure_work_folder()
    rs('Curating by quality metrics.')
    analyzer = config.load_analyzer(cfg)

    ext = analyzer.get_extension('quality_metrics')
    metrics = ext.get_data() if ext is not None else compute_quality_metrics(cfg)

    curated_folder = cfg.subfolders()['analyzer_curated']
    if os.path.exists(curated_folder):
        shutil.rmtree(curated_folder)

    keep_unit_ids = list(analyzer.unit_ids)
    try:
        keep = metrics.query(cfg.curation_query)
        keep_unit_ids = list(keep.index.values)
        print('Query: {}'.format(cfg.curation_query))
        print('Keeping {} / {} units'.format(len(keep_unit_ids), len(metrics)))
    except Exception as e:
        ws('Could not apply curation query (keeping all units): {}'.format(e))

    if len(keep_unit_ids) == 0:
        ws('No units passed the triage; not creating a curated analyzer.')
        return []

    analyzer.select_units(keep_unit_ids, folder=curated_folder, format='binary_folder')
    rs('Saved curated analyzer -> {}'.format(curated_folder))
    io.save_json(dict(curation_query=cfg.curation_query,
                      n_units_total=int(len(metrics)),
                      n_units_kept=int(len(keep_unit_ids)),
                      kept_unit_ids=[str(u) for u in keep_unit_ids]),
                 os.path.join(cfg.work_folder, 'curation_summary.json'))
    return keep_unit_ids
