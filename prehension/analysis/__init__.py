#!python3
# -*- coding: utf-8 -*-
"""
Analysis of prehension neural and behavioural data: state spaces (PCA / demixed PCA),
neuron-force cross-correlation, condition decoding, and Poisson GLM encoding models.

These orchestrators reuse the compute layers in tools (stats / decoding / encoding) and
neural_processing.common.population, the cross-session pooling and drawers in
neural_plotting.common, and the behavioural pooling / GLM assembly here.

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
__all__ = [
    'figure_spaces_pooled', 'figure_spaces_dpca_pooled', 'figure_cross_correlation',
    'figure_classification_time', 'behavior_pooling', 'spaces_behavior', 'encoding',
    'encoding_scatter', 'encoding_summary', 'figure_umap', 'decoding']

from . import figure_spaces_pooled
from . import figure_spaces_dpca_pooled
from . import figure_cross_correlation
from . import figure_classification_time
from . import behavior_pooling
from . import spaces_behavior
from . import encoding
from . import encoding_scatter
from . import encoding_summary
from . import figure_umap
from . import decoding
