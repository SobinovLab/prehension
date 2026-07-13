# Suppress threadpoolctl's "Found Intel OpenMP ('libiomp') and LLVM OpenMP ('libomp') loaded at
# the same time" RuntimeWarning. Both runtimes get pulled in (MKL/NumPy -> libiomp, PyTorch/
# Kilosort -> libomp); on Windows this coexistence is benign. Filter is set here, before the
# neural submodules (which lazily import spikeinterface) are imported.
import warnings

warnings.filterwarnings(
    "ignore",
    message="Found Intel OpenMP.*",
    category=RuntimeWarning,
    module="threadpoolctl",
)

__all__ = [
    'config', 'io_streams', 'ttl_sync', 'preprocessing', 'spike_sorting',
    'postprocessing', 'export_phy', 'export_nwb', 'pipeline']

from . import config
from . import io_streams
from . import ttl_sync
from . import preprocessing
from . import spike_sorting
from . import postprocessing
from . import export_phy
from . import export_nwb
from . import pipeline
