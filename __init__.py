"""GeneWeaver - GPU-accelerated CRISPR off-target search.

NOTE: importing this package must stay cheap and must NOT import numba, so the
CLI can set NUMBA_ENABLE_CUDASIM before numba is first loaded.
"""

__version__ = "0.1.0"
