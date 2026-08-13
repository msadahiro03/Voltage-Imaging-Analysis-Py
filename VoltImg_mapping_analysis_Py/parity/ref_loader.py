"""Reader for the MATLAB v7.3 (HDF5) reference .mat produced by the
050926_parallel MultiCell mapping script.

Handles MATLAB->HDF5 idioms:
  * everything is transposed (MATLAB col-major <-> HDF5 row-major), so a MATLAB
    (numFrames x nTrials) matrix reads as (nTrials, numFrames);
  * cell arrays are datasets of object refs; deref with f[ref];
  * char arrays are uint16 code points;
  * empty MATLAB arrays read as a 2-element uint64 dims marker.

Coordinates: MATLAB ROI vectors are 1-based; callers convert to 0-based.
"""
from __future__ import annotations

import os
import glob
import numpy as np
import h5py

BASE = ('/Users/masatosadahiro/Documents/Code/matlab_scripts/'
        'Voltage_Imaging_Analysis_Matlab/Voltage-Imaging-Analysis-ML/'
        'MC Imaging Data Sample')
_STEM = ('voltMapping_Analysis_SCNNCRE_hSynyASAP7Kv_DIOChrom2s_IC_InVivo_'
         'MS26_21_A7P_Chrome2s_050726_FOV1_2PMapping_Day1')
MAT_PATH = os.path.join(
    BASE, 'Parity_Check_MLAnalysisData',
    _STEM + '_MultiCellAnalysis_MCfineROI_laserRowArtifact.mat')
MC_DIR = os.path.join(
    BASE, _STEM + '_MultiCellAnalysis_MCfineROI_laserRowArtifact_parallel',
    'Motion_Corrected_Tiffs')


_MC_SORTED = None


def _mc_sorted():
    """Alphabetically-sorted list of MC TIFFs (matches MATLAB dir() order,
    which is how trial index tt maps to a file)."""
    global _MC_SORTED
    if _MC_SORTED is None:
        _MC_SORTED = sorted(glob.glob(os.path.join(MC_DIR, '*_mc.tif')))
    return _MC_SORTED


def mc_tiff_path(trial_1based: int) -> str:
    """MC TIFF for 1-based trial index, by sorted dir order."""
    return _mc_sorted()[trial_1based - 1]


class Ref:
    def __init__(self, path: str = MAT_PATH):
        self.f = h5py.File(path, 'r')

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # --- scalars -------------------------------------------------------
    def scalar(self, key: str) -> float:
        return float(np.array(self.f[key]).squeeze())

    def char(self, key: str) -> str:
        a = np.array(self.f[key]).ravel()
        return ''.join(chr(int(x)) for x in a)

    def is_empty(self, key: str) -> bool:
        o = self.f[key]
        # MATLAB empty -> uint64 dims marker like [0 0]
        a = np.array(o).ravel()
        return a.dtype.kind == 'u' and a.size == 2 and int(a[0]) == 0

    # --- ROI cell arrays ----------------------------------------------
    def _deref(self, ref):
        return self.f[ref]

    def roi_per_cell(self, key: str, cell_idx0: int) -> np.ndarray:
        """Per-cell (not per-trial) ROI vector, e.g. roughRoiXAllCells{nn}.
        Returns 1-based MATLAB index vector as int64 array."""
        outer = self.f[key]           # (1, nCells) of refs
        inner = self._deref(outer[0, cell_idx0])
        return np.array(inner).ravel().astype(np.int64)

    def roi_per_trial(self, key: str, cell_idx0: int, trial_idx0: int):
        """Per-cell, per-trial ROI vector, e.g. fineRoiXAllCells{nn}{tt}.
        Returns 1-based MATLAB index vector, or None if empty."""
        outer = self.f[key]                       # (1, nCells) refs
        cell = self._deref(outer[0, cell_idx0])   # (1, nTrials) refs
        r = cell[0, trial_idx0] if cell.ndim > 1 else cell[trial_idx0]
        arr = self._deref(r)
        a = np.array(arr).ravel()
        if a.dtype.kind == 'u' and a.size == 2 and int(a[0]) == 0:
            return None  # empty
        return a.astype(np.int64)

    # --- analysisStruct 2D traces (numFrames x nTrials in MATLAB) ------
    def trace_matrix(self, field: str, cell_1based: int) -> np.ndarray:
        """Return analysisStruct.<field>_cell<N> as (nTrials, numFrames)
        exactly as stored (HDF5-transposed). Row tt == trial tt."""
        return np.array(self.f['analysisStruct'][f'{field}_cell{cell_1based}'])

    # --- nested {cond}{holo} cell of columns --------------------------
    def holo_cell(self, field: str, cell_1based: int):
        """Return nested list [cond][holo] -> ndarray for an analysisStruct
        {1,nCond} cell whose entries are {1,nHolo} cells of numeric arrays."""
        top = self.f['analysisStruct'][f'{field}_cell{cell_1based}']  # (1,nCond) refs
        conds = []
        ncond = top.shape[1] if top.ndim > 1 else top.shape[0]
        for cc in range(ncond):
            cref = top[0, cc] if top.ndim > 1 else top[cc]
            cond = self._deref(cref)  # (1,nHolo) refs
            holos = []
            nholo = cond.shape[1] if cond.ndim > 1 else cond.shape[0]
            for hh in range(nholo):
                hr = cond[0, hh] if cond.ndim > 1 else cond[hh]
                holos.append(np.array(self._deref(hr)))
            conds.append(holos)
        return conds
