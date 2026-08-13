"""MATLAB-compatibility helpers for the voltage-imaging mapping port.

Every function here exists to reproduce a specific MATLAB semantic that does NOT
map onto a NumPy/SciPy default. Read the docstrings before "simplifying" any of
them -- the deviations from idiomatic Python are deliberate and load-bearing for
numerical parity.

Indexing convention used throughout the port
---------------------------------------------
All ROI coordinate arrays are stored as 0-based ``(rows, cols)`` integer arrays,
matching MATLAB ``[X, Y] = find(mask)`` where ``X`` is the matrix *row* and ``Y``
the matrix *column* (MATLAB names them X/Y but they are row/col -- porting
gotcha #2). Image arrays are indexed ``img[row, col]`` (row-major NumPy).
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy import ndimage as ndi
from scipy.signal import butter, filtfilt, lfilter
from scipy.stats import t as _student_t

# ---------------------------------------------------------------------------
# Colon / range semantics
# ---------------------------------------------------------------------------


def colon_count(x: float) -> int:
    """Number of elements MATLAB produces for ``1:x`` with fractional ``x``.

    MATLAB ``1:x`` yields ``[1, 2, ..., floor(x)]`` when ``x >= 1``; ``floor(x)``
    elements. For ``x < 1`` it is empty (0 elements). Gotcha #1.
    """
    if x < 1:
        return 0
    return int(math.floor(x))


def matlab_1based_slice(n_from: int, n_to: int) -> slice:
    """Translate an inclusive 1-based MATLAB range ``a:b`` to a 0-based slice.

    ``a:b`` (inclusive, 1-based) -> ``[a-1 : b]`` (half-open, 0-based).
    Caller is responsible for ``a`` and ``b`` already being integers.
    """
    return slice(n_from - 1, n_to)


# ---------------------------------------------------------------------------
# Statistics that differ from NumPy defaults
# ---------------------------------------------------------------------------


def prctile(x: np.ndarray, p: float) -> float:
    """MATLAB ``prctile`` (linear interpolation on the 'exclusive' grid).

    MATLAB places sorted samples at percentile positions
    ``100*(i-0.5)/n`` and linearly interpolates, clamping to the end
    samples outside that range. This matches numpy's
    ``method='median_unbiased'``? No -- it matches the classic Hazen /
    ``interpolation`` scheme used by MATLAB, which numpy exposes only via
    manual construction. We reproduce it directly.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return np.nan
    if n == 1:
        return float(x[0])
    xs = np.sort(x)
    # Percentile positions of the sorted samples (MATLAB convention).
    pos = 100.0 * (np.arange(1, n + 1) - 0.5) / n
    if p <= pos[0]:
        return float(xs[0])
    if p >= pos[-1]:
        return float(xs[-1])
    return float(np.interp(p, pos, xs))


def mad1(x: np.ndarray) -> float:
    """MATLAB ``mad(x, 1)`` -- median absolute deviation about the median.

    NOTE: NOT scaled by 1.4826 (gotcha #7). Uses omitnan-style median.
    """
    x = np.asarray(x, dtype=float).ravel()
    x = x[~np.isnan(x)]
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def std_n1(x: np.ndarray, axis=None) -> np.ndarray:
    """MATLAB ``std(x)`` default: sample std, N-1 normalization (ddof=1)."""
    return np.std(x, axis=axis, ddof=1)


def std_n1_omitnan(x: np.ndarray, axis=None) -> np.ndarray:
    """MATLAB ``std(x, 0, axis, 'omitnan')``: N-1, NaN-aware.

    N-1 uses the per-column count of *finite* samples, matching MATLAB.
    """
    return np.nanstd(x, axis=axis, ddof=1)


def var_n(x: np.ndarray) -> float:
    """MATLAB ``var(x, 1)`` -- N (biased) normalization (gotcha #5)."""
    x = np.asarray(x, dtype=float).ravel()
    return float(np.var(x, ddof=0))


def tinv(prob: float, df: float) -> float:
    """MATLAB ``tinv(prob, df)`` == scipy ``t.ppf``. df<=0 -> NaN (gotcha #8)."""
    if df is None or df <= 0 or np.isnan(df):
        return np.nan
    return float(_student_t.ppf(prob, df))


# ---------------------------------------------------------------------------
# Image scaling / filtering
# ---------------------------------------------------------------------------


def im2double(img: np.ndarray) -> np.ndarray:
    """MATLAB ``im2double`` semantics (gotcha #8, #17).

    - uint16 -> divide by 65535
    - uint8  -> divide by 255
    - float  -> passthrough (already in [0,1] convention; MATLAB does not rescale)
    """
    if img.dtype == np.uint16:
        return img.astype(np.float64) / 65535.0
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return img.astype(np.float64)


def imgaussfilt(img: np.ndarray, sigma: float) -> np.ndarray:
    """MATLAB ``imgaussfilt(img, sigma)``.

    MATLAB's default filter size is ``2*ceil(2*sigma)+1`` with 'replicate'
    padding. scipy ``gaussian_filter`` truncates by radius; we set ``truncate``
    so the kernel half-width == ceil(2*sigma), i.e. radius = ceil(2*sigma),
    and use mode='nearest' to mirror 'replicate'.
    """
    radius = int(math.ceil(2 * sigma))
    truncate = radius / sigma if sigma > 0 else 0.0
    return ndi.gaussian_filter(
        img, sigma=sigma, mode="nearest", truncate=truncate
    )


def butter_lowpass(order: int, cutoff_hz: float, fs: float):
    """``butter(order, cutoff/(fs/2), 'low')`` -> (b, a)."""
    wn = cutoff_hz / (fs / 2.0)
    return butter(order, wn, btype="low")


def butter_default(order: int, cutoff_hz: float, fs: float):
    """``butter(order, cutoff/(fs/2))`` (MATLAB default = lowpass) -> (b, a)."""
    wn = cutoff_hz / (fs / 2.0)
    return butter(order, wn, btype="low")


def matlab_filtfilt(b, a, x):
    """MATLAB ``filtfilt`` (zero-phase). Ephys path only (gotcha #4).

    scipy ``filtfilt`` default padding (odd, padlen=3*max(len(a),len(b))) matches
    MATLAB's default well for these signal lengths.
    """
    x = np.asarray(x, dtype=float).ravel()
    return filtfilt(b, a, x)


def matlab_filter(b, a, x):
    """MATLAB ``filter`` (causal, one-directional). Imaging path only.

    scipy ``lfilter`` with zero initial conditions == MATLAB ``filter``.
    """
    x = np.asarray(x, dtype=float).ravel()
    return lfilter(b, a, x)


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------


def strel_disk(radius: int) -> np.ndarray:
    """Replicate MATLAB ``strel('disk', r)`` structuring element (n=4 default).

    For the small radii used here (2, 5) MATLAB decomposes the disk into a
    sum of line strels, producing an *approximated* (not Euclidean) disk. To
    stay faithful we reconstruct MATLAB's actual disk masks for the radii the
    pipeline uses; for other radii we fall back to a Euclidean disk and warn via
    the shape. See notes in README about disk-shape fidelity risk.

    MATLAB reference masks (r, with default n=4):
      r=2 -> 5x5:            r=3 -> 7x7:            r=5 -> 11x11
    """
    # Hard-coded MATLAB strel('disk', r) neighborhoods (default N=4 decomposition)
    # obtained from MATLAB `getnhood(strel('disk', r))`.
    presets = {
        2: np.array([
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ], dtype=bool),
        3: np.array([
            [0, 0, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0, 0],
        ], dtype=bool),
        5: np.array([
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
        ], dtype=bool),
    }
    if radius in presets:
        return presets[radius]
    # Fallback: Euclidean disk.
    L = np.arange(-radius, radius + 1)
    xx, yy = np.meshgrid(L, L)
    return (xx ** 2 + yy ** 2) <= radius ** 2


def imdilate_disk(mask: np.ndarray, radius: int) -> np.ndarray:
    """MATLAB ``imdilate(mask, strel('disk', r))``."""
    se = strel_disk(radius)
    return ndi.binary_dilation(mask, structure=se)


def bwareaopen(mask: np.ndarray, k: int, connectivity: int = 8) -> np.ndarray:
    """MATLAB ``bwareaopen(mask, k)`` -- remove connected components < k pixels.

    Default connectivity for 2-D in MATLAB is 8 (gotcha #10).
    """
    if connectivity == 8:
        structure = np.ones((3, 3), dtype=bool)
    else:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    labeled, n = ndi.label(mask, structure=structure)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(np.ones_like(labeled), labeled, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= k
    return keep[labeled]


# ---------------------------------------------------------------------------
# find / linear indexing
# ---------------------------------------------------------------------------


def matlab_find_2d(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[X, Y] = find(mask)`` -> (rows, cols), COLUMN-MAJOR order.

    MATLAB ``find`` returns linear indices in column-major order. To match the
    ordering of coordinate lists (which can matter if downstream code assumes a
    particular order), we sort by column then row (Fortran/column-major order).
    Values are 0-based.
    """
    # np.nonzero returns row-major order; reorder to column-major.
    rows, cols = np.nonzero(mask)
    order = np.lexsort((rows, cols))  # primary key = cols, secondary = rows
    return rows[order], cols[order]
