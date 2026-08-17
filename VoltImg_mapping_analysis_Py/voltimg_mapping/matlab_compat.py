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
import warnings
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

    MATLAB's colon operator additionally snaps a limit that is within a few
    ulps of an integer onto that integer, so ``1:0.29*10000`` has 2900 elements
    even though ``0.29*10000 == 2899.9999999999995`` (verified in R2025b). A
    plain ``floor`` would give 2899 and shift every downstream window index.
    """
    x = float(x)
    nearest = round(x)
    if abs(x - nearest) <= 4.0 * np.spacing(max(1.0, abs(x))):
        x = float(nearest)
    if x < 1:
        return 0
    return int(math.floor(x))


def matlab_round(x: float) -> int:
    """MATLAB ``round``: half away from zero (Python/NumPy round is half-to-even).

    ``round(50.5)`` is 51 in MATLAB but 50 for Python's builtin. Window-length
    computations (``round(ms/1000*freq)``) land exactly on ``.5`` for real
    sampling rates, so banker's rounding changes window sizes by one sample.
    """
    x = float(x)
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


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
    """MATLAB ``std(x)`` default: sample std, N-1 normalization (ddof=1).

    MATLAB quirk (documented in ``var.m``): with a single observation the
    denominator N is used instead of N-1, so ``std(5)`` is 0, not NaN.
    ``np.std(..., ddof=1)`` on one sample gives NaN.
    """
    x = np.asarray(x, dtype=float)
    n = x.size if axis is None else x.shape[axis]
    if n == 1:
        # N normalization: 0 for a finite sample, NaN propagates naturally.
        return np.std(x, axis=axis, ddof=0)
    return np.std(x, axis=axis, ddof=1)


def std_n1_omitnan(x: np.ndarray, axis=None) -> np.ndarray:
    """MATLAB ``std(x, 0, axis, 'omitnan')``: N-1, NaN-aware.

    N-1 uses the per-column count of *finite* samples, matching MATLAB --
    except that a slice with exactly ONE non-NaN sample yields 0 in MATLAB
    (single-observation N normalization, ``std([NaN; 4], 0, 1, 'omitnan')``
    is 0, verified in R2025b) where ``np.nanstd(..., ddof=1)`` yields NaN.
    A slice with zero non-NaN samples stays NaN in both.
    """
    x = np.asarray(x, dtype=float)
    n_finite = np.sum(~np.isnan(x), axis=axis)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanstd(x, axis=axis, ddof=1)
    return np.where(n_finite == 1, 0.0, out)


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


def imgaussfilt(img: np.ndarray, sigma: float, filter_size: int | None = None) -> np.ndarray:
    """MATLAB ``imgaussfilt(img, sigma[, 'FilterSize', k])``.

    MATLAB's default filter size is ``2*ceil(2*sigma)+1`` with 'replicate'
    padding. scipy ``gaussian_filter`` truncates by radius; we set ``truncate``
    so the kernel half-width matches (radius = (filter_size-1)/2), and use
    mode='nearest' to mirror 'replicate'. ``filter_size`` must be odd when
    given (MATLAB requires odd FilterSize); fibermetric passes
    ``2*ceil(3*sigma)+1``.
    """
    if filter_size is None:
        radius = int(math.ceil(2 * sigma))
    else:
        radius = (int(filter_size) - 1) // 2
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

    scipy's odd-reflection extension and steady-state initial conditions match
    MATLAB's (verified against filtfilt.m), but the DEFAULT pad length does
    not: MATLAB pads ``3*(max(len(a),len(b)) - 1)`` samples per end
    (``l = max(1, 3*filtord(b,a))``, filtfilt.m:211) while scipy defaults to
    ``3*max(len(a),len(b))``. For the 4th-order Butterworth used here that is
    12 vs 15, which changes edge transients. Pass MATLAB's padlen explicitly.
    """
    x = np.asarray(x, dtype=float).ravel()
    padlen = 3 * (max(len(a), len(b)) - 1)
    return filtfilt(b, a, x, padlen=padlen)


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

    Masks below are verbatim ``getnhood(strel('disk', r))`` output from MATLAB
    R2025b. Two non-obvious facts about MATLAB's disk (see strel.m,
    MakeDiskStrel): for r < 3 the decomposition is bypassed and the strel is a
    plain Euclidean disk (r=2 is the 13-px diamond, NOT an octagon); for
    r >= 3 the periodic-line decomposition yields a mask that is SMALLER than
    a Euclidean disk (axial half-extent r-1, e.g. r=5 is 9x9 with 69 px). A
    Euclidean fallback is therefore wrong for every r >= 3, so unverified
    large radii raise instead of silently diverging -- add the MATLAB mask if
    the pipeline ever uses a new radius.
    """
    presets = {
        2: np.array([
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ], dtype=bool),
        3: np.ones((5, 5), dtype=bool),
        5: np.array([
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
        ], dtype=bool),
    }
    if radius in presets:
        return presets[radius]
    if radius < 3:
        # MATLAB forces n=0 (no decomposition) for r < 3: exact Euclidean disk.
        L = np.arange(-radius, radius + 1)
        xx, yy = np.meshgrid(L, L)
        return (xx ** 2 + yy ** 2) <= radius ** 2
    raise ValueError(
        f"strel_disk({radius}): no verified MATLAB mask for r >= 3 beyond the "
        "presets. MATLAB's decomposed disk is smaller than a Euclidean disk, "
        "so a fallback would silently break parity. Print "
        f"getnhood(strel('disk',{radius})) in MATLAB and add it to the presets."
    )


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
