"""Trial excluder (Stage H).

The external ``VoltImg_mapping_analysis_MultiCell_trialExcluder.m`` invoked at
line 1335 of the main script lives at
``archive_MultiCell/VoltImg_mapping_analysis_MultiCell_trialExcluder.m`` in the
MATLAB repo; its logic is identical to the multi-cell version inlined in
``globalPerTrialF0_...laserRowArtifact.m`` lines 372-512.
That version uses:
  * amplitude reject:  any sample < -2.5 * std(allTrials(:))   -> NaN the column
  * late-peak reject:  argmax(excl column) index (1-based) > 45 -> NaN the column

(The older single-cell ``VoltImg_mapping_analysis_trialExcluder.m`` present in
the repo uses -2 and >40; it is NOT the one the main script calls. We follow the
spec and use -2.5 / 45.)

std is over ALL elements of the (Lholo x nTrials) matrix, N-1 (MATLAB ``std``
default). Reductions are omitnan.

Data model: per cell, ``holo_all[cc][hh]`` is an (Lholo, nTrialsForHolo) float
array (columns = trials, NaN for ephys-excluded). Everything is 0-based here.
"""

from __future__ import annotations

import numpy as np

from .matlab_compat import std_n1, std_n1_omitnan, tinv


def _std_over_all(mat: np.ndarray) -> float:
    """MATLAB ``std(mat(:))`` -- N-1 over all elements, NaN-aware.

    MATLAB ``std`` WITHOUT 'omitnan' would propagate NaN, but the excluded
    columns are all-NaN and the spec's std feeds only the threshold comparison.
    MATLAB ``std(X(:))`` (no omitnan) returns NaN if any NaN is present, which
    would make ``x < -2.5*NaN`` false for every trial (no amplitude rejection).
    We reproduce MATLAB EXACTLY: no omitnan here.
    """
    flat = np.asarray(mat, dtype=float).ravel()
    if flat.size == 0:
        return np.nan  # MATLAB std([]) is NaN
    if flat.size == 1:
        # MATLAB single-observation rule: denominator N, so std(scalar) is 0
        # for a finite value (NaN input still propagates NaN).
        return float(np.std(flat, ddof=0))
    return float(np.std(flat, ddof=1))  # NaN-propagating, like MATLAB std(X(:))


def _matlab_max_index_1based(col: np.ndarray) -> int:
    """MATLAB ``[~, i] = max(x)``: NaN is ignored; all-NaN returns index 1.

    ``np.argmax`` instead returns the index of the FIRST NaN, which would
    spuriously trip the late-peak test on a partially-NaN column.
    """
    if np.all(np.isnan(col)):
        return 1
    return int(np.nanargmax(col)) + 1


def _ci_columns(mat, up_or_down, confidence_level=0.95):
    """Per-timepoint CI [lower, upper] with MATLAB sign-flip on UpOrDown.

    Mirrors MATLAB CI blocks: means = nanmean(mat,2);
    SEM = std(mat,0,2,'omitnan')/sqrt(ncols); t = tinv((1+cl)/2, ncols-1);
    if UpOrDown=='2': [lo, hi]; if '1': [-lo, -hi]. Returns (Lholo, 2).
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[1] == 0:
        return np.full((mat.shape[0] if mat.ndim >= 1 else 0, 2), np.nan)
    n_cols = mat.shape[1]
    means = np.nanmean(mat, axis=1)
    sem = std_n1_omitnan(mat, axis=1) / np.sqrt(n_cols)
    t_score = tinv((1 + confidence_level) / 2, n_cols - 1)
    moe = t_score * sem
    lower = means - moe
    upper = means + moe
    if str(up_or_down) == "2":
        return np.column_stack([lower, upper])
    else:  # '1'
        return np.column_stack([-lower, -upper])


def run_trial_excluder(holo_all, filt_holo_all, n_holos, up_or_down,
                       late_peak_thresh=45, amp_thresh_k=2.5):
    """Apply the trial excluder to one cell's holo-sorted dF/F.

    Parameters
    ----------
    holo_all, filt_holo_all : list over conds; each a list over holos of
        (Lholo, nTrials) arrays (unfiltered / filtered dF/F).
    n_holos : list/array of holo counts per condition.

    Returns a dict with:
        'std', 'std_filt'                 : list[cond][holo] scalar
        'excl_all', 'excl_filt_all'       : list[cond][holo] (Lholo, nTrials)
        'excl_mean', 'excl_filt_mean'     : list[cond][holo] (Lholo,)
        'excl_ci', 'excl_filt_ci'         : list[cond][holo] (Lholo, 2)
    matching the MATLAB per-cell std*/excl* outputs.
    """
    n_conds = len(holo_all)

    std_arr = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    std_filt_arr = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    excl_all = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    excl_filt_all = [[None] * int(n_holos[cc]) for cc in range(n_conds)]

    # Pass 1: per-holo std over all elements.
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            std_arr[cc][hh] = _std_over_all(holo_all[cc][hh])
            std_filt_arr[cc][hh] = _std_over_all(filt_holo_all[cc][hh])

    # Pass 2: amplitude rejection (any sample < -k*std -> NaN whole column).
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            mat = np.asarray(holo_all[cc][hh], dtype=float)
            fmat = np.asarray(filt_holo_all[cc][hh], dtype=float)
            excl = mat.copy()
            fexcl = fmat.copy()
            s = std_arr[cc][hh]
            fs = std_filt_arr[cc][hh]
            n_tr = mat.shape[1] if mat.ndim == 2 else 0
            for tt in range(n_tr):
                # NaN-propagating comparison matches MATLAB (any(x < -2.5*NaN)
                # is False, so a NaN std leaves the column intact).
                if np.any(mat[:, tt] < -amp_thresh_k * s):
                    excl[:, tt] = np.nan
                if np.any(fmat[:, tt] < -amp_thresh_k * fs):
                    fexcl[:, tt] = np.nan
            excl_all[cc][hh] = excl
            excl_filt_all[cc][hh] = fexcl

    # Pass 3: late-peak rejection. If a column has any NaN in the FILTERED excl
    # trace, skip it (MATLAB `continue`). Otherwise argmax (1-based) > 45 -> NaN.
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            excl = excl_all[cc][hh]
            fexcl = excl_filt_all[cc][hh]
            n_tr = excl.shape[1] if excl.ndim == 2 else 0
            for tt in range(n_tr):
                if np.any(np.isnan(fexcl[:, tt])):
                    continue
                # MATLAB max returns the first max index and ignores NaN.
                max_imaging_index = _matlab_max_index_1based(excl[:, tt])
                max_filt_index = _matlab_max_index_1based(fexcl[:, tt])
                if max_imaging_index > late_peak_thresh:
                    excl[:, tt] = np.nan
                if max_filt_index > late_peak_thresh:
                    fexcl[:, tt] = np.nan

    # Means + CIs on excluded data.
    excl_mean = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    excl_filt_mean = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    excl_ci = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    excl_filt_ci = [[None] * int(n_holos[cc]) for cc in range(n_conds)]

    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            excl_mean[cc][hh] = np.nanmean(excl_all[cc][hh], axis=1)
            excl_filt_mean[cc][hh] = np.nanmean(excl_filt_all[cc][hh], axis=1)
            excl_ci[cc][hh] = _ci_columns(excl_all[cc][hh], up_or_down)
            excl_filt_ci[cc][hh] = _ci_columns(excl_filt_all[cc][hh], up_or_down)

    return {
        "std": std_arr,
        "std_filt": std_filt_arr,
        "excl_all": excl_all,
        "excl_filt_all": excl_filt_all,
        "excl_mean": excl_mean,
        "excl_filt_mean": excl_filt_mean,
        "excl_ci": excl_ci,
        "excl_filt_ci": excl_filt_ci,
    }
