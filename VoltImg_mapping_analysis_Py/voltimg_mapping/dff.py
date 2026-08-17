"""dF/F computation (Stage F) and per-holo means/CIs (Stage G).

Ports MATLAB lines 912-1323. This is the scientific core.

Data model
----------
For each cell nn we accumulate, per condition cc and holo ID hh, a list of
per-trial dF/F columns of length Lholo, where

    Lholo = ceil((ipi*nPulses + (preStimWindow+postStimWindow))/1000 * imagingFreq) + 2

Excluded trials contribute an all-NaN column of length Lholo (unfiltered and
filtered). Non-excluded trials contribute the windowed dF/F. F0 is stored per
trial as a scalar (NaN for excluded).

alphaScalar is HARD-OVERWRITTEN to 0.85 (gotcha #6). The robustfit computation
in MATLAB (lines 1181-1186) is dead code and intentionally NOT ported.
"""

from __future__ import annotations

import math

import numpy as np

from .artifact import roi_mean_per_frame_exclude_bad_rows
from .fextract import roi_mean_per_frame_crossproduct
from .matlab_compat import (
    butter_default,
    colon_count,
    matlab_filter,
    matlab_round,
    std_n1_omitnan,
    tinv,
)

ALPHA_SCALAR = 0.85


def lholo(ipi, n_pulses, pre_stim_window, post_stim_window, imaging_freq):
    """Canonical per-holo length (columns concatenated to this length)."""
    return int(math.ceil(
        (ipi * n_pulses + (pre_stim_window + post_stim_window)) / 1000
        * imaging_freq
    )) + 2


def init_cell_accumulators(n_cells, n_conds, n_holos):
    """Nested containers mirroring analysisStruct per-cell fields.

    Returns a dict of lists indexed [nn][cc][hh] -> list of columns/scalars, plus
    per-cell full-trial matrices are handled separately by the caller.
    """
    acc = {
        "holo_all": [],       # [nn][cc][hh] list of dFF columns
        "filt_holo_all": [],  # [nn][cc][hh] list of filtered dFF columns
        "f0_all": [],         # [nn][cc][hh] list of F0 scalars
    }
    for _ in range(n_cells):
        acc["holo_all"].append(
            [[[] for _ in range(int(n_holos[cc]))] for cc in range(n_conds)]
        )
        acc["filt_holo_all"].append(
            [[[] for _ in range(int(n_holos[cc]))] for cc in range(n_conds)]
        )
        acc["f0_all"].append(
            [[[] for _ in range(int(n_holos[cc]))] for cc in range(n_conds)]
        )
    return acc


def compute_trial_dff(
    roi_mean_f,               # (numFrames,) ROI F this trial/cell
    bkgrnd_mean_f,            # (numFrames,) neuropil F this trial/cell
    is_excluded,              # bool
    cc_1based,                # condition of this trial
    holo_seq_this_trial,      # 1-based holo IDs, length nHolos(cc)
    first_stim_vec,           # per-cond first stim times (s)
    imaging_freq, ipi, n_pulses, pre_stim_window, post_stim_window,
    up_or_down,               # '1' or '2'
):
    """Compute per-holo dF/F for one trial/cell (MATLAB 1179-1247).

    Returns (per_holo_records, roi_mean_f_corrected) where per_holo_records is a
    list of dicts keyed by holo ID with 'f0', 'dff', 'filtdff' (arrays/scalars,
    or NaN column for excluded). roi_mean_f_corrected is the full-trial corrected
    trace (or None for excluded, matching MATLAB where filt is [] but corrected
    is still computed).
    """
    roi_mean_f = np.asarray(roi_mean_f, dtype=float).ravel()
    bkgrnd_mean_f = np.asarray(bkgrnd_mean_f, dtype=float).ravel()

    # Neuropil correction (alphaScalar hard 0.85).
    roi_mean_f_corrected = roi_mean_f - ALPHA_SCALAR * bkgrnd_mean_f

    # Causal imaging filter (filter, NOT filtfilt -- gotcha #4).
    b_im, a_im = butter_default(4, 40, imaging_freq)
    if not is_excluded:
        roi_mean_f_corrected_filt = matlab_filter(b_im, a_im, roi_mean_f_corrected)
    else:
        roi_mean_f_corrected_filt = None

    L = lholo(ipi, n_pulses, pre_stim_window, post_stim_window, imaging_freq)
    # preStim length: 1:(preStimWindow/1000*imagingFreq)-1 -> that many samples.
    pre_len = colon_count((pre_stim_window / 1000 * imaging_freq) - 1)

    records = []
    for hh in range(len(holo_seq_this_trial)):
        fst = first_stim_vec[hh]
        # iHoloLo (1-based), iHoloHi (1-based).
        i_holo_lo = math.floor((fst - pre_stim_window / 1000) * imaging_freq)
        i_holo_hi = math.ceil((fst - pre_stim_window / 1000) * imaging_freq) + \
            math.ceil((ipi * n_pulses + (pre_stim_window + post_stim_window))
                      / 1000 * imaging_freq)
        holo_id = int(holo_seq_this_trial[hh])

        if is_excluded:
            records.append({
                "holo_id": holo_id,
                "f0": np.nan,
                "dff": np.full(L, np.nan),
                "filtdff": np.full(L, np.nan),
            })
            continue

        # roiFCorrectedThisHolo = roiMeanFCorrected(iHoloLo:iHoloHi) (1-based incl)
        seg = roi_mean_f_corrected[i_holo_lo - 1 : i_holo_hi]
        pre = seg[:pre_len]
        f0 = float(np.mean(pre))
        dff = (seg - f0) / f0
        if str(up_or_down) == "2":
            dff = -dff
        # '1' -> unchanged

        seg_f = roi_mean_f_corrected_filt[i_holo_lo - 1 : i_holo_hi]
        pre_f = seg_f[:pre_len]
        f0_f = float(np.mean(pre_f))
        dff_f = (seg_f - f0_f) / f0_f
        if str(up_or_down) == "2":
            filtdff = -dff_f
        else:
            filtdff = dff_f

        records.append({
            "holo_id": holo_id,
            "f0": f0,
            "dff": dff,
            "filtdff": filtdff,
        })

    return records, roi_mean_f_corrected, roi_mean_f_corrected_filt


def compute_trial_dff_common_f0(
    roi_mean_f, bkgrnd_mean_f, is_excluded, cc_1based, holo_seq_this_trial,
    first_stim_vec, imaging_freq, ipi, n_pulses, pre_stim_window,
    post_stim_window, up_or_down, start_time, f0_win_ms=50,
):
    """commonF0 variant (companion script): trial-common min-variance F0.

    Replaces the per-holo prestim-mean F0 with a single early-baseline F0 for the
    whole trial (min-variance sliding window in the prestim region), used for all
    holos. Everything else identical. Ports globalPerTrialF0 lines 185-289.
    """
    roi_mean_f = np.asarray(roi_mean_f, dtype=float).ravel()
    bkgrnd_mean_f = np.asarray(bkgrnd_mean_f, dtype=float).ravel()
    roi_mean_f_corrected = roi_mean_f - ALPHA_SCALAR * bkgrnd_mean_f

    b_im, a_im = butter_default(4, 40, imaging_freq)
    if not is_excluded:
        roi_mean_f_corrected_filt = matlab_filter(b_im, a_im, roi_mean_f_corrected)
    else:
        roi_mean_f_corrected_filt = None

    start_time_imaging = int(math.floor(start_time * imaging_freq))

    f0_trial = np.nan
    f0_filt_trial = np.nan
    if not is_excluded:
        pre_end = min(start_time_imaging, roi_mean_f_corrected.size)
        pre_end = max(pre_end, 1)
        pre_trace = roi_mean_f_corrected[:pre_end]
        lpre = pre_trace.size
        win_frames = max(2, matlab_round(f0_win_ms / 1000 * imaging_freq))
        w = min(win_frames, lpre)
        if lpre >= 2 and w >= 2 and lpre >= w:
            n_win = lpre - w + 1
            win_vars = np.array([
                np.var(pre_trace[sw : sw + w], ddof=0)  # var(x,1) -> N norm
                for sw in range(n_win)
            ])
            # MATLAB min ignores NaN (all-NaN -> index 1); np.argmin would
            # pick the first NaN window (reachable when the corrected trace
            # has NaN frames under laserArtifactMcSecondSweepForDff).
            if np.all(np.isnan(win_vars)):
                iw = 0
            else:
                iw = int(np.nanargmin(win_vars))  # 0-based
            f0_trial = float(np.mean(pre_trace[iw : iw + w]))
            pre_filt = roi_mean_f_corrected_filt[:pre_end]
            f0_filt_trial = float(np.mean(pre_filt[iw : iw + w]))
        elif lpre >= 1:
            f0_trial = float(np.mean(pre_trace))
            f0_filt_trial = float(np.mean(roi_mean_f_corrected_filt[:pre_end]))

    L = lholo(ipi, n_pulses, pre_stim_window, post_stim_window, imaging_freq)
    records = []
    for hh in range(len(holo_seq_this_trial)):
        fst = first_stim_vec[hh]
        i_holo_lo = math.floor((fst - pre_stim_window / 1000) * imaging_freq)
        i_holo_hi = math.ceil((fst - pre_stim_window / 1000) * imaging_freq) + \
            math.ceil((ipi * n_pulses + (pre_stim_window + post_stim_window))
                      / 1000 * imaging_freq)
        holo_id = int(holo_seq_this_trial[hh])

        if is_excluded:
            records.append({"holo_id": holo_id, "f0": np.nan,
                            "dff": np.full(L, np.nan),
                            "filtdff": np.full(L, np.nan)})
            continue

        seg = roi_mean_f_corrected[i_holo_lo - 1 : i_holo_hi]
        dff = (seg - f0_trial) / f0_trial
        if str(up_or_down) == "2":
            dff = -dff
        seg_f = roi_mean_f_corrected_filt[i_holo_lo - 1 : i_holo_hi]
        dff_f = (seg_f - f0_filt_trial) / f0_filt_trial
        filtdff = -dff_f if str(up_or_down) == "2" else dff_f

        records.append({"holo_id": holo_id, "f0": f0_trial,
                        "dff": dff, "filtdff": filtdff})

    return records, roi_mean_f_corrected, roi_mean_f_corrected_filt


def extract_roi_mean_f(image_stack, roi_rows, roi_cols, use_bad_rows,
                       bad_row_mask=None):
    """Dispatch to the correct F path (gotcha #3)."""
    if use_bad_rows:
        return roi_mean_per_frame_exclude_bad_rows(
            image_stack, roi_rows, roi_cols, bad_row_mask
        )
    return roi_mean_per_frame_crossproduct(image_stack, roi_rows, roi_cols)


def finalize_holo_matrices(cell_lists, n_conds, n_holos, L):
    """Stack accumulated per-trial columns into (Lholo x nTrials) matrices."""
    out = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            cols = cell_lists[cc][hh]
            if len(cols) == 0:
                out[cc][hh] = np.zeros((L, 0))
            else:
                out[cc][hh] = np.column_stack([np.asarray(c).ravel() for c in cols])
    return out


def holo_means_and_ci(holo_all, filt_holo_all, n_conds, n_holos, up_or_down,
                      confidence_level=0.95):
    """Stage G (MATLAB 1264-1323). Returns means + CIs per cell.

    IMAGING sign flip (opposite ephys): '2' -> [lo,hi], '1' -> [-lo,-hi].
    """
    mean_ = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    filt_mean = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    ci = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    filt_ci = [[None] * int(n_holos[cc]) for cc in range(n_conds)]

    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            mat = holo_all[cc][hh]
            fmat = filt_holo_all[cc][hh]
            mean_[cc][hh] = np.nanmean(mat, axis=1)
            filt_mean[cc][hh] = np.nanmean(fmat, axis=1)

            n_cols = mat.shape[1] if mat.ndim == 2 else 0
            fn_cols = fmat.shape[1] if fmat.ndim == 2 else 0
            means = np.nanmean(mat, axis=1)
            fmeans = np.nanmean(fmat, axis=1)
            sem = std_n1_omitnan(mat, axis=1) / np.sqrt(n_cols) if n_cols else means * np.nan
            fsem = std_n1_omitnan(fmat, axis=1) / np.sqrt(fn_cols) if fn_cols else fmeans * np.nan
            t_score = tinv((1 + confidence_level) / 2, n_cols - 1)
            ft_score = tinv((1 + confidence_level) / 2, fn_cols - 1)
            lower = means - t_score * sem
            upper = means + t_score * sem
            flower = fmeans - ft_score * fsem
            fupper = fmeans + ft_score * fsem
            if str(up_or_down) == "2":
                ci[cc][hh] = np.column_stack([lower, upper])
                filt_ci[cc][hh] = np.column_stack([flower, fupper])
            else:
                ci[cc][hh] = np.column_stack([-lower, -upper])
                filt_ci[cc][hh] = np.column_stack([-flower, -fupper])

    return {
        "mean": mean_, "filt_mean": filt_mean,
        "ci": ci, "filt_ci": filt_ci,
    }
