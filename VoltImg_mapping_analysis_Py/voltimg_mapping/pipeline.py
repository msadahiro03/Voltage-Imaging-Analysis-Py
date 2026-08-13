"""Top-level orchestration of the voltage-imaging mapping analysis.

Stitches the ported stages together. The interactive MATLAB script is replaced
by explicit function arguments (rough ROIs, UpOrDown, ePhysAvail, nCells, etc.)
so the pipeline is scriptable and testable.

Trial alignment assumption (unchanged from MATLAB): trial index tt (1-based)
indexes ephys (mapping_inputs[tt-1], trial_cond[tt-1]) AND imaging (the tt-th
TIFF in alphabetical order). Keep both in the same order.

The dead interactive plotting / QC blocks (MATLAB ~1439-2380) are intentionally
not ported -- they do not affect numeric outputs (gotcha #10 / #16).
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List

import numpy as np

from . import dff as dff_mod
from . import roi as roi_mod
from .matlab_compat import matlab_find_2d


# ---------------------------------------------------------------------------
# Stage D helper: maxDvStack + grand-mean reference
# ---------------------------------------------------------------------------


def build_maxdv_stack(
    mc_stacks_iter,          # yields (tt_1based, mc_stack_hwt) for ALL trials
    n_trials,
    exclude_trials,          # set of 1-based tt
    max_dv_trial_mask,       # 0-based bool array length n_trials
    max_dv_frame_cap,        # int
    image_shape,             # (H, W)
):
    """MATLAB 668-679. maxDvStack (H, W, nTrials); NaN plane for excluded or
    trials outside the sampling mask. Returns (max_dv_stack, mean_fluor)."""
    H, W = image_shape
    max_dv_stack = np.full((H, W, n_trials), np.nan)
    for tt, mc_stack in mc_stacks_iter:
        n_cap = min(mc_stack.shape[2], max_dv_frame_cap)
        mean_plane = np.mean(mc_stack[:, :, :n_cap].astype(np.float64), axis=2)
        if (tt in exclude_trials) or (not max_dv_trial_mask[tt - 1]):
            max_dv_stack[:, :, tt - 1] = np.nan
        else:
            max_dv_stack[:, :, tt - 1] = mean_plane
    mean_fluor = np.nanmean(max_dv_stack, axis=2)
    return max_dv_stack, mean_fluor


# ---------------------------------------------------------------------------
# Stage E helper: global fine ROIs from rough ROIs
# ---------------------------------------------------------------------------


def compute_global_rois(mean_fluor_max_dv, max_dv_stack, rough_rois):
    """rough_rois: list per cell of (rows, cols) 0-based tuples.

    Returns dict with 'roi_global', 'bkgrnd_global' lists per cell of (rows,cols).
    """
    roi_global = []
    bkgrnd_global = []
    for rough_rows, rough_cols in rough_rois:
        rr, rc, br, bc = roi_mod.compute_global_fine_roi(
            mean_fluor_max_dv, max_dv_stack, rough_rows, rough_cols
        )
        roi_global.append((rr, rc))
        bkgrnd_global.append((br, bc))
    return {"roi_global": roi_global, "bkgrnd_global": bkgrnd_global}


# ---------------------------------------------------------------------------
# Stage F: single-trial dF/F (the per-trial body of run_dff, factored out)
# ---------------------------------------------------------------------------


def process_one_trial(
    image_stack,                # (H, W, numFrames), one trial's MC stack
    n_cells,
    rough_rois,                 # list per cell (rows, cols)
    bkgrnd_global,              # list per cell (rows, cols) global ring fallback
    cc_1based,                  # condition (1-based) of this trial
    sequence_this_trial,        # this trial's sequence vector, or None/empty
    zero_dummy_sequence,
    first_stim_times,           # list per cond of vectors (s)
    is_excluded,                # bool
    up_or_down,                 # '1' | '2'
    imaging_freq, ipi, n_pulses, pre_stim_window, post_stim_window, start_time,
    use_bad_rows=False,
    bad_row_mask=None,          # (H, numFrames) or None
    common_f0=False,
    f0_win_ms=50,
    fine_rois_override=None,     # optional list per cell (rows, cols) to reuse
):
    """Compute one trial's per-cell dF/F, ROIs, and per-holo records.

    This is the exact per-trial body of :func:`run_dff` (MATLAB 960-1247),
    factored out so it can be driven either in a batch loop (``run_dff``) or one
    trial at a time as TIFFs arrive (the ``live`` subpackage). Given identical
    inputs it produces identical outputs; the caller owns accumulation.

    Exclusion policy note: for excluded trials this still returns the extracted
    ``roi_mean_f`` / ``bkgrnd_mean_f`` / ``roi_corr`` (as MATLAB computes them);
    it is the caller (``run_dff`` / the accumulator) that NaNs those columns.

    Returns a dict::

        {
          "num_frames": int,
          "cc": int,                       # echoed 1-based condition
          "holo_seq_this_trial": ndarray,  # 1-based holo IDs
          "per_cell": [ {                  # one entry per cell nn
              "records": [ {holo_id, f0, dff, filtdff}, ... ],
              "roi_mean_f": ndarray,       # (numFrames,)
              "bkgrnd_mean_f": ndarray,    # (numFrames,)
              "roi_corr": ndarray,         # roiMeanFCorrected (numFrames,)
              "fine_roi": (rows, cols),
              "bkgrnd_roi": (rows, cols),
          }, ... ]
        }
    """
    H, W, num_frames = image_stack.shape

    mean_img = np.mean(image_stack.astype(np.float32), axis=2)
    mean_img_double = roi_mod.im2double(mean_img.astype(np.float32))

    # Pass 1: per-trial fine ROI per cell (unless caller supplied them).
    fine_rois = [None] * n_cells
    for nn in range(n_cells):
        if fine_rois_override is not None and fine_rois_override[nn] is not None:
            fine_rois[nn] = fine_rois_override[nn]
        else:
            rough_rows, rough_cols = rough_rois[nn]
            fine_rois[nn] = roi_mod.compute_trial_fine_roi(
                mean_img, rough_rows, rough_cols
            )

    # Union ROI mask across all cells (neuropil exclusion).
    all_trial_roi_mask = np.zeros((H, W), dtype=bool)
    for nn in range(n_cells):
        fr, fc = fine_rois[nn]
        if np.asarray(fr).size > 0:
            all_trial_roi_mask[fr, fc] = True

    cc = int(cc_1based)

    # sequenceThisTrial fallback + holo IDs (once per trial).
    seq = sequence_this_trial
    if seq is None or np.asarray(seq).size == 0:
        seq = zero_dummy_sequence
    seq = np.asarray(seq).ravel()
    _, first_idx = np.unique(seq, return_index=True)
    uniq_stable = seq[np.sort(first_idx)]
    holo_seq_this_trial = (uniq_stable - uniq_stable.min() + 1).astype(int)

    fst_vec = first_stim_times[cc - 1]
    if fst_vec is None or np.asarray(fst_vec).size == 0:
        fst_vec = first_stim_times[1]
    fst_vec = np.asarray(fst_vec, dtype=float).ravel()

    # Pass 2: F extraction + neuropil + dF/F per cell.
    per_cell = []
    for nn in range(n_cells):
        fine_rows, fine_cols = fine_rois[nn]

        roi_mean_f = dff_mod.extract_roi_mean_f(
            image_stack, fine_rows, fine_cols, use_bad_rows, bad_row_mask
        )

        gbr, gbc = bkgrnd_global[nn]
        bk_rows, bk_cols = roi_mod.compute_trial_neuropil_ring(
            mean_img_double, fine_rows, fine_cols, all_trial_roi_mask, gbr, gbc,
        )

        bkgrnd_mean_f = dff_mod.extract_roi_mean_f(
            image_stack, bk_rows, bk_cols, use_bad_rows, bad_row_mask
        )

        if common_f0:
            records, roi_corr, _ = dff_mod.compute_trial_dff_common_f0(
                roi_mean_f, bkgrnd_mean_f, is_excluded, cc,
                holo_seq_this_trial, fst_vec, imaging_freq, ipi, n_pulses,
                pre_stim_window, post_stim_window, up_or_down, start_time,
                f0_win_ms,
            )
        else:
            records, roi_corr, _ = dff_mod.compute_trial_dff(
                roi_mean_f, bkgrnd_mean_f, is_excluded, cc,
                holo_seq_this_trial, fst_vec, imaging_freq, ipi, n_pulses,
                pre_stim_window, post_stim_window, up_or_down,
            )

        per_cell.append({
            "records": records,
            "roi_mean_f": roi_mean_f,
            "bkgrnd_mean_f": bkgrnd_mean_f,
            "roi_corr": roi_corr,
            "fine_roi": (fine_rows, fine_cols),
            "bkgrnd_roi": (bk_rows, bk_cols),
        })

    return {
        "num_frames": num_frames,
        "cc": cc,
        "holo_seq_this_trial": holo_seq_this_trial,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Stage F+G: dF/F over all trials/cells from MC stacks
# ---------------------------------------------------------------------------


def run_dff(
    mc_stack_loader: Callable[[int], np.ndarray],  # tt_1based -> (H, W, T)
    n_trials,
    n_cells,
    n_conds,
    n_holos,
    trial_cond,                 # length n_trials, 1-based
    rough_rois,                 # list per cell (rows, cols)
    bkgrnd_global,              # list per cell (rows, cols) global ring fallback
    first_stim_times,           # list per cond of vectors (s)
    sequence_this_trial,        # list length n_trials
    zero_dummy_sequence,
    exclude_trials,             # set of 1-based
    up_or_down,                 # '1' | '2'
    imaging_freq, ipi, n_pulses, pre_stim_window, post_stim_window, start_time,
    use_bad_rows=False,
    bad_row_mask_loader=None,   # tt_1based -> (H, numFrames) or None
    common_f0=False,
    f0_win_ms=50,
) -> Dict:
    """Ports MATLAB 960-1262 (dF/F loop) then 1264-1323 (means/CIs).

    Returns a per-cell dict of results mirroring the analysisStruct fields.
    """
    L = dff_mod.lholo(ipi, n_pulses, pre_stim_window, post_stim_window,
                      imaging_freq)
    acc = dff_mod.init_cell_accumulators(n_cells, n_conds, n_holos)

    # Per-cell full-trial matrices (numFrames x nTrials); filled lazily once we
    # know numFrames from the first stack.
    roi_mean_f_mat = [None] * n_cells
    bkgrnd_mean_f_mat = [None] * n_cells
    roi_corr_mat = [None] * n_cells
    sub_scalar = [np.full(n_trials, np.nan) for _ in range(n_cells)]

    # Per-trial fine ROIs: fine_rois[nn][tt-1] = (rows, cols).
    fine_rois = [[None] * n_trials for _ in range(n_cells)]
    bkgrnd_rois_trial = [[None] * n_trials for _ in range(n_cells)]

    for tt in range(1, n_trials + 1):
        image_stack = mc_stack_loader(tt)  # (H, W, numFrames)
        H, W, num_frames = image_stack.shape

        if roi_mean_f_mat[0] is None:
            for nn in range(n_cells):
                roi_mean_f_mat[nn] = np.full((num_frames, n_trials), np.nan)
                bkgrnd_mean_f_mat[nn] = np.full((num_frames, n_trials), np.nan)
                roi_corr_mat[nn] = np.full((num_frames, n_trials), np.nan)

        bad_row_mask = None
        if use_bad_rows and bad_row_mask_loader is not None:
            bad_row_mask = bad_row_mask_loader(tt)

        is_excluded = tt in exclude_trials
        cc = int(trial_cond[tt - 1])

        # Persist the sequenceThisTrial fallback into the shared list, matching
        # the original in-place behavior (some trials arrive empty).
        if (sequence_this_trial[tt - 1] is None
                or np.asarray(sequence_this_trial[tt - 1]).size == 0):
            sequence_this_trial[tt - 1] = zero_dummy_sequence

        result = process_one_trial(
            image_stack, n_cells, rough_rois, bkgrnd_global, cc,
            sequence_this_trial[tt - 1], zero_dummy_sequence, first_stim_times,
            is_excluded, up_or_down, imaging_freq, ipi, n_pulses,
            pre_stim_window, post_stim_window, start_time,
            use_bad_rows=use_bad_rows, bad_row_mask=bad_row_mask,
            common_f0=common_f0, f0_win_ms=f0_win_ms,
        )

        for nn in range(n_cells):
            cell = result["per_cell"][nn]
            fine_rois[nn][tt - 1] = cell["fine_roi"]
            bkgrnd_rois_trial[nn][tt - 1] = cell["bkgrnd_roi"]

            # Append per-holo records.
            for rec in cell["records"]:
                hid = rec["holo_id"]  # 1-based holo ID
                acc["f0_all"][nn][cc - 1][hid - 1].append(rec["f0"])
                acc["holo_all"][nn][cc - 1][hid - 1].append(rec["dff"])
                acc["filt_holo_all"][nn][cc - 1][hid - 1].append(rec["filtdff"])

            # Full-trial matrices (NaN for excluded).
            if is_excluded:
                roi_mean_f_mat[nn][:, tt - 1] = np.nan
                bkgrnd_mean_f_mat[nn][:, tt - 1] = np.nan
                sub_scalar[nn][tt - 1] = np.nan
                roi_corr_mat[nn][:, tt - 1] = np.nan
            else:
                roi_mean_f_mat[nn][:, tt - 1] = cell["roi_mean_f"]
                bkgrnd_mean_f_mat[nn][:, tt - 1] = cell["bkgrnd_mean_f"]
                sub_scalar[nn][tt - 1] = dff_mod.ALPHA_SCALAR
                roi_corr_mat[nn][:, tt - 1] = cell["roi_corr"]

    # Finalize matrices + means/CIs per cell.
    per_cell = []
    for nn in range(n_cells):
        holo_all = dff_mod.finalize_holo_matrices(
            acc["holo_all"][nn], n_conds, n_holos, L
        )
        filt_holo_all = dff_mod.finalize_holo_matrices(
            acc["filt_holo_all"][nn], n_conds, n_holos, L
        )
        # F0 rows (1 x nTrialsForHolo).
        f0_all = [[np.asarray(acc["f0_all"][nn][cc][hh], dtype=float)
                   for hh in range(int(n_holos[cc]))] for cc in range(n_conds)]

        mci = dff_mod.holo_means_and_ci(
            holo_all, filt_holo_all, n_conds, n_holos, up_or_down
        )

        per_cell.append({
            "holoSortedImagingAllTrials": holo_all,
            "filtHoloSortedImagingAllTrials": filt_holo_all,
            "F0AllTrials": f0_all,
            "holoSortedImagingMean": mci["mean"],
            "filtHoloSortedImagingMean": mci["filt_mean"],
            "CIDffAllConds": mci["ci"],
            "filtCIDffAllConds": mci["filt_ci"],
            "roiMeanF": roi_mean_f_mat[nn],
            "bkgrndMeanF": bkgrnd_mean_f_mat[nn],
            "roiMeanFCorrected": roi_corr_mat[nn],
            "subScalar": sub_scalar[nn],
            "fineRois": fine_rois[nn],
            "bkgrndRoisTrial": bkgrnd_rois_trial[nn],
        })

    return {"per_cell": per_cell, "Lholo": L}
