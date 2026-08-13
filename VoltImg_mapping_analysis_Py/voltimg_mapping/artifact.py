"""Laser-row-artifact helpers (ports of the local MATLAB helper functions).

Ports of:
  * VoltImg_mapping_removeArtifact_v2.m
  * VoltImg_laserRowArtifact_badRowMaskStack.m
  * VoltImg_applyLaserRowArtifactToStack.m
  * VoltImg_roiMeanPerFrameExcludeBadRows.m

All of these are OFF by default in the canonical pipeline
(useLaserRowArtifactFilter=false, laserArtifactMcSecondSweepForDff=false) but are
ported for completeness and switched on by the same flags as MATLAB.

Stack convention: image stacks are (H, W, T) as in MATLAB (frame index last).
"""

from __future__ import annotations

import numpy as np

from .matlab_compat import mad1, prctile


def remove_artifact_v2(
    curr_frame: np.ndarray,
    gate_col_first: int,
    gate_col_last: int,
    thresh_mode: str,
    thresh_param: float,
    apply_nan: bool = False,
):
    """Port of VoltImg_mapping_removeArtifact_v2.

    Parameters use 1-based MATLAB gate columns (gate_col_first/last), converted
    internally.

    Returns (clean_frame, bad_rows, line_var, thresh_used).
    """
    H, W = curr_frame.shape
    # MATLAB: g1 = max(1, min(gcf,gcl)); g2 = min(W, max(gcf,gcl)) -- 1-based.
    g1 = max(1, min(gate_col_first, gate_col_last))
    g2 = min(W, max(gate_col_first, gate_col_last))
    if g2 < g1:
        raise ValueError("Invalid gate column range.")

    # single(currFrame(:, g1:g2)) -- 1-based inclusive -> [g1-1 : g2] 0-based.
    stat_area = curr_frame[:, g1 - 1 : g2].astype(np.float32)
    # var(statArea, 0, 2): variance across columns (axis=1), N-1 (ddof=1).
    line_var = np.var(stat_area, axis=1, ddof=1)

    mode = thresh_mode.lower()
    if mode == "fixed":
        thresh_used = thresh_param
        bad_rows = line_var > thresh_used
    elif mode == "mad":
        k = thresh_param
        med_lv = np.nanmedian(line_var)
        mad_lv = mad1(line_var)
        thresh_used = med_lv + k * mad_lv
        bad_rows = line_var > thresh_used
    elif mode == "percentile":
        thresh_used = prctile(line_var, thresh_param)
        bad_rows = line_var > thresh_used
    else:
        raise ValueError("threshMode must be fixed, mad, or percentile.")

    clean_frame = curr_frame
    if apply_nan:
        clean_frame = curr_frame.astype(np.float32, copy=True)
        clean_frame[bad_rows, :] = np.nan

    return clean_frame, bad_rows.astype(bool), line_var, thresh_used


def bad_row_mask_stack(
    image_stack: np.ndarray,
    gate_col_first: int,
    gate_col_last: int,
    thresh_mode: str,
    thresh_param: float,
) -> np.ndarray:
    """Port of VoltImg_laserRowArtifact_badRowMaskStack. Returns (H, T) bool."""
    H, W, n_keep = image_stack.shape
    bad_row_mask = np.zeros((H, n_keep), dtype=bool)
    for ki in range(n_keep):
        fr = image_stack[:, :, ki].astype(np.float32)
        _, bad_rows, _, _ = remove_artifact_v2(
            fr, gate_col_first, gate_col_last, thresh_mode, thresh_param, False
        )
        bad_row_mask[:, ki] = bad_rows
    return bad_row_mask


def apply_laser_row_artifact_to_stack(
    image_stack: np.ndarray,
    gate_col_first: int,
    gate_col_last: int,
    thresh_mode: str,
    thresh_param: float,
    mc_mode: str = "fill_for_mc",
):
    """Port of VoltImg_applyLaserRowArtifactToStack.

    Returns (image_stack_out, stack_stats) where stack_stats is a dict with
    'threshUsed' and 'nBadRows' per-frame arrays.
    """
    mc_mode = mc_mode.lower()
    if mc_mode not in ("fill_for_mc", "nan"):
        raise ValueError("mcMode must be fill_for_mc or nan.")

    H, W, n_keep = image_stack.shape
    if mc_mode == "nan":
        image_stack_out = np.zeros((H, W, n_keep), dtype=np.float32)
    else:
        image_stack_out = np.zeros((H, W, n_keep), dtype=image_stack.dtype)

    thresh_used_arr = np.zeros(n_keep)
    n_bad_rows_arr = np.zeros(n_keep)

    for ki in range(n_keep):
        fr = image_stack[:, :, ki]
        apply_nan = mc_mode == "nan"
        fr_detect = fr.astype(np.float32) if apply_nan else fr
        fr_out, bad_rows, _, tu = remove_artifact_v2(
            fr_detect, gate_col_first, gate_col_last, thresh_mode,
            thresh_param, apply_nan,
        )
        thresh_used_arr[ki] = float(tu)
        n_bad_rows_arr[ki] = int(np.sum(bad_rows))

        if mc_mode == "fill_for_mc" and np.any(bad_rows):
            good_mask = ~bad_rows
            if not np.any(good_mask):
                mv = np.nanmedian(fr)
            else:
                mv = np.nanmedian(fr[good_mask, :])
            if np.isnan(mv) or np.isinf(mv):
                mv = 0
            fr_out = fr.copy()
            mv = np.array(mv).astype(fr_out.dtype)
            fr_out[bad_rows, :] = mv
        elif mc_mode == "nan":
            pass  # already float32 with NaN rows
        else:
            fr_out = fr

        image_stack_out[:, :, ki] = fr_out

    stack_stats = {"threshUsed": thresh_used_arr, "nBadRows": n_bad_rows_arr}
    return image_stack_out, stack_stats


def roi_mean_per_frame_exclude_bad_rows(
    image_stack: np.ndarray,
    roi_rows: np.ndarray,
    roi_cols: np.ndarray,
    bad_row_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Port of VoltImg_roiMeanPerFrameExcludeBadRows.

    EXACT-PIXEL path (sub2ind). Used only when laserArtifactMcSecondSweepForDff
    is True (gotcha #3 contrast). roi_rows/roi_cols are 0-based coordinate
    arrays. Returns (numFrames,) column of per-frame ROI means.
    """
    H, W, num_frames = image_stack.shape
    roi_mean_f = np.zeros(num_frames)

    roi_rows = np.asarray(roi_rows).ravel()
    roi_cols = np.asarray(roi_cols).ravel()

    if roi_rows.size == 0:
        roi_mean_f[:] = np.nan
        return roi_mean_f

    use_mask = bad_row_mask is not None and bad_row_mask.size > 0
    if use_mask and bad_row_mask.shape != (H, num_frames):
        raise ValueError("badRowMask must be H x numFrames matching imageStack.")

    for ff in range(num_frames):
        slice_ = image_stack[:, :, ff]
        pv = slice_[roi_rows, roi_cols].astype(np.float64)
        if use_mask:
            br = bad_row_mask[roi_rows, ff]
            pv = pv[~br]
        if pv.size == 0:
            roi_mean_f[ff] = np.nan
        else:
            roi_mean_f[ff] = np.nanmean(pv)

    return roi_mean_f
