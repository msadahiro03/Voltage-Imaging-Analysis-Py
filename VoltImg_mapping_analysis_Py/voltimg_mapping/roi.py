"""ROI computation: global fine ROI, per-trial fine ROI, neuropil rings.

Coordinate convention (gotcha #2): all ROI coordinate outputs are 0-based
``(rows, cols)`` integer arrays, from ``matlab_find_2d`` (column-major order).
Index images as ``img[rows, cols]``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .fibermetric import fibermetric
from .matlab_compat import (
    bwareaopen,
    im2double,
    imdilate_disk,
    imgaussfilt,
    matlab_find_2d,
    prctile,
)

_EPS = np.finfo(float).eps


def _gaussian_percentile_norm(roi_img_double: np.ndarray):
    """Shared: mask>0 pixels, imgaussfilt(0.7), percentile-normalize [10,99].

    Returns (roi_filt, roi_norm). ``roi_img_double`` must already be im2double.
    Matches MATLAB lines 781-792 / 1043-1054.
    """
    roi_pixels = roi_img_double > 0
    roi_filt = imgaussfilt(roi_img_double, 0.7)
    vals = roi_filt[roi_pixels]
    if vals.size > 0:
        lo = prctile(vals, 10)
        hi = prctile(vals, 99)
        roi_norm = (roi_filt - lo) / max(hi - lo, _EPS)
        roi_norm = np.clip(roi_norm, 0, 1)
    else:
        roi_norm = roi_filt
    return roi_filt, roi_norm


def compute_global_fine_roi(
    mean_fluor_max_dv_stack: np.ndarray,
    max_dv_stack: np.ndarray,
    rough_rows: np.ndarray,
    rough_cols: np.ndarray,
    struct_sensitivity: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Global fine ROI + neuropil ring per cell (MATLAB lines 765-854).

    Returns (roi_rows, roi_cols, bkgrnd_rows, bkgrnd_cols), all 0-based.

    The global fine ROI is built on ``roiMeanMaxDvStack`` -- an image that is
    zero everywhere except the rough-ROI pixels, where each pixel holds the
    NaN-omitting mean across the maxDvStack third dimension (MATLAB line 774).
    """
    H, W = mean_fluor_max_dv_stack.shape

    # roiMeanMaxDvStack: zeros with rough-ROI pixels set to per-pixel omitnan
    # mean across trials (third dim of maxDvStack).
    roi_mean_max_dv = np.zeros((H, W))
    for r, c in zip(rough_rows, rough_cols):
        roi_mean_max_dv[r, c] = np.nanmean(max_dv_stack[r, c, :])

    roi_double = im2double(roi_mean_max_dv)
    _, roi_norm = _gaussian_percentile_norm(roi_double)

    # Global path applies fibermetric to the WHOLE (masked-to-ROI) image.
    ridge = fibermetric(roi_norm, struct_sensitivity)
    vals_r = ridge[ridge != 0]
    thr = prctile(vals_r, 50) if vals_r.size > 0 else 0.0
    ridge_reduced = ridge.copy()
    ridge_reduced[ridge_reduced < thr] = 0
    ridge_reduced = (ridge_reduced > 0)

    roi_rows, roi_cols = matlab_find_2d(ridge_reduced)
    if roi_rows.size == 0:
        roi_rows = np.asarray(rough_rows).copy()
        roi_cols = np.asarray(rough_cols).copy()

    # Background ring from the *continuous* ridge image (MATLAB line 828-847).
    bkgrnd_rows, bkgrnd_cols = _neuropil_ring(
        ridge_mask_for_dilate=ridge,
        image_double=roi_double,
        inner_buffer=2,
        ring_width=3,
        min_area=50,
        exclude_mask=None,
    )
    return roi_rows, roi_cols, bkgrnd_rows, bkgrnd_cols


def compute_trial_fine_roi(
    mean_img_this_trial: np.ndarray,
    rough_rows: np.ndarray,
    rough_cols: np.ndarray,
    struct_sensitivity: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial fine ROI for one cell (MATLAB lines 1035-1096).

    Uses im2double of the WHOLE trial mean image (line 1043 -- the masked
    ``roiMeanThisTrial`` is dead code, gotcha #16), gaussian + percentile norm,
    then fibermetric restricted to a padded bbox around the rough-ROI pixels,
    zeroed outside the rough-ROI mask, threshold prctile(nonzeros, 60).

    Returns (fine_rows, fine_cols), 0-based. Falls back to rough ROI if empty.
    """
    H, W = mean_img_this_trial.shape

    roi_double = im2double(mean_img_this_trial.astype(np.float32))
    _, roi_norm = _gaussian_percentile_norm(roi_double)

    # Rough-ROI mask.
    roi_mask = np.zeros((H, W), dtype=bool)
    roi_mask[rough_rows, rough_cols] = True

    ridge = np.zeros_like(roi_norm)
    rows_m, cols_m = np.nonzero(roi_mask)
    if rows_m.size > 0:
        # Tight bbox padded by 2 (MATLAB clamps to image bounds). 1-based ->
        # 0-based: rmin = max(min(rows)-2, 1) in MATLAB; here rows are already
        # 0-based, so rmin0 = max(min(rows0)-2, 0), rmax0 = min(max(rows0)+2, H-1).
        rmin = max(rows_m.min() - 2, 0)
        rmax = min(rows_m.max() + 2, H - 1)
        cmin = max(cols_m.min() - 2, 0)
        cmax = min(cols_m.max() + 2, W - 1)
        sub_img = roi_norm[rmin : rmax + 1, cmin : cmax + 1]
        sub_mask = roi_mask[rmin : rmax + 1, cmin : cmax + 1]
        sub_ridge = fibermetric(sub_img, struct_sensitivity)
        sub_ridge[~sub_mask] = 0
        ridge[rmin : rmax + 1, cmin : cmax + 1] = sub_ridge

    vals_r = ridge[ridge != 0]
    thr = prctile(vals_r, 60) if vals_r.size > 0 else 0.0
    ridge_reduced = ridge.copy()
    ridge_reduced[ridge_reduced < thr] = 0
    ridge_reduced = (ridge_reduced > 0)

    fine_rows, fine_cols = matlab_find_2d(ridge_reduced)
    if fine_rows.size == 0:
        fine_rows = np.asarray(rough_rows).copy()
        fine_cols = np.asarray(rough_cols).copy()
    return fine_rows, fine_cols


def _neuropil_ring(
    ridge_mask_for_dilate,
    image_double,
    inner_buffer,
    ring_width,
    min_area,
    exclude_mask=None,
):
    """Shared neuropil ring construction (dilate/ring/brightcut/areaopen).

    ``ridge_mask_for_dilate`` is the mask that is dilated to form the ring; in
    the global path this is the continuous fibermetric ridge image (MATLAB
    dilates the non-binary ridge, which binary_dilation treats as nonzero==True),
    in the per-trial path it is the binary fine-ROI mask.

    Returns (bkgrnd_rows, bkgrnd_cols), 0-based.
    """
    seed = ridge_mask_for_dilate != 0
    inner_select = imdilate_disk(seed, inner_buffer)
    outer_select = imdilate_disk(seed, inner_buffer + ring_width)
    background_ring = outer_select & ~inner_select
    if exclude_mask is not None:
        background_ring = background_ring & ~exclude_mask

    vals_bk = image_double[background_ring]
    if vals_bk.size > 0:
        bright_cut = prctile(vals_bk, 95)
        ring_clean = background_ring & (image_double <= bright_cut)
    else:
        ring_clean = background_ring

    ring_clean = bwareaopen(ring_clean, 7)
    if np.count_nonzero(ring_clean) < min_area:
        ring_clean = background_ring

    return matlab_find_2d(ring_clean)


def compute_trial_neuropil_ring(
    mean_img_this_trial_double: np.ndarray,
    fine_rows: np.ndarray,
    fine_cols: np.ndarray,
    all_trial_roi_mask: np.ndarray,
    global_bkgrnd_rows: np.ndarray,
    global_bkgrnd_cols: np.ndarray,
    inner_buffer: int = 2,
    ring_width: int = 3,
    min_area: int = 50,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-trial, per-cell neuropil ring (MATLAB lines 1128-1166).

    Ring excludes all cells' ROI pixels (all_trial_roi_mask). Fallbacks:
      1) bright-cut + area-open ring;
      2) if < min_area -> raw background ring;
      3) if still empty -> global ring minus all_trial_roi_mask.
    Returns (bkgrnd_rows, bkgrnd_cols), 0-based.
    """
    H, W = mean_img_this_trial_double.shape

    roi_mask_this_cell = np.zeros((H, W), dtype=bool)
    roi_mask_this_cell[fine_rows, fine_cols] = True

    inner_select = imdilate_disk(roi_mask_this_cell, inner_buffer)
    outer_select = imdilate_disk(roi_mask_this_cell, inner_buffer + ring_width)
    background_ring = outer_select & ~inner_select
    background_ring = background_ring & ~all_trial_roi_mask

    vals_bk = mean_img_this_trial_double[background_ring]
    if vals_bk.size > 0:
        bright_cut = prctile(vals_bk, 95)
        ring_clean = background_ring & (mean_img_this_trial_double <= bright_cut)
    else:
        ring_clean = background_ring

    ring_clean = bwareaopen(ring_clean, 7)
    if np.count_nonzero(ring_clean) < min_area:
        ring_clean = background_ring

    if np.count_nonzero(ring_clean) < 1:
        ring_global = np.zeros((H, W), dtype=bool)
        if np.asarray(global_bkgrnd_rows).size > 0:
            ring_global[global_bkgrnd_rows, global_bkgrnd_cols] = True
        ring_clean = ring_global & ~all_trial_roi_mask

    return matlab_find_2d(ring_clean)
