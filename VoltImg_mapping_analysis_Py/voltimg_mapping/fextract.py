"""ROI fluorescence extraction.

Gotcha #3 -- the two F-extraction paths:

  * DEFAULT (laserArtifactMcSecondSweepForDff == False):
      MATLAB ``imageStack(roiX, roiY, :)`` with *vector* row and column indices
      is a full CROSS-PRODUCT: it selects the sub-block of ALL rows in roiX
      crossed with ALL cols in roiY -- i.e. the bounding RECTANGLE spanned by the
      ROI's row-set x col-set, NOT the exact ROI pixels. The per-frame value is
      then mean(mean(block)). This is the path the canonical pipeline uses.

  * EXACT PIXELS (laserArtifactMcSecondSweepForDff == True):
      VoltImg_roiMeanPerFrameExcludeBadRows uses sub2ind -> exact ROI pixels.
      (See artifact.roi_mean_per_frame_exclude_bad_rows.)

Both are reproduced; select by the flag in the pipeline.
"""

from __future__ import annotations

import numpy as np


def roi_mean_per_frame_crossproduct(
    image_stack: np.ndarray,
    roi_rows: np.ndarray,
    roi_cols: np.ndarray,
) -> np.ndarray:
    """DEFAULT path: MATLAB ``imageStack(roiX, roiY, :)`` cross-product mean.

    Reproduces:
        rawWholeRoiF = imageStack(roiX, roiY, :);   % (numel(roiX) x numel(roiY) x T)
        for ff: roiMeanF(ff) = mean(mean(rawWholeRoiF(:,:,ff)));

    roi_rows/roi_cols are 0-based coordinate arrays. Returns (numFrames,).

    NOTE: MATLAB's ``mean(mean(block))`` averages over the *unique* index values
    taken as sets, but MATLAB vector indexing keeps duplicates and order. Since
    ROI coordinate arrays here come from ``find`` (each linear index once), the
    row-set and col-set passed to cross-product indexing may contain repeated
    values (e.g. two ROI pixels share a row). MATLAB reproduces those repeats in
    the rectangle. We replicate that exactly via np.ix_ on the raw (possibly
    duplicated) index vectors -- do NOT deduplicate.
    """
    roi_rows = np.asarray(roi_rows).ravel()
    roi_cols = np.asarray(roi_cols).ravel()
    if roi_rows.size == 0:
        # Matches MATLAB: mean of an empty block -> NaN per frame.
        return np.full(image_stack.shape[2], np.nan)

    # np.ix_ builds the (rows x cols) open mesh -> full cross-product block,
    # preserving duplicates and order exactly like MATLAB A(rows, cols, :).
    block = image_stack[np.ix_(roi_rows, roi_cols)].astype(np.float64)
    # block shape: (len(rows), len(cols), T). mean over first two dims per frame.
    # MATLAB mean(mean(X)) with no 'omitnan' -> a NaN anywhere makes NaN. Use
    # plain mean to match (non-bad-row default path has no NaNs anyway).
    return block.mean(axis=(0, 1))
