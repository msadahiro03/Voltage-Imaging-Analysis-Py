"""Parity layers 1-2 (version-independent): raw ROI F, neuropil F, corrected F.

Given ROI pixel sets loaded from the reference .mat (so fibermetric geometry is
neutralized) and the exact MC TIFFs, the port's cross-product extraction MUST
reproduce the reference roiMeanF / bkgrndMeanF to float tolerance, and
roiMeanFCorrected = roiMeanF - alpha*bkgrndMeanF with alpha from the reference.

Usage: python layer12_F.py [nSampleTrials]
"""
from __future__ import annotations

import os
import sys
import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voltimg_mapping.fextract import roi_mean_per_frame_crossproduct
from parity.ref_loader import Ref, mc_tiff_path


def load_stack_hwt(trial_1based: int) -> np.ndarray:
    """Read MC TIFF -> (H, W, T) float64 (pages are (H,W))."""
    with tifffile.TiffFile(mc_tiff_path(trial_1based)) as t:
        arr = t.asarray()  # (T, H, W)
    return np.transpose(arr, (1, 2, 0)).astype(np.float64)


def summarize(name, py, ref):
    """Return dict of abs/rel error stats between two 1-D arrays."""
    py = np.asarray(py, float); ref = np.asarray(ref, float)
    both = np.isfinite(py) & np.isfinite(ref)
    d = np.abs(py[both] - ref[both])
    denom = np.maximum(np.abs(ref[both]), 1e-12)
    rel = d / denom
    nanmatch = np.array_equal(np.isnan(py), np.isnan(ref))
    return dict(name=name, n=int(both.sum()), max_abs=float(d.max()) if d.size else 0.0,
                max_rel=float(rel.max()) if rel.size else 0.0,
                nan_layout_match=bool(nanmatch), len_py=py.size, len_ref=ref.size)


def main():
    n_sample = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    r = Ref()
    n_cells = int(r.scalar('nCells'))
    n_trials = int(r.scalar('nTrials'))
    alpha_ref = r.scalar('alphaScalar')
    print(f"nCells={n_cells} nTrials={n_trials} alphaScalar(ref)={alpha_ref}")
    print(f"excludeTrials empty: {r.is_empty('excludeTrials')}  UpOrDown={r.char('UpOrDown')!r}")

    # sample trials spread across the full range (1-based)
    trials = sorted(set(np.linspace(1, n_trials, n_sample).astype(int).tolist()))
    print(f"Sampling {len(trials)} trials: {trials}\n")

    # preload reference trace matrices per cell (nTrials x numFrames)
    ref_roiF = {nn: r.trace_matrix('roiMeanF', nn) for nn in range(1, n_cells + 1)}
    ref_bkg = {nn: r.trace_matrix('bkgrndMeanF', nn) for nn in range(1, n_cells + 1)}
    ref_corr = {nn: r.trace_matrix('roiMeanFCorrected', nn) for nn in range(1, n_cells + 1)}

    worst = {'roiMeanF': 0.0, 'bkgrndMeanF': 0.0, 'roiMeanFCorrected': 0.0}
    rows = []
    for tt in trials:
        stack = load_stack_hwt(tt)             # (H,W,T)
        tt0 = tt - 1
        for nn in range(1, n_cells + 1):
            fx = r.roi_per_trial('fineRoiXAllCells', nn - 1, tt0)
            fy = r.roi_per_trial('fineRoiYAllCells', nn - 1, tt0)
            bx = r.roi_per_trial('bkgrndRoiXAllCells_trial', nn - 1, tt0)
            by = r.roi_per_trial('bkgrndRoiYAllCells_trial', nn - 1, tt0)
            if fx is None or fy is None:
                continue
            # 1-based -> 0-based (X=row, Y=col)
            roiF_py = roi_mean_per_frame_crossproduct(stack, fx - 1, fy - 1)
            s_roi = summarize('roiMeanF', roiF_py, ref_roiF[nn][tt0])
            worst['roiMeanF'] = max(worst['roiMeanF'], s_roi['max_abs'])
            if bx is not None and by is not None:
                bkg_py = roi_mean_per_frame_crossproduct(stack, bx - 1, by - 1)
                s_bkg = summarize('bkgrndMeanF', bkg_py, ref_bkg[nn][tt0])
                corr_py = roiF_py - alpha_ref * bkg_py
                s_cor = summarize('roiMeanFCorrected', corr_py, ref_corr[nn][tt0])
                worst['bkgrndMeanF'] = max(worst['bkgrndMeanF'], s_bkg['max_abs'])
                worst['roiMeanFCorrected'] = max(worst['roiMeanFCorrected'], s_cor['max_abs'])
                rows.append((tt, nn, s_roi['max_abs'], s_bkg['max_abs'], s_cor['max_abs'],
                             s_roi['len_py'], s_roi['len_ref']))
            else:
                rows.append((tt, nn, s_roi['max_abs'], None, None,
                             s_roi['len_py'], s_roi['len_ref']))

    print(f"{'trial':>6} {'cell':>4} {'roiMeanF':>12} {'bkgrndMeanF':>12} {'corrected':>12}  lens")
    for tt, nn, a, b, c, lp, lr in rows:
        bs = f"{b:12.3e}" if b is not None else f"{'--':>12}"
        cs = f"{c:12.3e}" if c is not None else f"{'--':>12}"
        flag = '' if lp == lr else f'  LEN py={lp} ref={lr}'
        print(f"{tt:6d} {nn:4d} {a:12.3e} {bs} {cs}{flag}")

    print("\n=== WORST max-abs error across all sampled (trial,cell) ===")
    for k, v in worst.items():
        print(f"  {k:22s} {v:.6e}")
    r.close()


if __name__ == '__main__':
    main()
