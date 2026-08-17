# Python port: VoltImg MultiCell MCfineROI TrialSpecROIandNeuropil (laserRowArtifact)

Faithful Python translation of
`VoltImg_mapping_analysis_MultiCell_newDFF_021226_MCfineROI_TrialSpecROIandNeuropil_laserRowArtifact.m`
and its local helpers. Goal: numerical equivalence with the MATLAB original.

> Note on git: the repo `.gitignore` is `*` + `!*.m` + `!*/`, so these `.py`
> files are **not tracked by git** by default. They live under `python_port/`.
> Add `!*.py` (and `!*.md`) to `.gitignore` if you want them committed.

## Install

```
pip install numpy scipy tifffile
```
`tifffile` is only needed for the TIFF I/O and channel-detection stages
(`tiff_io.py`). The pure-numeric stages need only NumPy + SciPy.

## Module map (MATLAB stage -> file)

| MATLAB stage | lines | Python |
|---|---|---|
| A: channel detection, TIFF probing | 20-211 | `tiff_io.detect_channels`, `tiff_io.read_stack` |
| B: ephys baseline + trial exclusion | 213-236 | `ephys.baseline_and_exclude` |
| C: ephys hologram sorting | 238-431 | `ephys.sort_holograms`, `ephys.ephys_confidence_intervals` |
| D: motion correction + maxDvStack | 433-694 | `motion_correction.build_global_template`, `pipeline.build_maxdv_stack`, `sampling.max_dv_stack_sampling_plan` |
| E: rough ROIs + global fine ROIs | 732-910 | `auto_roi.detect_rough_rois_cellpose` (automatic) or external masks; `pipeline.compute_global_rois`, `roi.compute_global_fine_roi` |
| F: dF/F with per-trial fine ROIs | 912-1262 | `pipeline.run_dff`, `roi.compute_trial_fine_roi`, `roi.compute_trial_neuropil_ring`, `fextract`, `dff.compute_trial_dff` |
| G: per-holo means + CIs | 1264-1323 | `dff.holo_means_and_ci` |
| H: trial excluder | 1335 (external) | `trial_excluder.run_trial_excluder` |
| I: reorganize per cell + save | 1337-1437, 2381-2435 | assembled by the caller from `run_dff` + `run_trial_excluder` outputs |
| commonF0 companion | separate script | `dff.compute_trial_dff_common_f0` (+ `common_f0=True` in `run_dff`) |
| local helpers | helper .m files | `artifact.py`, `sampling.py` |

Dead code intentionally not ported: robustfit `alphaScalar` (overwritten to
0.85), masked `roiMeanThisTrial` in the fine-ROI block (overwritten by the
whole-image `im2double`), and the commented-out plotting/QC blocks
(~lines 1439-2380). None affect numeric outputs.

## How to run (typical order)

1. `expstruct.load_expstruct(mat_path)` — load ephys ExpStruct, derive scalars.
2. `ephys.baseline_and_exclude(...)` then `ephys.sort_holograms(...)`.
3. **Motion correction** — see "External libraries" below. Produce
   `Motion_Corrected_Tiffs/<rawName>_mc.tif` per trial.
4. `pipeline.build_maxdv_stack(...)` over the MC stacks -> `meanFluorMaxDvStack`.
5. Rough ROIs on `meanFluorMaxDvStack` — two options:
   - **Automatic (2026-08)**: `auto_roi.detect_rough_rois_cellpose(mean_img)` —
     Cellpose segmentation, bit-identical to the MATLAB
     `auto_roi/voltimg_autoRoi_cellpose.m` path (shared wrapper + venv, see
     "Automatic rough ROIs" below).
   - Manual: draw in MATLAB and supply as 0-based `(rows, cols)` arrays.
   Then `pipeline.compute_global_rois(...)`.
6. `pipeline.run_dff(mc_stack_loader=..., rough_rois=..., bkgrnd_global=...)`.
7. `trial_excluder.run_trial_excluder(...)` per cell.

## Automatic rough ROIs (Cellpose, added 2026-08)

`voltimg_mapping/auto_roi.py` ports the MATLAB auto-ROI orchestrator
(`Voltage-Imaging-Analysis-ML/auto_roi/voltimg_autoRoi_cellpose.m`). Both call
the SAME `cellpose_wrapper.py` in the SAME `.venv_cellpose` (python 3.11 +
cellpose 4 `cpsam`; built by `auto_roi/setup_cellpose_env.sh` in the MATLAB
repo) via subprocess — this port's own 3.14 interpreter cannot host torch.
Pipeline: sanitize -> uint16 exchange TIFF (MATLAB-identical rounding) ->
Cellpose -> quality filters (border/area/greedy-separation/maxCells) ->
centroid (row, col) ordering -> `imdilate_disk` (strel-exact; default 3 px,
tuned to hand-drawn margins) -> `matlab_find_2d` -> 0-based `(rows, cols)`.

Cross-language parity verified 2026-08-14 on the sample FOV: exchange TIFF,
label masks, accepted set/order, and final rough pixel lists (including
column-major order) all identical between MATLAB and Python; 17/20 masks
accepted, all 8 hand-drawn ground-truth cells recovered.

Live runner: `--roi-format cellpose --rois <mean_image.tif|.npy>`
(+ optional `--cellpose-cfg overrides.json`); rough ROIs are detected up front
and the global neuropil rings are bootstrapped from the same mean image via
`compute_global_rois` with a 1-slice stack. Detection artifacts (exchange
files, `autoRoi_report.json`, QC overlay if matplotlib is installed, log) land
in `<out>/AutoROI/`. Zero surviving cells raises `auto_roi.AutoRoiError` with
per-label reject reasons.

## Coordinate & indexing conventions (the load-bearing ones)

- **1-based -> 0-based** everywhere; fractional colon `1:x` yields `floor(x)`
  elements (`matlab_compat.colon_count`). Trial index `tt` is kept **1-based**
  in the ephys/pipeline public API to preserve the ephys<->imaging alignment
  assumption; internal arrays are 0-based.
- **`[X,Y]=find` returns X=row, Y=col.** All ROI coordinates are stored as
  0-based `(rows, cols)` and images indexed `img[rows, cols]`
  (`matlab_compat.matlab_find_2d`, column-major order to match MATLAB `find`).
- **`imageStack(roiX, roiY, :)` is a row×col cross-product (rectangle)**, the
  DEFAULT F-extraction path (`fextract.roi_mean_per_frame_crossproduct`). The
  exact-pixel `sub2ind` path (`artifact.roi_mean_per_frame_exclude_bad_rows`) is
  used only when `laserArtifactMcSecondSweepForDff=True`. Both are ported and
  switched by `use_bad_rows` in `run_dff` (gotcha #3).
- **filtfilt (ephys, zero-phase) vs filter (imaging, causal)** are kept distinct
  (`matlab_compat.matlab_filtfilt` vs `matlab_compat.matlab_filter`) (gotcha #4).
- **std N-1** (`std_n1`, default) vs **var N** (`var_n`, the commonF0 F0-search
  `var(x,1)`) (gotcha #5). Reductions use nan-aware variants where MATLAB uses
  `'omitnan'`.
- **`alphaScalar` is hard 0.85** (`dff.ALPHA_SCALAR`); robustfit is dead code.
- **`im2double` divides uint16 by 65535** before percentile-norm + fibermetric
  (`matlab_compat.im2double`).
- Empty/edge fallbacks (0 mW / sham: empty `sequenceThisTrial` ->
  `zeroDummySequence`; empty `firstStimTimes{cc}` -> `{1,2}`; empty ROI ->
  rough ROI; empty ring -> global ring) are all reproduced.

## Per-holo column length (subtle, deliberate)

`Lholo = ceil((ipi*nPulses + preStim + postStim)/1000 * imagingFreq) + 2` is the
length MATLAB uses for **excluded** trials' NaN columns. Non-excluded columns
come from `roiMeanFCorrected(iHoloLo:iHoloHi)`, whose length is
`ceil(a) - floor(a) + ceil(b) + 1` with `a = (firstStimTime - preStim/1000)*imagingFreq`.
When `a` is non-integer (the normal case) this equals `Lholo`, so all columns
concatenate. If `a` happens to be an exact integer, the windowed length is
`Lholo - 1` and MATLAB's `horzcat` of a length-(Lholo-1) column with a
length-Lholo NaN column would **error**. The port reproduces the exact index
arithmetic and does **not** pad, matching MATLAB behavior (including its
fragility). In practice real stimulus times make `a` non-integer.

## External libraries / hardest pieces

### NoRMCorre motion correction (Stage D)
No faithful NumPy equivalent. Two options:
1. **Recommended for parity:** run the MATLAB NoRMCorre step (params:
   `d1=H, d2=W, bin_width=15, max_shift=4, us_fac=50, init_batch=1`, rigid,
   one global template) to produce `_mc.tif`, then run the Python dF/F stage on
   those exact stacks. `motion_correction.build_global_template` ports the
   Pass-1 template accumulation; `pipeline.build_maxdv_stack` ports maxDvStack.
2. **Python-native:** CaImAn's `caiman.motion_correction` is the same NoRMCorre
   algorithm but will not be bit-identical (different FFT/subpixel code). Inject
   it and validate against MATLAB.

### fibermetric (Frangi ridge filter) — now EXACT (gotcha #7, resolved 2026-08)
`fibermetric.py` reproduces MATLAB `fibermetric` exactly (max abs diff ~1e-16
vs R2025b on ridge/random test images, borders included, bright and dark
polarity, single- and multi-scale). Verified structure: `sigma = thickness/6`,
pre-smooth `imgaussfilt(V, sigma, 'FilterSize', 2*ceil(3*sigma)+1)` (from
fibermetric.m source); the C++ builtin's Hessian was identified empirically as
the 5-tap stencil `[1 0 -2 0 1]/4` (cross term `[1 0 -1]/2` per axis) with
half-sample symmetric borders, sigma^2 scale normalization, Frangi beta=0.5
(MATLAB exposes no blobness knob). Earlier ports used sigma=thickness/(2*sqrt(3))
and a global-max rescale — both wrong.

### strel('disk', r) shape (gotcha #10, corrected 2026-08)
`matlab_compat.strel_disk` hard-codes `getnhood(strel('disk', r))` output
verified against MATLAB R2025b for r = 2, 3, 5: r=2 is the 13-px Euclidean
diamond (MATLAB skips decomposition below r=3), r=3 is the 5x5 solid square,
r=5 is a 9x9 69-px mask. Note MATLAB's decomposed disks for r>=3 are SMALLER
than Euclidean disks (axial half-extent r-1), so `strel_disk` now RAISES for
unverified r>=3 instead of silently substituting a Euclidean disk. If you
change `innerBuffer`/`ringWidth`, print the MATLAB neighborhood and add it.

## Known parity caveats (data flow — read before end-to-end comparison)

1. **maxDvStack input scaling.** MATLAB builds the maxDvStack planes (line 669)
   from the in-memory **pre-rescale float** NoRMCorre output, BEFORE the
   per-trial `uint16((x-min)/(max-min)*65535)` rescale that produces the saved
   `_mc.tif`. Feeding `pipeline.build_maxdv_stack` the saved `_mc.tif` applies
   a different affine per trial, so `meanFluorMaxDvStack` (and the ROI stage
   downstream) will NOT match MATLAB. For parity, feed it the unscaled float
   MC stacks (export them from MATLAB, or save an unscaled companion).
2. **Trial ordering / file discovery.** MATLAB (lines 39-46) pairs ephys trial
   `tt` with the tt-th `.tif` in **alphabetical `dir` order** (skipping
   dotfiles). `live/watcher.py` sorts by the parsed trial number instead. The
   two agree only when trial numbers in filenames are zero-padded. With
   unpadded names (`_1, _2, ..., _10`) MATLAB's order is 1,10,11,...,2,20,...
   — reproduce THAT order (or zero-pad filenames) when comparing.
3. **Bad-row mask fallback.** With `laserArtifactMcSecondSweepForDff = true`,
   MATLAB loads `<raw>_mc_badRows.mat` and, if absent, COMPUTES the mask
   in-line (lines 998-1013); on a size mismatch it warns and proceeds with no
   exclusion. `run_dff` only calls the `bad_row_mask_loader` callback: with
   `use_bad_rows=True` and no loader it silently excludes nothing, and a
   size-mismatched mask raises. Wire `artifact.bad_row_mask_stack` into your
   loader to reproduce the compute fallback (compute it on the FLOAT MC stack
   like MATLAB line 622, not the rescaled TIFF, if using 'fixed' thresholds).
4. **Trial excluder wiring.** `trial_excluder.run_trial_excluder` is a faithful
   port but no in-repo caller invokes it; `runner.snapshot_to_matlab` writes
   only the 11 pre-exclusion fields and none of the 8 `std*`/`excl*` fields
   MATLAB saves per cell. Callers must run step 7 themselves and rename
   `"std"`/`"std_filt"` to the MATLAB field names when writing a `.mat`.

## MATLAB-semantics fixes applied 2026-08 (verified against R2025b)

`strel_disk` masks (see gotcha #10 above), `fibermetric` (exact, see gotcha
#7), `matlab_filtfilt` padlen 12 vs scipy default 15 (0.3 absolute edge error
before the fix), `matlab_round` (half-away-from-zero; Python `round` is
half-to-even — routed into the F0/window-length call sites in `dff.py` and
`ephys.py`), `colon_count` ulp-snap (`1:0.29*10000` has 2900 elements, not
2899), `std_n1`/`std_n1_omitnan`/excluder std returning 0 (not NaN) for a
single (finite) observation, NaN-aware max/min index in the excluder late-peak
test and the commonF0 window search, and MATLAB-style rounding in the uint16
TIFF write and the laser-artifact integer fill value.

## Parity checking

The pure-numeric layer is directly checkable. Recommended approach:
1. From MATLAB, `save` intermediate variables for one experiment:
   `meanFluorMaxDvStack`, `roughRoiXAllCells`/`roughRoiYAllCells`,
   `fineRoiXAllCells`/`fineRoiYAllCells`, `analysisStruct.roiMeanF_cell1`,
   `analysisStruct.roiMeanFCorrected_cell1`,
   `analysisStruct.holoSortedImagingMean_cell1`, and the excluder outputs.
2. In Python, feed the SAME MC stacks and the SAME rough ROIs (loaded from the
   MATLAB `.mat`) into `run_dff`, and diff:
   - `roiMeanF` / `roiMeanFCorrected` (should match to float tolerance if the MC
     stacks and ROI pixel sets are identical),
   - `holoSortedImagingMean` and CIs,
   - trial-excluder masks (which trials/columns got NaN'd).
3. Isolate ROI-geometry differences (fibermetric) by first checking `roiMeanF`
   with ROI pixel sets **loaded from MATLAB** (`fineRoiXAllCells`) rather than
   recomputed — this separates dF/F arithmetic parity from ridge-filter parity.

Non-bit-identical by construction: NoRMCorre (external) and any RNG (none here
since `pickPercentage=1.0` selects all eligible trials deterministically —
gotcha #14). fibermetric is now exact (see above); float roundoff at the
~1e-12 level remains in butter/filtfilt coefficients.
