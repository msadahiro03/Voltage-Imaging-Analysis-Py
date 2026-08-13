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
| E: rough ROIs + global fine ROIs | 732-910 | `pipeline.compute_global_rois`, `roi.compute_global_fine_roi` |
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
5. Draw rough ROIs on `meanFluorMaxDvStack` (interactive in MATLAB; supply as
   0-based `(rows, cols)` arrays here). Then `pipeline.compute_global_rois(...)`.
6. `pipeline.run_dff(mc_stack_loader=..., rough_rois=..., bkgrnd_global=...)`.
7. `trial_excluder.run_trial_excluder(...)` per cell.

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

### fibermetric (Frangi ridge filter) — highest fidelity risk (gotcha #7)
`fibermetric.py` implements a Frangi-1998 2-D vesselness with MATLAB's exposed
knobs (`StructureSensitivity`, default `BlobnessSensitivity=0.5`, bright
polarity, default thickness range 4:2:14). MathWorks does not document the exact
kernel scales/normalization, so **response values are not bit-identical**.
Downstream, the response is only used through relative percentile thresholds
(`prctile(nonzeros, 50)` global, `prctile(nonzeros, 60)` per-trial) then
binarized, so ROI masks are robust to monotone rescaling — but ridge *geometry*
can still differ. **Validate ROI masks against MATLAB on sample images before
trusting dF/F numbers.**

### strel('disk', r) shape (gotcha #10)
`matlab_compat.strel_disk` hard-codes MATLAB's actual `strel('disk', r)`
neighborhoods for the radii the pipeline uses (2, 3, 5) from
`getnhood(strel('disk',r))` (default N=4 decomposition). Other radii fall back
to a Euclidean disk. If you change `innerBuffer`/`ringWidth`, add the
corresponding MATLAB neighborhood.

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

Non-bit-identical by construction: NoRMCorre (external), fibermetric ridge
geometry, and any RNG (none here since `pickPercentage=1.0` selects all eligible
trials deterministically — gotcha #14).
