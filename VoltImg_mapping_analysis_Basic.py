#!/usr/bin/env python3
"""
Voltage Imaging Mapping Analysis — Python port (basic setup section).

Parallel variant (future): parfor over NoRMCorre pass 2 and dF/F trials.
F0 for all holograms in a trial/cell is shared: within the last
commonEarlyF0BaselineMs before the first pulse of the first hologram, a rolling
window (commonEarlyF0RollingWinMs) with minimum variance defines F0.

Pipeline overview:
  1) Per-trial NoRMCorre motion correction of raw image stacks
  2) maxDvStack from motion-corrected trials
  3) Hand-drawn rough ROIs
  4) Per-trial fine ROIs inside rough ROIs
  5) F, F0, dF, dF/F0 from motion-corrected stacks and per-trial fine ROIs

This module covers MATLAB lines 1–225: load files, setup, and raw TIFF channel detection.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tkinter import Tk, filedialog

import numpy as np
import tifffile
from scipy.io import loadmat


# ---------------------------------------------------------------------------
# Defaults (mirror commented / active paths in the MATLAB script)
# ---------------------------------------------------------------------------
DEFAULT_EPHYS_DIR = (
    "/Volumes/phoenixinthesky/Masato/Voltage Imaging Data_Phoenix/"
    "SliceMapping/ASAP7y Original Slice Experiment/DAQ Ephys Data/"
)
DEFAULT_IMAGING_DIR = (
    "/Users/masatosadahiro/Documents/Data/Voltage Imaging/"
    "Voltage Imaging/Slice Mapping/"
)
DEFAULT_NORMCORRE_PATH = r"C:\Users\lamia\OneDrive\Documents\MATLAB\NoRMCorre-master"
DEFAULT_SAVE_DIR = r"D:\Data\Voltage Imaging\voltMapping\Analysis Results"


# --- Laser row artifact (applied to raw stacks before global template + NoRMCorre) ---
USE_LASER_ROW_ARTIFACT_FILTER = False
LASER_ARTIFACT_GATE_COL_FIRST = 130
LASER_ARTIFACT_GATE_COL_LAST = 382
LASER_ARTIFACT_THRESH_MODE = "mad"  # 'fixed' | 'mad' | 'percentile'
LASER_ARTIFACT_THRESH_PARAM = 5
LASER_ARTIFACT_MC_MODE = "fill_for_mc"  # 'fill_for_mc' | 'nan'
MC_USE_GATE_COLUMNS_ONLY = True
LASER_ARTIFACT_MC_SECOND_SWEEP_FOR_DFF = False

# --- Parallel execution ---
USE_PARALLEL_MC_PASS2 = False
USE_PARALLEL_DFF_TRIALS = False


def select_directory(title: str, initial_dir: str | None = None) -> str:
    """MATLAB uigetdir equivalent."""
    root = Tk()
    root.withdraw()
    root.update()
    folder = filedialog.askdirectory(
        title=title,
        initialdir=initial_dir if initial_dir and os.path.isdir(initial_dir) else None,
    )
    root.destroy()
    if not folder:
        raise SystemExit("No folder selected.")
    return folder


def matlab_dir_entries(folder: str) -> list[dict]:
    """Return dir()-like entries sorted the way MATLAB dir() does (by name)."""
    entries = []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        entries.append(
            {
                "name": name,
                "folder": folder,
                "isdir": os.path.isdir(full),
            }
        )
    return entries


def load_latest_mat_variable(folder: str) -> tuple[object, str]:
    """
    Load the last entry in folder (MATLAB: load(ephysFileDir(end).name)).
    Returns (loaded_struct, filename).
    """
    entries = matlab_dir_entries(folder)
    if not entries:
        raise FileNotFoundError(f"No files found in {folder}")

    last = entries[-1]
    mat_path = os.path.join(last["folder"], last["name"])
    data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    if "ExpStruct" not in data:
        raise KeyError(f"'ExpStruct' not found in {last['name']}")

    return data["ExpStruct"], last["name"]


def list_tiff_indices(folder: str, file_type: str = ".tif") -> tuple[list[str], list[int]]:
    """
    Step 2a: collect TIFF filenames and 1-based indices (MATLAB imagesIndex).
    """
    entries = matlab_dir_entries(folder)
    file_names: list[str | None] = [None] * len(entries)
    for ii, entry in enumerate(entries):
        name = entry["name"]
        if (
            not entry["isdir"]
            and not name.startswith(".")
            and name.endswith(file_type)
        ):
            file_names[ii] = name

    # 1-based indices of non-empty slots (MATLAB imagesIndex)
    images_index = [ii + 1 for ii, name in enumerate(file_names) if name]
    file_names_clean = [name for name in file_names if name]
    return file_names_clean, images_index


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        return 1.0 if x.size else 0.0
    corr = np.corrcoef(x, y)
    return float(corr[0, 1]) if corr.size >= 4 else 1.0


def detect_raw_tiff_n_channels(tiff_path: str) -> tuple[int, dict[str, float]]:
    """
    Auto-detect whether raw TIFF stacks are single-channel or 2-color interleaved.
    Mirrors MATLAB logic (lines 110–206).
    """
    with tifffile.TiffFile(tiff_path) as tif:
        n_dirs = len(tif.pages)

    if n_dirs < 4:
        return 1, {}

    n_sample = min(120, n_dirs)
    frame_means = np.zeros(n_sample, dtype=np.float32)
    odd_mean_img: np.ndarray | None = None
    even_mean_img: np.ndarray | None = None
    n_odd = 0
    n_even = 0

    with tifffile.TiffFile(tiff_path) as tif:
        for pp in range(n_sample):
            frame = tif.pages[pp].asarray().astype(np.float32)
            frame_means[pp] = frame.mean()
            if pp % 2 == 0:
                if odd_mean_img is None:
                    odd_mean_img = np.zeros_like(frame, dtype=np.float32)
                odd_mean_img += frame
                n_odd += 1
            else:
                if even_mean_img is None:
                    even_mean_img = np.zeros_like(frame, dtype=np.float32)
                even_mean_img += frame
                n_even += 1

    assert odd_mean_img is not None and even_mean_img is not None
    odd_mean_img = odd_mean_img / max(1, n_odd)
    even_mean_img = even_mean_img / max(1, n_even)

    odd_even_img_corr = _safe_corr(odd_mean_img.ravel(), even_mean_img.ravel())

    if frame_means.size >= 3:
        lag1_corr = _safe_corr(frame_means[:-1], frame_means[1:])
        lag2_corr = _safe_corr(frame_means[:-2], frame_means[2:])
        alt_step_diff = float(np.mean(np.abs(np.diff(frame_means))))
        same_chan_diff = float(np.mean(np.abs(frame_means[2:] - frame_means[:-2])))
    else:
        lag1_corr = 0.0
        lag2_corr = 0.0
        alt_step_diff = 0.0
        same_chan_diff = 0.0

    is_interleaved = (odd_even_img_corr < 0.90) and (
        (lag2_corr > lag1_corr + 0.10)
        or (alt_step_diff > 1.15 * max(same_chan_diff, np.finfo(float).eps))
    )
    raw_img_n_channels = 2 if is_interleaved else 1

    metrics = {
        "odd_even_img_corr": odd_even_img_corr,
        "lag1_corr": lag1_corr,
        "lag2_corr": lag2_corr,
    }
    return raw_img_n_channels, metrics


def _unique_nonzero(values) -> np.ndarray:
    """MATLAB unique(nonzeros(x))."""
    arr = np.asarray(values).ravel()
    arr = arr[arr != 0]
    return np.unique(arr)


def _set_attr(obj, name: str, value) -> None:
    setattr(obj, name, value)


def main() -> None:
    # Step 1: Read the ephys file
    ephys_file_path = select_directory(
        "Select DAQ ephys data folder",
        initial_dir=DEFAULT_EPHYS_DIR,
    )
    exp_struct, ephys_file_name = load_latest_mat_variable(ephys_file_path)
    print(ephys_file_name)

    # Step 2: Imaging folder
    imgs_file_path = select_directory(
        "Select imaging data folder",
        initial_dir=DEFAULT_IMAGING_DIR,
    )
    img_entries = matlab_dir_entries(imgs_file_path)
    print(img_entries[-1]["name"])

    _, images_index = list_tiff_indices(imgs_file_path)
    if not images_index:
        raise RuntimeError("No TIFF files found in selected imaging folder.")

    # Step 3: NoRMCorre path (MATLAB addpath; kept for later sections)
    normcorre_path = DEFAULT_NORMCORRE_PATH
    if not os.path.isdir(normcorre_path):
        print(f"Note: NoRMCorre path not found on this machine: {normcorre_path}")

    # Step 4: Setup struct
    volt_mapping = exp_struct

    # If the experiment has a second patch electrode, ExpStruct2 is in the same .mat
    mat_path = os.path.join(ephys_file_path, ephys_file_name)
    mat_data = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    if "ExpStruct2" in mat_data:
        volt_mapping.ExpStruct2 = mat_data["ExpStruct2"]

    # Stimulation properties
    imaging_freq = volt_mapping.sampleFreq
    fs = volt_mapping.daqParams.Fs
    trial_time = volt_mapping.daqParams.maxSweepLengthSec
    n_trials = len(np.atleast_1d(volt_mapping.trialCond))
    powers = volt_mapping.outParams.power
    n_conds = len(np.atleast_1d(volt_mapping.outParams.sequence))
    n_holos = np.atleast_1d(volt_mapping.holoStimParams.nHolos).copy()
    n_holos[0] = np.max(n_holos)  # hack for 0-holo conditions (0 mW trials)

    pulse_durs = _unique_nonzero(volt_mapping.outParams.pulseDur)
    n_pulses = _unique_nonzero(volt_mapping.outParams.nPulses)
    ipi = _unique_nonzero(volt_mapping.outParams.ipi)
    next_holo_delay = _unique_nonzero(volt_mapping.holoStimParams.nextHoloDelay)
    start_time = volt_mapping.holoStimParams.startTime / 1000.0

    # Step 5: GEVI type and ephys availability
    up_or_down = input("1 for upward GEVI, 2 for downward GEVI: ").strip()
    e_phys_avail = input("1 if ephys readout avail, 2 if none: ").strip()

    # Auto-detect raw TIFF channel mode from first indexed stack
    first_idx = images_index[0] - 1  # convert 1-based MATLAB index
    test_tiff_name = img_entries[first_idx]["name"]
    test_tiff_path = os.path.join(imgs_file_path, test_tiff_name)

    raw_img_n_channels, ch_metrics = detect_raw_tiff_n_channels(test_tiff_path)
    if ch_metrics:
        print(
            f"Auto-detected raw TIFF channel mode from {test_tiff_name}: "
            f"rawImgNChannels = {raw_img_n_channels} "
            f"(odd-even mean image corr = {ch_metrics['odd_even_img_corr']:.3f}, "
            f"lag1 = {ch_metrics['lag1_corr']:.3f}, "
            f"lag2 = {ch_metrics['lag2_corr']:.3f})."
        )

    # Step 6: Save directory (later overwritten near save step)
    mouse_id = exp_struct.mouseID
    directory = DEFAULT_SAVE_DIR
    file_name = f"voltMapping {mouse_id}.mat"

    _set_attr(volt_mapping, "imagesIndex", np.array(images_index, dtype=np.int64))
    _set_attr(volt_mapping, "imagingFreq", imaging_freq)
    _set_attr(volt_mapping, "UpOrDown", up_or_down)
    _set_attr(volt_mapping, "rawImgNChannels", raw_img_n_channels)
    _set_attr(volt_mapping, "ephysFilePath", ephys_file_path)
    _set_attr(volt_mapping, "ImgsFilePath", imgs_file_path)
    _set_attr(volt_mapping, "nConds", n_conds)
    _set_attr(volt_mapping, "nHolos", n_holos)
    _set_attr(volt_mapping, "pulseDurs", pulse_durs)
    _set_attr(volt_mapping, "nPulses", n_pulses)
    _set_attr(volt_mapping, "ipi", ipi)
    _set_attr(volt_mapping, "nextHoloDelay", next_holo_delay)

    # Expose config flags on volt_mapping for downstream sections
    _set_attr(volt_mapping, "useLaserRowArtifactFilter", USE_LASER_ROW_ARTIFACT_FILTER)
    _set_attr(volt_mapping, "laserArtifactGateColFirst", LASER_ARTIFACT_GATE_COL_FIRST)
    _set_attr(volt_mapping, "laserArtifactGateColLast", LASER_ARTIFACT_GATE_COL_LAST)
    _set_attr(volt_mapping, "laserArtifactThreshMode", LASER_ARTIFACT_THRESH_MODE)
    _set_attr(volt_mapping, "laserArtifactThreshParam", LASER_ARTIFACT_THRESH_PARAM)
    _set_attr(volt_mapping, "laserArtifactMcMode", LASER_ARTIFACT_MC_MODE)
    _set_attr(volt_mapping, "mcUseGateColumnsOnly", MC_USE_GATE_COLUMNS_ONLY)
    _set_attr(
        volt_mapping,
        "laserArtifactMcSecondSweepForDff",
        LASER_ARTIFACT_MC_SECOND_SWEEP_FOR_DFF,
    )
    _set_attr(volt_mapping, "useParallelMcPass2", USE_PARALLEL_MC_PASS2)
    _set_attr(volt_mapping, "useParallelDffTrials", USE_PARALLEL_DFF_TRIALS)
    _set_attr(volt_mapping, "normcorrePath", normcorre_path)
    _set_attr(volt_mapping, "directory", directory)
    _set_attr(volt_mapping, "fileName", file_name)

    # Variables used later in the pipeline (MATLAB workspace equivalents)
    Fs = fs
    trialTime = trial_time
    nTrials = n_trials
    powers = powers
    startTime = start_time

    print(f"\nSetup complete for mouse {mouse_id}.")
    print(f"  Trials: {n_trials}, conditions: {n_conds}, holos: {n_holos}")
    print(f"  Imaging freq: {imaging_freq} Hz, DAQ Fs: {fs} Hz")
    print(f"  Save target (placeholder): {os.path.join(directory, file_name)}")

    return volt_mapping


if __name__ == "__main__":
    main()
