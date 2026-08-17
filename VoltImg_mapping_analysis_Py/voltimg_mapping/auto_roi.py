"""Automatic rough-ROI detection via Cellpose (port of voltimg_autoRoi_cellpose.m).

Replaces the hand-drawn/externally-supplied rough ROIs with Cellpose
segmentation of the motion-corrected mean image, matching the MATLAB
implementation in ``Voltage-Imaging-Analysis-ML/auto_roi/`` step for step:

1. sanitize NaN/Inf -> finite min, write uint16 TIFF (MATLAB
   ``uint16(rescale(img)*65535)``; uint16() rounds half away from zero),
2. run ``cellpose_wrapper.py`` in the shared ``.venv_cellpose`` via subprocess
   (this port's own interpreter is 3.14, which torch/cellpose do not support),
3. read back the uint16 label mask,
4. quality-filter chain on the tight masks: border exclusion -> area min/max ->
   greedy min-separation (keep by descending mean intensity) -> maxCells cap,
5. deterministic cell ordering by centroid (row, then col),
6. dilate each accepted mask with MATLAB's ``strel('disk', r)`` shape
   (``matlab_compat.imdilate_disk``; r=3 default is a verified preset) to
   emulate the generous hand-drawn margins,
7. emit 0-based ``(rows, cols)`` per cell in MATLAB ``find`` column-major
   order (``matlab_find_2d``) -- the exact ``RoiList`` contract the rest of
   the port consumes.

The detector and its venv are shared with the MATLAB repo (one wrapper, one
model cache) so both pipelines produce identical masks for identical images.
Given the same float mean image, the exchange TIFF written here is
bit-identical to MATLAB's, so Cellpose (deterministic on CPU) returns the same
labels and the final ROIs match pixel for pixel.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tifffile

from .matlab_compat import imdilate_disk, matlab_find_2d

RoiList = List[Tuple[np.ndarray, np.ndarray]]

# Shared detector installed by Voltage-Imaging-Analysis-ML/auto_roi/setup_cellpose_env.sh
_ML_AUTO_ROI = Path(
    "/Users/masatosadahiro/Code/matlab_scripts/Voltage-Imaging-Analysis-ML/auto_roi"
)

DEFAULT_CFG = {
    "python_exe": str(_ML_AUTO_ROI / ".venv_cellpose" / "bin" / "python"),
    "wrapper_path": str(_ML_AUTO_ROI / "cellpose_wrapper.py"),
    "model": "cpsam",
    "diameter": 30.0,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
    "use_gpu": False,
    "min_area_px": 150,
    "max_area_px": 5000,
    "min_separation_px": 15.0,
    "max_cells": None,          # None = no cap (MATLAB Inf)
    "exclude_border_px": 0,
    # 3 px best matches hand-drawn margins (validation sweep 2026-08); also a
    # verified strel_disk preset, so dilation is exactly MATLAB's.
    "dilate_radius_px": 3,
}


class AutoRoiError(RuntimeError):
    """Cellpose auto-ROI failed (wrapper error or zero cells)."""


def _write_exchange_tiff(img: np.ndarray, path: Path) -> np.ndarray:
    """Sanitize and write the uint16 exchange TIFF exactly as MATLAB does."""
    img = np.asarray(img, dtype=np.float64)
    finite = np.isfinite(img)
    if not finite.any():
        raise AutoRoiError("mean image has no finite pixels")
    img = np.where(finite, img, img[finite].min())
    lo, hi = img.min(), img.max()
    scaled = (img - lo) / max(hi - lo, np.finfo(float).tiny) * 65535.0
    # MATLAB uint16() rounds half away from zero; values are >= 0 here.
    u16 = np.floor(scaled + 0.5).astype(np.uint16)
    tifffile.imwrite(str(path), u16)
    return img  # sanitized double image (used for mean intensities)


def detect_rough_rois_cellpose(
    mean_image: np.ndarray,
    cfg: Optional[dict] = None,
    out_dir: str = "AutoROI",
) -> Tuple[RoiList, dict]:
    """Detect rough ROIs on a motion-corrected mean image with Cellpose.

    Parameters
    ----------
    mean_image : (H, W) float array -- ``meanFluorMaxDvStack`` equivalent.
    cfg : optional dict overriding :data:`DEFAULT_CFG` keys.
    out_dir : directory for exchange files, log, and report (created).

    Returns
    -------
    (rough_rois, report) : ``RoiList`` of 0-based (rows, cols) per accepted
    cell in centroid (row, col) order, and a JSON-serializable report dict
    (also written to ``out_dir/autoRoi_report.json``).

    Raises :class:`AutoRoiError` if the wrapper fails or no cell survives the
    filters (report + log are saved first, mirroring the MATLAB behavior).
    """
    t_total = time.time()
    p = dict(DEFAULT_CFG)
    p.update(cfg or {})
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = _write_exchange_tiff(mean_image, out / "input_mean_image.tif")
    H, W = img.shape

    params = {
        "model": p["model"],
        "diameter": p["diameter"],
        "flow_threshold": p["flow_threshold"],
        "cellprob_threshold": p["cellprob_threshold"],
        "use_gpu": bool(p["use_gpu"]),
    }
    (out / "params.json").write_text(json.dumps(params))

    cmd = [
        str(p["python_exe"]), str(p["wrapper_path"]),
        "--image", str(out / "input_mean_image.tif"),
        "--params", str(out / "params.json"),
        "--outdir", str(out),
    ]
    t_cp = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cellpose_elapsed = time.time() - t_cp
    (out / "cellpose_log.txt").write_text(
        f"{' '.join(cmd)}\n\nexit status: {proc.returncode}\n"
        f"elapsed: {cellpose_elapsed:.1f} s\n\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    masks_path = out / "masks.tif"
    if proc.returncode != 0 or not masks_path.exists():
        tail = (proc.stderr or proc.stdout)[-1500:]
        raise AutoRoiError(
            f"Cellpose wrapper failed (exit {proc.returncode}). "
            f"Log: {out / 'cellpose_log.txt'}\n--- output tail ---\n{tail}"
        )

    label_mask = np.asarray(tifffile.imread(str(masks_path)))
    if label_mask.shape != (H, W):
        raise AutoRoiError(
            f"masks.tif shape {label_mask.shape} != image shape {(H, W)}"
        )
    n_raw = int(label_mask.max())

    # ---- per-label stats + filter chain (tight masks, before dilation) ------
    labels = []
    for ll in range(1, n_raw + 1):
        rows, cols = np.nonzero(label_mask == ll)
        rec = {
            "label": ll,
            "areaPx": int(rows.size),
            "accepted": rows.size > 0,
            "rejectReason": "" if rows.size else "empty label",
        }
        if rows.size:
            rec["centroidRow"] = float(rows.mean())
            rec["centroidCol"] = float(cols.mean())
            rec["meanIntensity"] = float(img[rows, cols].mean())
            border = int(p["exclude_border_px"])
            if border > 0 and (
                (rows < border).any() or (rows >= H - border).any()
                or (cols < border).any() or (cols >= W - border).any()
            ):
                rec["accepted"] = False
                rec["rejectReason"] = f"within {border} px of border"
            elif rec["areaPx"] < p["min_area_px"]:
                rec["accepted"] = False
                rec["rejectReason"] = (
                    f"area {rec['areaPx']} < minAreaPx {p['min_area_px']}"
                )
            elif rec["areaPx"] > p["max_area_px"]:
                rec["accepted"] = False
                rec["rejectReason"] = (
                    f"area {rec['areaPx']} > maxAreaPx {p['max_area_px']}"
                )
        labels.append(rec)

    # min separation: greedy keep by descending mean intensity
    survivors = [r for r in labels if r["accepted"]]
    kept_centroids: list = []
    for rec in sorted(survivors, key=lambda r: -r["meanIntensity"]):
        c = np.array([rec["centroidRow"], rec["centroidCol"]])
        if kept_centroids and min(
            np.linalg.norm(c - k) for k in kept_centroids
        ) < p["min_separation_px"]:
            rec["accepted"] = False
            rec["rejectReason"] = (
                f"centroid < minSeparationPx ({p['min_separation_px']:g} px) "
                "from a kept cell"
            )
        else:
            kept_centroids.append(c)

    # maxCells cap: keep top-N by mean intensity
    survivors = [r for r in labels if r["accepted"]]
    if p["max_cells"] is not None and len(survivors) > p["max_cells"]:
        for rec in sorted(survivors, key=lambda r: -r["meanIntensity"])[
            int(p["max_cells"]):
        ]:
            rec["accepted"] = False
            rec["rejectReason"] = f"beyond maxCells cap ({p['max_cells']})"

    # deterministic ordering: centroid row, then column
    accepted = sorted(
        (r for r in labels if r["accepted"]),
        key=lambda r: (r["centroidRow"], r["centroidCol"]),
    )

    # ---- dilate + emit the RoiList contract ---------------------------------
    rough_rois: RoiList = []
    rad = int(p["dilate_radius_px"])
    for rec in accepted:
        mask = label_mask == rec["label"]
        if rad > 0:
            mask = imdilate_disk(mask, rad)
        rough_rois.append(matlab_find_2d(mask))

    report = {
        "paramsUsed": {k: v for k, v in p.items()},
        "nRaw": n_raw,
        "nAccepted": len(accepted),
        "acceptedLabelsInOrder": [r["label"] for r in accepted],
        "labels": labels,
        "cellposeElapsedSec": round(cellpose_elapsed, 2),
        "totalElapsedSec": round(time.time() - t_total, 2),
        "autoRoiDir": str(out),
    }
    cp_json = out / "cellpose_output.json"
    if cp_json.exists():
        try:
            report["cellposeVersion"] = json.loads(cp_json.read_text()).get(
                "cellposeVersion"
            )
        except (json.JSONDecodeError, OSError):
            pass
    (out / "autoRoi_report.json").write_text(json.dumps(report, indent=2))

    _save_qc_overlay(img, label_mask, labels, accepted, rough_rois, p, out)

    if not rough_rois:
        reasons = "\n".join(
            f"  label {r['label']}: {r['rejectReason']}"
            for r in labels if r["rejectReason"]
        )
        if n_raw == 0:
            raise AutoRoiError(
                f"Cellpose found 0 masks. See QC + log in {out}"
            )
        raise AutoRoiError(
            f"All {n_raw} Cellpose masks rejected by quality filters:\n"
            f"{reasons}\nSee QC in {out}"
        )

    print(
        f"AutoROI: {len(accepted)}/{n_raw} Cellpose masks accepted "
        f"({report['totalElapsedSec']:.1f} s total). QC: {out}"
    )
    return rough_rois, report


def _save_qc_overlay(img, label_mask, labels, accepted, rough_rois, p, out):
    """Best-effort QC overlay (matplotlib optional; never raises)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(14, 9))
        ax.imshow(img, cmap="gray")
        for rec in labels:
            if rec["areaPx"] == 0:
                continue
            mask = label_mask == rec["label"]
            ax.contour(mask, levels=[0.5], colors="yellow", linewidths=0.75)
            if not rec["accepted"]:
                ax.contour(mask, levels=[0.5], colors="red", linewidths=1.2)
                ax.text(rec["centroidCol"], rec["centroidRow"],
                        rec["rejectReason"], color="red", fontsize=6,
                        ha="center")
        for nn, (rec, (rows, cols)) in enumerate(zip(accepted, rough_rois), 1):
            mask = np.zeros(img.shape, dtype=bool)
            mask[rows, cols] = True
            ax.contour(mask, levels=[0.5], colors="lime", linewidths=1.5)
            ax.text(rec["centroidCol"], rec["centroidRow"], str(nn),
                    color="lime", fontsize=11, fontweight="bold", ha="center")
        ax.set_title(
            f"AutoROI: {len(labels)} raw -> {len(accepted)} accepted | "
            f"model={p['model']} diam={p['diameter']:g} "
            f"dilate={p['dilate_radius_px']}px "
            f"area=[{p['min_area_px']:g} {p['max_area_px']:g}] "
            f"sep={p['min_separation_px']:g}px"
        )
        fig.savefig(out / "autoRoi_overlay.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        (out / "autoRoi_overlay.SKIPPED.txt").write_text(
            "matplotlib not installed in this environment; overlay skipped.\n"
            "The MATLAB-side QC overlay or the report JSON carry the same "
            "information.\n"
        )
    except Exception as exc:  # noqa: BLE001 - QC must never block detection
        print(f"AutoROI: QC overlay failed (non-fatal): {exc}")


def load_mean_image(path: str) -> np.ndarray:
    """Load a mean image for detection from .tif/.tiff or .npy."""
    lower = path.lower()
    if lower.endswith((".tif", ".tiff")):
        return np.asarray(tifffile.imread(path), dtype=np.float64)
    if lower.endswith(".npy"):
        return np.asarray(np.load(path), dtype=np.float64)
    raise ValueError(f"unsupported mean-image format: {path}")
