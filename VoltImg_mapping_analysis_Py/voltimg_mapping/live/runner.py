"""Live runner: watch folder -> per-trial dF/F -> condition/holo-sorted struct.

Wires :class:`~voltimg_mapping.live.watcher.TrialFileWatcher`,
:func:`voltimg_mapping.pipeline.process_one_trial`, and
:class:`~voltimg_mapping.live.accumulator.LiveMappingAccumulator`, persisting a
rolling snapshot after every processed trial so downstream tools (or a GUI) can
read the latest sorted dF/F while acquisition continues.

Run as a module::

    python -m voltimg_mapping.live.runner \
        --expstruct  /path/ExpStruct.mat \
        --rois       /path/prior_voltMapping.mat \
        --watch      /path/Motion_Corrected_Tiffs \
        --out        /path/live_out \
        --up-or-down 2

or drive it programmatically via :func:`run_live`.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .. import pipeline, tiff_io
from . import context as context_mod
from .accumulator import LiveMappingAccumulator
from .watcher import TrialFileWatcher


@dataclass
class LiveConfig:
    watch_folder: str
    output_dir: str
    pattern: str = "*_mc.tif"
    trial_regex: str = r"_(\d+)_mc\.tif$"
    stable_seconds: float = 3.0
    poll_interval: float = 2.0
    verify_tiff: bool = True
    keep_odd_pages: bool = False       # True for 2-color interleaved stacks
    backfill_existing: bool = True     # process files already present at start
    max_idle_seconds: Optional[float] = None  # stop after this idle gap
    save_mat: bool = False             # also write a .mat snapshot
    snapshot_basename: str = "live_snapshot"
    status_basename: str = "live_status"
    reload_expstruct_on_miss: bool = True  # re-read ExpStruct if metadata absent
    max_defer_retries: int = 30        # per-trial retries waiting for metadata


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, path)


def _atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# .mat export (optional) -- nested cell/struct via numpy object arrays.
# ---------------------------------------------------------------------------


def _to_object_cell(nested):
    """Recursively convert nested python lists -> numpy object arrays so
    scipy.io.savemat writes them as MATLAB cell arrays. Leaves ndarrays/scalars
    as-is."""
    if isinstance(nested, list):
        arr = np.empty(len(nested), dtype=object)
        for i, v in enumerate(nested):
            arr[i] = _to_object_cell(v)
        return arr
    return nested


def snapshot_to_matlab(snapshot: dict) -> dict:
    """Flatten a snapshot into a savemat-friendly dict (one struct per cell)."""
    cells = np.empty(len(snapshot["per_cell"]), dtype=object)
    for nn, pc in enumerate(snapshot["per_cell"]):
        cells[nn] = {
            "holoSortedImagingAllTrials": _to_object_cell(pc["holoSortedImagingAllTrials"]),
            "filtHoloSortedImagingAllTrials": _to_object_cell(pc["filtHoloSortedImagingAllTrials"]),
            "F0AllTrials": _to_object_cell(pc["F0AllTrials"]),
            "holoSortedImagingMean": _to_object_cell(pc["holoSortedImagingMean"]),
            "filtHoloSortedImagingMean": _to_object_cell(pc["filtHoloSortedImagingMean"]),
            "CIDffAllConds": _to_object_cell(pc["CIDffAllConds"]),
            "filtCIDffAllConds": _to_object_cell(pc["filtCIDffAllConds"]),
            "roiMeanF": pc["roiMeanF"],
            "bkgrndMeanF": pc["bkgrndMeanF"],
            "roiMeanFCorrected": pc["roiMeanFCorrected"],
            "subScalar": pc["subScalar"],
        }
    return {
        "liveSnapshot": {
            "perCell": cells,
            "Lholo": snapshot["Lholo"],
            "numFrames": snapshot["num_frames"]
            if snapshot["num_frames"] is not None else np.nan,
            "trialsSeen": np.asarray(snapshot["trials_seen"], dtype=float),
            "nTrialsSeen": snapshot["n_trials_seen"],
            "nTrialsTotal": snapshot["n_trials_total"],
        }
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LiveRunner:
    """Stateful live runner. Reusable via :meth:`process_trial_file`."""

    def __init__(self, ctx: context_mod.SessionContext, cfg: LiveConfig,
                 on_trial: Optional[Callable[[int, dict], None]] = None):
        self.ctx = ctx
        self.cfg = cfg
        self.on_trial = on_trial
        os.makedirs(cfg.output_dir, exist_ok=True)

        self.acc = LiveMappingAccumulator(
            n_cells=ctx.n_cells, n_conds=ctx.n_conds, n_holos=ctx.n_holos,
            up_or_down=ctx.up_or_down, imaging_freq=ctx.imaging_freq,
            ipi=ctx.ipi, n_pulses=ctx.n_pulses,
            pre_stim_window=ctx.pre_stim_window,
            post_stim_window=ctx.post_stim_window,
            n_trials_total=ctx.n_trials,
        )
        self.watcher = TrialFileWatcher(
            folder=cfg.watch_folder, pattern=cfg.pattern,
            trial_regex=cfg.trial_regex, stable_seconds=cfg.stable_seconds,
            poll_interval=cfg.poll_interval, verify_tiff=cfg.verify_tiff,
        )
        # trials whose metadata was not yet available: {tt: (path, retries)}
        self._deferred: deque = deque()

    # ------------------------------------------------------------------
    def _load_stack(self, path: str) -> np.ndarray:
        return tiff_io.read_stack(path, keep_odd_pages=self.cfg.keep_odd_pages)

    def process_trial_file(self, tt: int, path: str) -> bool:
        """Process one complete trial TIFF. Returns True if processed, False if
        deferred (metadata not yet available)."""
        if not self.ctx.has_metadata_for(tt):
            if self.cfg.reload_expstruct_on_miss:
                self.ctx.reload_expstruct()
            if not self.ctx.has_metadata_for(tt):
                return False

        meta = self.ctx.metadata_for_trial(tt)
        image_stack = self._load_stack(path)

        result = pipeline.process_one_trial(
            image_stack, self.ctx.n_cells, self.ctx.rough_rois,
            self.ctx.bkgrnd_global, meta["cc"], meta["sequence_this_trial"],
            self.ctx.zero_dummy_sequence, self.ctx.first_stim_times,
            meta["is_excluded"], self.ctx.up_or_down, self.ctx.imaging_freq,
            self.ctx.ipi, self.ctx.n_pulses, self.ctx.pre_stim_window,
            self.ctx.post_stim_window, self.ctx.start_time,
            use_bad_rows=self.ctx.use_bad_rows,
            common_f0=self.ctx.common_f0, f0_win_ms=self.ctx.f0_win_ms,
        )
        self.acc.add_trial(tt, result, is_excluded=meta["is_excluded"])
        return True

    # ------------------------------------------------------------------
    def _persist(self) -> None:
        cfg = self.cfg
        snap = self.acc.snapshot()
        _atomic_write_bytes(
            os.path.join(cfg.output_dir, cfg.snapshot_basename + ".pkl"),
            pickle.dumps(snap, protocol=pickle.HIGHEST_PROTOCOL),
        )
        _atomic_write_text(
            os.path.join(cfg.output_dir, cfg.status_basename + ".json"),
            json.dumps(self.acc.status(), indent=2, default=str),
        )
        if cfg.save_mat:
            from scipy.io import savemat
            mat_path = os.path.join(cfg.output_dir, cfg.snapshot_basename + ".mat")
            tmp = mat_path + ".tmp"
            savemat(tmp, snapshot_to_matlab(snap), do_compression=True)
            os.replace(tmp, mat_path)

    def _handle(self, tt: int, path: str, tag: str) -> None:
        try:
            done = self.process_trial_file(tt, path)
        except Exception as exc:  # keep the session alive; log and move on
            print(f"[live] ERROR processing trial {tt} ({path}): "
                  f"{type(exc).__name__}: {exc}", flush=True)
            return
        if not done:
            self._deferred.append((tt, path, 0))
            print(f"[live] deferred trial {tt}: stim metadata not yet available",
                  flush=True)
            return
        self._persist()
        st = self.acc.status()
        print(f"[live] {tag} trial {tt} (cond {self.acc.trial_cond.get(tt)}): "
              f"{st['n_trials_seen']}/{st['n_trials_total']} trials, "
              f"per-cond {st['trials_per_condition']}", flush=True)
        if self.on_trial is not None:
            self.on_trial(tt, self.acc.status())

    def _drain_deferred(self) -> None:
        if not self._deferred:
            return
        if self.cfg.reload_expstruct_on_miss:
            self.ctx.reload_expstruct()
        for _ in range(len(self._deferred)):
            tt, path, retries = self._deferred.popleft()
            if self.ctx.has_metadata_for(tt):
                self._handle(tt, path, "deferred->processed")
            elif retries + 1 >= self.cfg.max_defer_retries:
                print(f"[live] giving up on trial {tt} after "
                      f"{retries + 1} retries (no metadata)", flush=True)
            else:
                self._deferred.append((tt, path, retries + 1))

    # ------------------------------------------------------------------
    def run(self) -> dict:
        """Backfill existing files (optional), then watch until idle/stopped.

        Returns the final snapshot.
        """
        cfg = self.cfg
        if cfg.backfill_existing:
            for tf in self.watcher.existing_files():
                self.watcher.mark_emitted(tf.path)
                self._handle(tf.trial_number, tf.path, "backfill")

        print(f"[live] watching {cfg.watch_folder!r} for {cfg.pattern!r} "
              f"(poll {cfg.poll_interval}s, stable {cfg.stable_seconds}s)",
              flush=True)
        for tf in self.watcher.watch(max_idle_seconds=cfg.max_idle_seconds):
            self._handle(tf.trial_number, tf.path, "new")
            self._drain_deferred()

        # final drain attempt
        self._drain_deferred()
        self._persist()
        print("[live] watch loop ended.", flush=True)
        return self.acc.snapshot()


# ---------------------------------------------------------------------------
# High-level entry + CLI
# ---------------------------------------------------------------------------


def run_live(ctx: context_mod.SessionContext, cfg: LiveConfig,
             on_trial=None) -> dict:
    return LiveRunner(ctx, cfg, on_trial=on_trial).run()


def _load_rois(roi_path: str, roi_format: Optional[str],
               output_dir: Optional[str] = None,
               cellpose_cfg_path: Optional[str] = None):
    fmt = roi_format
    if fmt is None:
        ext = os.path.splitext(roi_path)[1].lower()
        fmt = "mat" if ext == ".mat" else "pickle"
    if fmt == "mat":
        return context_mod.load_rois_from_voltmapping_mat(roi_path)
    if fmt in ("pkl", "pickle"):
        return context_mod.load_rois_from_pickle(roi_path)
    if fmt == "cellpose":
        # --rois = mean image (.tif/.npy, e.g. exported meanFluorMaxDvStack).
        # Rough ROIs come from Cellpose; the global neuropil rings are then
        # bootstrapped from the same mean image via the standard global pass
        # (a 1-slice maxDvStack reproduces MATLAB's roiMeanMaxDvStack exactly,
        # since mean-over-trials of the mean image is itself).
        from .. import auto_roi as auto_roi_mod

        cfg = None
        if cellpose_cfg_path:
            import json as _json
            with open(cellpose_cfg_path) as fh:
                cfg = _json.load(fh)
        out_dir = os.path.join(output_dir or ".", "AutoROI")
        mean_img = auto_roi_mod.load_mean_image(roi_path)
        rough_rois, _report = auto_roi_mod.detect_rough_rois_cellpose(
            mean_img, cfg, out_dir
        )
        _, bkgrnd_global = context_mod.build_rois_from_reference(
            mean_img, mean_img[:, :, None], rough_rois
        )
        return rough_rois, bkgrnd_global
    raise ValueError(f"unknown roi_format {fmt!r}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live voltage-imaging mapping dF/F + sorted-struct builder."
    )
    p.add_argument("--expstruct", required=True,
                   help="Path to the ephys ExpStruct .mat (stim params).")
    p.add_argument("--rois", required=True,
                   help="ROI source: a prior voltMapping .mat, a rois .pkl, or "
                        "(with --roi-format cellpose) a mean-image .tif/.npy "
                        "to segment automatically.")
    p.add_argument("--roi-format", choices=["mat", "pickle", "cellpose"],
                   default=None,
                   help="Override ROI source format (else inferred from ext). "
                        "'cellpose' detects rough ROIs on the given mean image "
                        "via the shared Cellpose wrapper (see auto_roi.py).")
    p.add_argument("--cellpose-cfg", default=None,
                   help="Optional JSON file overriding auto_roi.DEFAULT_CFG "
                        "keys (diameter, min_area_px, dilate_radius_px, ...).")
    p.add_argument("--watch", required=True, dest="watch_folder",
                   help="Folder to watch for *_mc.tif trial files.")
    p.add_argument("--out", required=True, dest="output_dir",
                   help="Output directory for rolling snapshots.")
    p.add_argument("--up-or-down", required=True, choices=["1", "2"],
                   help="'1' upward GEVI, '2' downward GEVI.")
    p.add_argument("--common-f0", action="store_true",
                   help="Use the commonF0 (min-variance early baseline) variant.")
    p.add_argument("--pattern", default="*_mc.tif")
    p.add_argument("--trial-regex", default=r"_(\d+)_mc\.tif$")
    p.add_argument("--stable-seconds", type=float, default=3.0)
    p.add_argument("--poll-interval", type=float, default=2.0)
    p.add_argument("--no-verify-tiff", action="store_true")
    p.add_argument("--keep-odd-pages", action="store_true",
                   help="2-color interleaved stacks: keep MATLAB odd pages.")
    p.add_argument("--no-backfill", action="store_true",
                   help="Do not process files already present at startup.")
    p.add_argument("--max-idle-seconds", type=float, default=None,
                   help="Stop after this many idle seconds (default: run forever).")
    p.add_argument("--save-mat", action="store_true",
                   help="Also write a .mat snapshot each trial (scipy).")
    return p


def main(argv=None) -> int:
    args = build_argparser().parse_args(argv)
    rough_rois, bkgrnd_global = _load_rois(
        args.rois, args.roi_format,
        output_dir=args.output_dir, cellpose_cfg_path=args.cellpose_cfg,
    )
    ctx = context_mod.SessionContext.from_expstruct(
        args.expstruct, rough_rois, bkgrnd_global, args.up_or_down,
        common_f0=args.common_f0, roi_source=args.rois,
    )
    cfg = LiveConfig(
        watch_folder=args.watch_folder, output_dir=args.output_dir,
        pattern=args.pattern, trial_regex=args.trial_regex,
        stable_seconds=args.stable_seconds, poll_interval=args.poll_interval,
        verify_tiff=not args.no_verify_tiff, keep_odd_pages=args.keep_odd_pages,
        backfill_existing=not args.no_backfill,
        max_idle_seconds=args.max_idle_seconds, save_mat=args.save_mat,
    )
    print(f"[live] session: nCells={ctx.n_cells} nConds={ctx.n_conds} "
          f"nTrials(planned)={ctx.n_trials} imagingFreq={ctx.imaging_freq} "
          f"ipi={ctx.ipi} nPulses={ctx.n_pulses} "
          f"pre/post={ctx.pre_stim_window}/{ctx.post_stim_window}ms "
          f"upOrDown={ctx.up_or_down} commonF0={ctx.common_f0}", flush=True)
    run_live(ctx, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
