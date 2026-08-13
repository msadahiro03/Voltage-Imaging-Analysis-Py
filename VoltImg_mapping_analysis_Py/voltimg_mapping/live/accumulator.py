"""Incremental condition/holo-sorted dF/F accumulator.

Mirrors the accumulation + finalization of
:func:`voltimg_mapping.pipeline.run_dff`, but fed one trial at a time. Columns
are stored keyed by 1-based trial index and assembled in trial-ascending order
at ``snapshot`` time, so the result is identical to a batch ``run_dff`` over the
same set of trials **regardless of the order trials arrive** (rig TIFFs may
appear out of order).

``snapshot()`` returns the same per-cell struct ``run_dff`` returns:
``holoSortedImagingAllTrials`` / ``filtHoloSortedImagingAllTrials`` (per
condition cc, per holo hh: an ``Lholo x nTrialsForThatHolo`` matrix), plus
``F0AllTrials``, the per-holo means and 95% CIs, and the full-trial
``roiMeanF`` / ``bkgrndMeanF`` / ``roiMeanFCorrected`` matrices.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .. import dff as dff_mod


class LiveMappingAccumulator:
    """Accumulates per-trial :func:`pipeline.process_one_trial` outputs.

    Parameters mirror the shape inputs ``run_dff`` needs. ``n_trials_total`` is
    the planned trial count (from the ExpStruct); full-trial matrices are shaped
    to it, with columns for not-yet-seen trials left as NaN.
    """

    def __init__(self, n_cells, n_conds, n_holos, up_or_down,
                 imaging_freq, ipi, n_pulses, pre_stim_window,
                 post_stim_window, n_trials_total):
        self.n_cells = int(n_cells)
        self.n_conds = int(n_conds)
        self.n_holos = np.asarray(n_holos)
        self.up_or_down = str(up_or_down)
        self.n_trials_total = int(n_trials_total)
        self.L = dff_mod.lholo(ipi, n_pulses, pre_stim_window,
                               post_stim_window, imaging_freq)

        # [nn][cc][hh] -> {tt: column/scalar}
        self._holo = [[[dict() for _ in range(int(self.n_holos[cc]))]
                       for cc in range(self.n_conds)]
                      for _ in range(self.n_cells)]
        self._filt = [[[dict() for _ in range(int(self.n_holos[cc]))]
                       for cc in range(self.n_conds)]
                      for _ in range(self.n_cells)]
        self._f0 = [[[dict() for _ in range(int(self.n_holos[cc]))]
                     for cc in range(self.n_conds)]
                    for _ in range(self.n_cells)]

        # [nn] -> {tt: full-trial column/scalar/roi}
        self._roi_mean_f: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.n_cells)]
        self._bkgrnd_mean_f: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.n_cells)]
        self._roi_corr: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.n_cells)]
        self._sub_scalar: List[Dict[int, float]] = [dict() for _ in range(self.n_cells)]
        self._fine_rois: List[Dict[int, tuple]] = [dict() for _ in range(self.n_cells)]
        self._bkgrnd_rois: List[Dict[int, tuple]] = [dict() for _ in range(self.n_cells)]

        self.num_frames: Optional[int] = None
        self.trials_seen: set = set()      # 1-based
        self.trial_cond: Dict[int, int] = {}

    # ------------------------------------------------------------------
    def add_trial(self, tt: int, result: dict, is_excluded: bool = False) -> None:
        """Add one trial's :func:`pipeline.process_one_trial` output.

        ``tt`` is the 1-based trial index (aligns to ExpStruct + the imaging
        trial order). Re-adding the same ``tt`` overwrites it (idempotent
        reprocessing is safe).
        """
        tt = int(tt)
        cc = int(result["cc"])
        self.num_frames = int(result["num_frames"])
        self.trials_seen.add(tt)
        self.trial_cond[tt] = cc

        for nn in range(self.n_cells):
            cell = result["per_cell"][nn]
            self._fine_rois[nn][tt] = cell["fine_roi"]
            self._bkgrnd_rois[nn][tt] = cell["bkgrnd_roi"]

            for rec in cell["records"]:
                hid = int(rec["holo_id"])  # 1-based
                self._holo[nn][cc - 1][hid - 1][tt] = rec["dff"]
                self._filt[nn][cc - 1][hid - 1][tt] = rec["filtdff"]
                self._f0[nn][cc - 1][hid - 1][tt] = rec["f0"]

            if is_excluded:
                nf = self.num_frames
                self._roi_mean_f[nn][tt] = np.full(nf, np.nan)
                self._bkgrnd_mean_f[nn][tt] = np.full(nf, np.nan)
                self._roi_corr[nn][tt] = np.full(nf, np.nan)
                self._sub_scalar[nn][tt] = np.nan
            else:
                self._roi_mean_f[nn][tt] = np.asarray(cell["roi_mean_f"]).ravel()
                self._bkgrnd_mean_f[nn][tt] = np.asarray(cell["bkgrnd_mean_f"]).ravel()
                self._roi_corr[nn][tt] = np.asarray(cell["roi_corr"]).ravel()
                self._sub_scalar[nn][tt] = dff_mod.ALPHA_SCALAR

    # ------------------------------------------------------------------
    def _finalize_holo(self, cols_by_holo):
        """One cell's [cc][hh] {tt:col} dicts -> [cc][hh] (Lholo x nCols) mats,
        columns ordered by ascending tt (matches run_dff)."""
        out = [[None] * int(self.n_holos[cc]) for cc in range(self.n_conds)]
        for cc in range(self.n_conds):
            for hh in range(int(self.n_holos[cc])):
                d = cols_by_holo[cc][hh]
                if not d:
                    out[cc][hh] = np.zeros((self.L, 0))
                else:
                    cols = [np.asarray(d[tt]).ravel() for tt in sorted(d)]
                    out[cc][hh] = np.column_stack(cols)
        return out

    def _full_trial_matrix(self, per_trial_cols):
        """{tt: column} -> (num_frames x n_trials_total), unseen trials NaN."""
        if self.num_frames is None:
            return np.zeros((0, self.n_trials_total))
        mat = np.full((self.num_frames, self.n_trials_total), np.nan)
        for tt, col in per_trial_cols.items():
            if 1 <= tt <= self.n_trials_total:
                mat[:, tt - 1] = np.asarray(col).ravel()
        return mat

    def _per_trial_list(self, per_trial_vals, default=None):
        """{tt: value} -> list length n_trials_total (1-based tt at index tt-1)."""
        out = [default] * self.n_trials_total
        for tt, v in per_trial_vals.items():
            if 1 <= tt <= self.n_trials_total:
                out[tt - 1] = v
        return out

    # ------------------------------------------------------------------
    def snapshot(self) -> dict:
        """Build the current per-cell result struct (== ``run_dff`` so far)."""
        per_cell = []
        for nn in range(self.n_cells):
            holo_all = self._finalize_holo(self._holo[nn])
            filt_holo_all = self._finalize_holo(self._filt[nn])

            f0_all = [[np.asarray([self._f0[nn][cc][hh][tt]
                                   for tt in sorted(self._f0[nn][cc][hh])],
                                  dtype=float)
                       for hh in range(int(self.n_holos[cc]))]
                      for cc in range(self.n_conds)]

            mci = dff_mod.holo_means_and_ci(
                holo_all, filt_holo_all, self.n_conds, self.n_holos,
                self.up_or_down
            )

            sub = np.full(self.n_trials_total, np.nan)
            for tt, v in self._sub_scalar[nn].items():
                if 1 <= tt <= self.n_trials_total:
                    sub[tt - 1] = v

            per_cell.append({
                "holoSortedImagingAllTrials": holo_all,
                "filtHoloSortedImagingAllTrials": filt_holo_all,
                "F0AllTrials": f0_all,
                "holoSortedImagingMean": mci["mean"],
                "filtHoloSortedImagingMean": mci["filt_mean"],
                "CIDffAllConds": mci["ci"],
                "filtCIDffAllConds": mci["filt_ci"],
                "roiMeanF": self._full_trial_matrix(self._roi_mean_f[nn]),
                "bkgrndMeanF": self._full_trial_matrix(self._bkgrnd_mean_f[nn]),
                "roiMeanFCorrected": self._full_trial_matrix(self._roi_corr[nn]),
                "subScalar": sub,
                "fineRois": self._per_trial_list(self._fine_rois[nn]),
                "bkgrndRoisTrial": self._per_trial_list(self._bkgrnd_rois[nn]),
            })

        return {
            "per_cell": per_cell,
            "Lholo": self.L,
            "num_frames": self.num_frames,
            "trials_seen": sorted(self.trials_seen),
            "n_trials_seen": len(self.trials_seen),
            "n_trials_total": self.n_trials_total,
            "trial_cond_seen": dict(self.trial_cond),
        }

    # ------------------------------------------------------------------
    def status(self) -> dict:
        """Lightweight progress summary (no heavy arrays)."""
        seen = sorted(self.trials_seen)
        per_cond = {}
        for tt in seen:
            cc = self.trial_cond.get(tt)
            per_cond[cc] = per_cond.get(cc, 0) + 1
        return {
            "n_trials_seen": len(seen),
            "n_trials_total": self.n_trials_total,
            "trials_seen": seen,
            "trials_per_condition": per_cond,
            "num_frames": self.num_frames,
        }
