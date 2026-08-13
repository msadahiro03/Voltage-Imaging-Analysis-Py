"""Session setup for the live pipeline.

A :class:`SessionContext` bundles everything the per-trial dF/F needs that does
*not* change trial-to-trial:

- scalar/vector stim parameters derived from the ephys ``ExpStruct`` .mat
  (imaging freq, ipi, nPulses, pre/post-stim windows, startTime, per-condition
  firstStimTimes, per-trial trialCond and sequenceThisTrial, nHolos, powers),
- the per-cell **rough ROIs** and **global neuropil ring** (``bkgrnd_global``),
  which in batch come from the maxDvStack + hand-drawn masks. Live, they are
  supplied up front -- typically reused from a prior analysis of the same FOV.

It also resolves per-trial stim metadata: given a 1-based trial index ``tt`` it
returns the condition ``cc``, that trial's ``sequenceThisTrial``, and whether the
trial is excluded (live default: nothing excluded; run the trial-excluder later).

Coordinate convention matches the rest of the port: ROIs are 0-based
``(rows, cols)`` arrays (MATLAB ``[X,Y]=find`` -> X=row, Y=col; subtract 1).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .. import expstruct as expstruct_mod

# MATLAB constant (line 248): postStimWindow = 50 ms added after last pulse+ipi.
POST_STIM_WINDOW_MS = 50.0

RoiList = List[Tuple[np.ndarray, np.ndarray]]  # per cell: (rows0, cols0)


# ---------------------------------------------------------------------------
# ROI loading helpers
# ---------------------------------------------------------------------------


def _cellarr_to_roi_list(x_cells, y_cells) -> RoiList:
    """MATLAB cell arrays of 1-based X(row)/Y(col) coords -> 0-based (rows, cols).

    Accepts scipy.io.loadmat output (object arrays) or plain lists.
    """
    x_cells = np.atleast_1d(np.asarray(x_cells, dtype=object)).ravel()
    y_cells = np.atleast_1d(np.asarray(y_cells, dtype=object)).ravel()
    if x_cells.size != y_cells.size:
        raise ValueError(
            f"rough/bkgrnd X and Y cell counts differ ({x_cells.size} vs "
            f"{y_cells.size})"
        )
    out: RoiList = []
    for xr, yr in zip(x_cells, y_cells):
        rows = np.atleast_1d(np.asarray(xr)).ravel().astype(np.int64) - 1
        cols = np.atleast_1d(np.asarray(yr)).ravel().astype(np.int64) - 1
        out.append((rows, cols))
    return out


def load_rois_from_voltmapping_mat(
    mat_path: str,
) -> Tuple[RoiList, RoiList]:
    """Load rough ROIs + global neuropil rings from a prior ``voltMapping`` .mat.

    Reads ``roughRoiXAllCells``/``roughRoiYAllCells`` and
    ``bkgrndRoiXAllCells``/``bkgrndRoiYAllCells`` (the *global* rings, MATLAB
    lines 904-909) and returns them as 0-based ``(rows, cols)`` lists.

    The variables may live at the top level of the .mat or inside a struct named
    ``voltMapping``.
    """
    from scipy.io import loadmat

    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    def _get(name):
        if name in m:
            return m[name]
        if "voltMapping" in m and hasattr(m["voltMapping"], name):
            return getattr(m["voltMapping"], name)
        raise KeyError(
            f"{name!r} not found in {mat_path} (checked top level and "
            f"voltMapping struct)"
        )

    rough = _cellarr_to_roi_list(_get("roughRoiXAllCells"),
                                 _get("roughRoiYAllCells"))
    bkgrnd = _cellarr_to_roi_list(_get("bkgrndRoiXAllCells"),
                                  _get("bkgrndRoiYAllCells"))
    if len(rough) != len(bkgrnd):
        raise ValueError(
            f"nCells mismatch: {len(rough)} rough vs {len(bkgrnd)} bkgrnd"
        )
    return rough, bkgrnd


def load_rois_from_pickle(pkl_path: str) -> Tuple[RoiList, RoiList]:
    """Load ROIs from a pickle written by :func:`save_rois_to_pickle`.

    Expects a dict ``{"rough_rois": RoiList, "bkgrnd_global": RoiList}``.
    """
    with open(pkl_path, "rb") as fh:
        d = pickle.load(fh)
    return d["rough_rois"], d["bkgrnd_global"]


def save_rois_to_pickle(pkl_path: str, rough_rois: RoiList,
                        bkgrnd_global: RoiList) -> None:
    """Persist ROIs (0-based (rows, cols) lists) for reuse across sessions."""
    with open(pkl_path, "wb") as fh:
        pickle.dump({"rough_rois": rough_rois,
                     "bkgrnd_global": bkgrnd_global}, fh)


def build_rois_from_reference(
    mean_fluor_max_dv_stack: np.ndarray,
    max_dv_stack: np.ndarray,
    rough_rois: RoiList,
    struct_sensitivity: float = 2.0,
) -> Tuple[RoiList, RoiList]:
    """Compute global fine ROIs + global rings from a reference stack.

    Use this to bootstrap ``bkgrnd_global`` from a short pre-run (e.g. the first
    K trials' maxDvStack) when no prior analysis .mat is available. Delegates to
    :func:`voltimg_mapping.pipeline.compute_global_rois`.
    """
    from .. import pipeline

    gg = pipeline.compute_global_rois(
        mean_fluor_max_dv_stack, max_dv_stack, rough_rois
    )
    return gg["roi_global"], gg["bkgrnd_global"]


# ---------------------------------------------------------------------------
# Session context
# ---------------------------------------------------------------------------


@dataclass
class SessionContext:
    """Immutable-ish per-session setup consumed by the live runner.

    Build with :meth:`from_expstruct`. ``reload_expstruct`` re-reads the .mat in
    place (useful when trialCond / sequenceThisTrial are filled in as the
    experiment proceeds).
    """

    # --- stim params ---
    imaging_freq: float
    ipi: float
    n_pulses: float
    pre_stim_window: float
    post_stim_window: float
    start_time: float
    n_conds: int
    n_holos: np.ndarray
    powers: np.ndarray
    zero_dummy_sequence: np.ndarray
    up_or_down: str  # '1' upward GEVI, '2' downward GEVI

    # --- per-trial arrays (planned session; may grow) ---
    n_trials: int
    trial_cond: np.ndarray            # 1-based condition per trial
    sequence_this_trial: list         # per trial vector or None
    first_stim_times: list            # per cond vector (s)

    # --- ROIs (0-based (rows, cols) per cell) ---
    rough_rois: RoiList
    bkgrnd_global: RoiList

    # --- processing options ---
    n_cells: int = 0
    common_f0: bool = False
    f0_win_ms: float = 50.0
    use_bad_rows: bool = False
    exclude_trials: set = field(default_factory=set)  # 1-based

    # --- provenance ---
    expstruct_path: Optional[str] = None
    roi_source: Optional[str] = None

    # ------------------------------------------------------------------
    @classmethod
    def from_expstruct(
        cls,
        expstruct_path: str,
        rough_rois: RoiList,
        bkgrnd_global: RoiList,
        up_or_down: str,
        *,
        common_f0: bool = False,
        f0_win_ms: float = 50.0,
        use_bad_rows: bool = False,
        exclude_trials=None,
        post_stim_window_ms: float = POST_STIM_WINDOW_MS,
        pre_stim_window_ms: Optional[float] = None,
        roi_source: Optional[str] = None,
    ) -> "SessionContext":
        d = expstruct_mod.load_expstruct(expstruct_path)

        post_win = float(post_stim_window_ms)
        if pre_stim_window_ms is not None:
            pre_win = float(pre_stim_window_ms)
        else:
            # MATLAB line 249: preStimWindow = nextHoloDelay - postStimWindow.
            pre_win = float(d["next_holo_delay"]) - post_win

        # zeroDummySequence = outParams.sequence{1,2} (cond 2, 0-based idx 1).
        seqs = d["sequence"]
        if len(seqs) >= 2 and seqs[1] is not None:
            zero_dummy = np.asarray(seqs[1]).ravel()
        elif len(seqs) >= 1 and seqs[0] is not None:
            zero_dummy = np.asarray(seqs[0]).ravel()
        else:
            zero_dummy = np.asarray([0.0])

        n_cells = len(rough_rois)
        if len(bkgrnd_global) != n_cells:
            raise ValueError(
                f"nCells mismatch: {n_cells} rough vs {len(bkgrnd_global)} "
                f"bkgrnd_global"
            )

        if str(up_or_down) not in ("1", "2"):
            raise ValueError("up_or_down must be '1' (upward) or '2' (downward)")

        return cls(
            imaging_freq=float(d["imaging_freq"]),
            ipi=float(d["ipi"]),
            n_pulses=float(d["n_pulses"]),
            pre_stim_window=pre_win,
            post_stim_window=post_win,
            start_time=float(d["start_time"]),
            n_conds=int(d["n_conds"]),
            n_holos=np.asarray(d["n_holos"]),
            powers=np.asarray(d["powers"]),
            zero_dummy_sequence=zero_dummy,
            up_or_down=str(up_or_down),
            n_trials=int(d["n_trials"]),
            trial_cond=np.asarray(d["trial_cond"]),
            sequence_this_trial=list(d["sequence_this_trial"]),
            first_stim_times=list(d["first_stim_times"]),
            rough_rois=rough_rois,
            bkgrnd_global=bkgrnd_global,
            n_cells=n_cells,
            common_f0=common_f0,
            f0_win_ms=f0_win_ms,
            use_bad_rows=use_bad_rows,
            exclude_trials=set(exclude_trials or ()),
            expstruct_path=expstruct_path,
            roi_source=roi_source,
        )

    # ------------------------------------------------------------------
    def reload_expstruct(self) -> None:
        """Re-read the ExpStruct .mat, refreshing per-trial arrays and nTrials.

        Call when the rig fills in trialCond / sequenceThisTrial as trials run.
        Scalar stim params and ROIs are left untouched.
        """
        if self.expstruct_path is None:
            return
        d = expstruct_mod.load_expstruct(self.expstruct_path)
        self.n_trials = int(d["n_trials"])
        self.trial_cond = np.asarray(d["trial_cond"])
        self.sequence_this_trial = list(d["sequence_this_trial"])
        self.first_stim_times = list(d["first_stim_times"])

    # ------------------------------------------------------------------
    def has_metadata_for(self, tt: int) -> bool:
        """True if trialCond for 1-based trial ``tt`` is available and valid."""
        if tt < 1 or tt > len(self.trial_cond):
            return False
        cc = int(self.trial_cond[tt - 1])
        return 1 <= cc <= self.n_conds

    def metadata_for_trial(self, tt: int) -> dict:
        """Resolve per-trial stim metadata for 1-based trial ``tt``.

        Returns ``{"cc": int, "sequence_this_trial": ndarray|None,
        "is_excluded": bool}``. Raises if trialCond is missing/invalid -- the
        caller should defer the trial and retry after ``reload_expstruct``.
        """
        if not self.has_metadata_for(tt):
            raise KeyError(
                f"no valid trialCond for trial tt={tt} "
                f"(have {len(self.trial_cond)} trials); reload ExpStruct?"
            )
        cc = int(self.trial_cond[tt - 1])
        seq = None
        if tt - 1 < len(self.sequence_this_trial):
            seq = self.sequence_this_trial[tt - 1]
        return {
            "cc": cc,
            "sequence_this_trial": seq,
            "is_excluded": tt in self.exclude_trials,
        }
