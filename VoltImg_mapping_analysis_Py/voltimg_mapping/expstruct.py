"""Load the ephys ExpStruct .mat and normalize its nested fields.

scipy.io.loadmat returns MATLAB structs as nested numpy object arrays; this
module flattens the fields the pipeline needs into a plain dict/attribute view.
Use squeeze_me + struct_as_record=False so structs behave like objects.
"""

from __future__ import annotations

import numpy as np
from scipy.io import loadmat


def _as_list_of_vectors(cellarr):
    """MATLAB cell array of vectors -> python list of 1-D float arrays.

    Empty cells -> None (so downstream fallback logic can detect them).
    """
    out = []
    arr = np.atleast_1d(cellarr)
    for el in arr.ravel():
        a = np.atleast_1d(np.asarray(el)).ravel()
        if a.size == 0:
            out.append(None)
        else:
            out.append(a.astype(float) if np.issubdtype(a.dtype, np.number)
                       else a)
    return out


def load_expstruct(mat_path):
    """Load ExpStruct (and ExpStruct2 if present).

    Returns a dict with the scalar/vector fields the pipeline consumes. Trial
    ordering is 1-based in MATLAB; here we return 0-based arrays but keep the
    1-based trial semantics documented in the pipeline.
    """
    m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    exp = m["ExpStruct"]

    daq = exp.daqParams
    out_params = exp.outParams
    holo = exp.holoStimParams

    d = {
        "imaging_freq": float(np.asarray(exp.sampleFreq).item()),
        "fs": float(np.asarray(daq.Fs).item()),
        "trial_time": float(np.asarray(daq.maxSweepLengthSec).item()),
        "trial_cond": np.atleast_1d(np.asarray(exp.trialCond)).ravel().astype(int),
        "powers": np.atleast_1d(np.asarray(out_params.power)).ravel().astype(float),
        "sequence": _as_list_of_vectors(out_params.sequence),
        "sequence_this_trial": _as_list_of_vectors(out_params.sequenceThisTrial),
        "first_stim_times": _as_list_of_vectors(out_params.firstStimTimes),
        "pulse_dur": np.atleast_1d(np.asarray(out_params.pulseDur)).ravel().astype(float),
        "n_pulses_raw": np.atleast_1d(np.asarray(out_params.nPulses)).ravel().astype(float),
        "ipi_raw": np.atleast_1d(np.asarray(out_params.ipi)).ravel().astype(float),
        "n_holos": np.atleast_1d(np.asarray(holo.nHolos)).ravel().astype(int),
        "next_holo_delay_raw": np.atleast_1d(np.asarray(holo.nextHoloDelay)).ravel().astype(float),
        "start_time_ms": float(np.asarray(holo.startTime).item()),
        "mouse_id": exp.mouseID,
        "inputs": _as_list_of_vectors(exp.inputs),
    }

    d["n_trials"] = d["trial_cond"].size
    d["n_conds"] = len(d["sequence"])
    d["start_time"] = d["start_time_ms"] / 1000.0

    # Header "hacks" (MATLAB 77-88):
    n_holos = d["n_holos"].copy()
    n_holos[0] = n_holos.max()  # 0-holo/0mW hack
    d["n_holos"] = n_holos

    d["pulse_durs"] = _nonzeros_unique(d["pulse_dur"])
    d["n_pulses"] = _nonzeros_unique(d["n_pulses_raw"])[0]
    d["ipi"] = _nonzeros_unique(d["ipi_raw"])[0]
    d["next_holo_delay"] = _nonzeros_unique(d["next_holo_delay_raw"])[0]

    if "ExpStruct2" in m:
        d["ExpStruct2"] = m["ExpStruct2"]

    return d


def _nonzeros_unique(x):
    """MATLAB ``nonzeros(unique(x))`` -> sorted unique nonzero values."""
    u = np.unique(np.asarray(x, dtype=float))
    return u[u != 0]
