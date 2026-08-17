"""Ephys stages: baselining/exclusion (Stage B) and hologram sorting (Stage C).

Ports MATLAB lines 213-431. Trace signals are 1-D float arrays. Sample indices
computed from float times use MATLAB floor/ceil semantics EXACTLY -- the
floor/ceil/+2 interplay is designed to make per-holo lengths concatenatable, so
it is reproduced verbatim (gotcha #1).

Trial indexing is 1-based in the public API (tt in 1..nTrials) to match the
MATLAB caller and the imaging-trial alignment assumption, but excludeTrials is
stored as a set of 1-based ints. Arrays are 0-based internally.
"""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from .matlab_compat import (
    butter_lowpass,
    colon_count,
    matlab_filtfilt,
    matlab_round,
    std_n1_omitnan,
    tinv,
)


def baseline_and_exclude(mapping_inputs, start_time, fs, n_trials, v_threshold,
                         ephys_avail):
    """Stage B (MATLAB 213-236).

    mapping_inputs : list length n_trials of 1-D sweeps (mV), 1-based trial tt at
        index tt-1.
    Returns (base_volt_all, exclude_trials(set of 1-based ints),
             mapping_inputs_baselined(list)).
    """
    base_volt_all = np.zeros(n_trials)
    exclude_trials = []
    mapping_inputs_baselined = [None] * n_trials

    # 1:(startTime*Fs) -> floor(startTime*Fs) samples (gotcha #1).
    n_base = colon_count(start_time * fs)
    for tt in range(n_trials):
        sweep = np.asarray(mapping_inputs[tt], dtype=float).ravel()
        base_volt = float(np.mean(sweep[:n_base]))
        base_volt_all[tt] = base_volt
        mapping_inputs_baselined[tt] = sweep - base_volt
        if base_volt > v_threshold:
            exclude_trials.append(tt + 1)  # 1-based

    if ephys_avail == 2:
        exclude_trials = []

    return base_volt_all, set(exclude_trials), mapping_inputs_baselined


def _first_stim_vec(first_stim_times, cc_1based):
    """firstStimTimes{cc} with fallback to firstStimTimes{1,2} (MATLAB {1,2}).

    first_stim_times is a list; cc_1based is 1-based cond index. The MATLAB
    ``{1, 2}`` fallback references the 2nd cell (1-based) -> index 1 here.
    """
    vec = first_stim_times[cc_1based - 1]
    if vec is None or np.asarray(vec).size == 0:
        vec = first_stim_times[1]  # {1,2} -> second element, 0-based index 1
    return np.asarray(vec, dtype=float).ravel()


def sort_holograms(
    mapping_inputs_baselined,
    trial_cond,          # length n_trials, 1-based cond per trial
    n_trials,
    n_conds,
    n_holos,             # array/list, n_holos[cc-1] holos in cond cc
    first_stim_times,    # list of per-cond vectors (seconds)
    sequence_this_trial, # list length n_trials of arrays (may be empty)
    zero_dummy_sequence, # fallback sequence for empty sequenceThisTrial
    fs,
    ipi,
    n_pulses,
    pre_stim_window,     # ms
    post_stim_window,    # ms
    exclude_trials,      # set of 1-based ints
    cut_off_freq=480,
) -> Dict:
    """Stage C (MATLAB 238-431). Returns a dict of ephys outputs."""
    blp, alp = butter_lowpass(4, cut_off_freq, fs)

    # nPulseCoords (samples) -- MATLAB 251-254.
    n_pulse_coords = np.array([
        ((pp) * ipi / 1000 * fs) + pre_stim_window / 1000 * fs
        for pp in range(int(n_pulses))
    ])  # pp from 0..nPulses-1 mirrors (pp-1) with 1-based pp.

    # Prestim baseline index setup (MATLAB 264-274) -- computed but mostly used
    # for reporting; the per-holo detrend uses its own local indices.
    n_first_pulse_sample = matlab_round(n_pulse_coords[0])
    i_last_pre_stim = max(1, n_first_pulse_sample - 1)
    pre_stim_baseline_trim_ms = 2
    edge_samp = matlab_round(pre_stim_baseline_trim_ms / 1000 * fs)
    idx1 = min(i_last_pre_stim, 1 + edge_samp)
    idx2 = max(idx1, i_last_pre_stim - edge_samp)
    if idx2 < idx1:
        idx1 = 1
        idx2 = i_last_pre_stim
    pre_stim_baseline_idx = np.arange(idx1, idx2 + 1)  # 1-based inclusive

    # Unified nominal holo length (MATLAB 280-296).
    ephys_len_nominal = math.inf
    for cc in range(1, n_conds + 1):
        fst_vec = _first_stim_vec(first_stim_times, cc)
        for hh in range(int(n_holos[cc - 1])):
            fst = fst_vec[min(hh, fst_vec.size - 1)]
            i_lo = math.floor((fst - pre_stim_window / 1000) * fs)
            i_hi = math.ceil((fst - pre_stim_window / 1000) * fs) + \
                math.ceil((ipi * n_pulses + pre_stim_window + post_stim_window)
                          / 1000 * fs)
            ephys_len_nominal = min(ephys_len_nominal, i_hi - i_lo + 1)
    if not (ephys_len_nominal < math.inf) or ephys_len_nominal < 1:
        ephys_len_nominal = math.ceil(
            (ipi * n_pulses + pre_stim_window + post_stim_window) / 1000 * fs
        ) + 2

    # Smallest available overlap across non-excluded trials (MATLAB 298-320).
    min_avail = math.inf
    for tt in range(1, n_trials + 1):
        if tt in exclude_trials:
            continue
        n_sweep = mapping_inputs_baselined[tt - 1].size
        cc = int(trial_cond[tt - 1])
        fst_vec = _first_stim_vec(first_stim_times, cc)
        for hh in range(int(n_holos[cc - 1])):
            fst = fst_vec[min(hh, fst_vec.size - 1)]
            i_lo = math.floor((fst - pre_stim_window / 1000) * fs)
            i_hi = math.ceil((fst - pre_stim_window / 1000) * fs) + \
                math.ceil((ipi * n_pulses + pre_stim_window + post_stim_window)
                          / 1000 * fs)
            src_lo = max(1, i_lo)
            src_hi = min(n_sweep, i_hi)
            if src_lo <= src_hi:
                min_avail = min(min_avail, src_hi - src_lo + 1)

    ephys_holo_sweep_len = ephys_len_nominal
    if min_avail < math.inf and min_avail >= 1:
        ephys_holo_sweep_len = min(ephys_len_nominal, min_avail)
    if ephys_holo_sweep_len < 1:
        ephys_holo_sweep_len = 1
    ephys_holo_sweep_len = int(ephys_holo_sweep_len)

    # holoSeqIndex + holoSortedDataAllTrials (MATLAB 335-388).
    holo_seq_index = [[] for _ in range(n_conds)]  # list of holo-ID lists (cols)
    holo_sorted = [[[] for _ in range(int(n_holos[cc]))]
                   for cc in range(n_conds)]        # [cc][hh] -> list of columns
    cond_sorted_inputs = [[] for _ in range(n_conds)]

    n_pre_samp = max(1, matlab_round(pre_stim_window / 1000 * fs) - 1)

    for tt in range(1, n_trials + 1):
        cc = int(trial_cond[tt - 1])

        seq = sequence_this_trial[tt - 1]
        if seq is None or np.asarray(seq).size == 0:
            seq = zero_dummy_sequence
            sequence_this_trial[tt - 1] = seq
        seq = np.asarray(seq).ravel()

        # unique(seq,'stable') - min(...) + 1  (normalized 1-based holo IDs).
        _, first_idx = np.unique(seq, return_index=True)
        uniq_stable = seq[np.sort(first_idx)]
        holo_seq_this_trial = uniq_stable - uniq_stable.min() + 1  # 1-based IDs
        holo_seq_index[cc - 1].append(holo_seq_this_trial.astype(int))

        cond_sorted_inputs[cc - 1].append(mapping_inputs_baselined[tt - 1])

        if tt not in exclude_trials:
            sweep_this_trial = matlab_filtfilt(
                blp, alp, mapping_inputs_baselined[tt - 1]
            )
        else:
            sweep_this_trial = None

        fst_vec = _first_stim_vec(first_stim_times, cc)

        for hh in range(int(n_holos[cc - 1])):
            fst = fst_vec[hh]
            i_ephys_lo = math.floor((fst - pre_stim_window / 1000) * fs)  # 1-based

            if tt in exclude_trials:
                this_holo = np.full(ephys_holo_sweep_len, np.nan)
            else:
                n_sweep = sweep_this_trial.size
                this_holo = np.full(ephys_holo_sweep_len, np.nan)
                for ii in range(1, ephys_holo_sweep_len + 1):
                    src_idx = i_ephys_lo + ii - 1  # 1-based
                    if 1 <= src_idx <= n_sweep:
                        this_holo[ii - 1] = sweep_this_trial[src_idx - 1]
                v = this_holo.copy()
                n_s = v.size
                # preStimBaselineIdxLocal = 1:min(nPreSamp, nS)  (1-based)
                pre_len = min(n_pre_samp, n_s)
                pre_idx0 = np.arange(pre_len)  # 0-based indices 0..pre_len-1
                # tRel = ((1:nS)' - mean(preStimBaselineIdxLocal)) / Fs
                pre_idx_1based = np.arange(1, pre_len + 1)
                mean_pre = pre_idx_1based.mean() if pre_len > 0 else 0.0
                t_rel = (np.arange(1, n_s + 1) - mean_pre) / fs
                if pre_len >= 3:
                    # polyfit deg 1 on prestim, subtract polyval over full trace.
                    p = np.polyfit(t_rel[pre_idx0], v[pre_idx0], 1)
                    v = v - np.polyval(p, t_rel)
                # subtract omitnan median of prestim portion
                v = v - np.nanmedian(v[pre_idx0])
                this_holo = v.reshape(this_holo.shape)

            # holoSeqIndex{cc}(hh, end) -- the just-appended trial's holo ID.
            holo_id = int(holo_seq_this_trial[hh])
            holo_sorted[cc - 1][holo_id - 1].append(this_holo)

    # Convert holo_sorted lists of columns -> (Lholo x nTrials) matrices.
    holo_sorted_all = _stack_columns(holo_sorted, n_conds, n_holos)

    # Means + CIs (MATLAB 390-415).
    holo_sorted_mean = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    ci_ephys = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    return_dict = {}
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            mat = holo_sorted_all[cc][hh]
            holo_sorted_mean[cc][hh] = np.nanmean(mat, axis=1)

    # (CIs assembled by pipeline with UpOrDown; expose raw matrices + helper.)
    return_dict.update({
        "blp": blp, "alp": alp,
        "n_pulse_coords": n_pulse_coords,
        "holo_seq_index": holo_seq_index,
        "holo_sorted_all_trials": holo_sorted_all,
        "holo_sorted_mean": holo_sorted_mean,
        "cond_sorted_inputs": cond_sorted_inputs,
        "pre_stim_baseline_idx": pre_stim_baseline_idx,
        "ephys_holo_sweep_len": ephys_holo_sweep_len,
        "ephys_holo_sweep_len_nominal": int(ephys_len_nominal),
        "min_avail_across_trials": (None if min_avail == math.inf
                                    else int(min_avail)),
        "pre_stim_baseline_trim_ms": pre_stim_baseline_trim_ms,
        "n_first_pulse_sample": n_first_pulse_sample,
        "i_last_pre_stim": i_last_pre_stim,
    })
    return return_dict


def _stack_columns(holo_sorted, n_conds, n_holos):
    """Convert [cc][hh] list-of-column-vectors to (Lholo x nTrials) arrays."""
    out = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            cols = holo_sorted[cc][hh]
            if len(cols) == 0:
                out[cc][hh] = np.zeros((0, 0))
            else:
                out[cc][hh] = np.column_stack(cols)
    return out


def ephys_confidence_intervals(holo_sorted_all, n_conds, n_holos, up_or_down,
                               confidence_level=0.95):
    """MATLAB 398-415: CI per holo with sign flip (note: ephys flip is opposite
    of imaging -- '2' -> [lo,hi], '1' -> [-lo,-hi])."""
    ci = [[None] * int(n_holos[cc]) for cc in range(n_conds)]
    for cc in range(n_conds):
        for hh in range(int(n_holos[cc])):
            mat = holo_sorted_all[cc][hh]
            n_cols = mat.shape[1] if mat.ndim == 2 else 0
            means = np.nanmean(mat, axis=1)
            sem = std_n1_omitnan(mat, axis=1) / np.sqrt(n_cols) if n_cols else means * np.nan
            t_score = tinv((1 + confidence_level) / 2, n_cols - 1)
            moe = t_score * sem
            lower = means - moe
            upper = means + moe
            if str(up_or_down) == "2":
                ci[cc][hh] = np.column_stack([lower, upper])
            else:
                ci[cc][hh] = np.column_stack([-lower, -upper])
    return ci
