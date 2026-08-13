"""maxDvStack sampling plan (port of VoltImg_mapping_maxDvStackSamplingPlan.m).

pickPercentage is hard-coded to 100/100 = 1.0 in the MATLAB source, so ALL
eligible trials are picked (the "50%" in the comment is inactive -- gotcha #14).
Because pickPercentage == 1.0, randperm selects every eligible trial and the RNG
is irrelevant; the port is therefore deterministic and takes all eligible trials
(no seeding needed). If pickPercentage were < 1 you would need to match MATLAB's
RNG, which is not bit-reproducible in NumPy.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def max_dv_stack_sampling_plan(
    n_img_trials: int,
    imaging_freq: float,
    eligible_trial_tt,
) -> Tuple[np.ndarray, int]:
    """Return (max_dv_trial_mask, max_dv_frame_cap).

    eligible_trial_tt : 1-based trial indices (as in MATLAB). The returned
    max_dv_trial_mask is a 0-based boolean array of length n_img_trials where
    mask[tt0] is True if the 1-based trial (tt0+1) is selected.
    """
    max_dv_frame_cap = max(1, int(math.floor(float(imaging_freq) * 4)))
    max_dv_trial_mask = np.zeros(n_img_trials, dtype=bool)

    elig = np.asarray(eligible_trial_tt).ravel()
    elig = elig[(elig >= 1) & (elig <= n_img_trials)]
    # unique(..., 'stable')
    _, first_idx = np.unique(elig, return_index=True)
    elig = elig[np.sort(first_idx)]
    n_el = elig.size
    if n_el == 0:
        return max_dv_trial_mask, max_dv_frame_cap

    pick_percentage = 100 / 100  # 1.0 -- all eligible trials.
    n_pick = max(1, int(math.ceil(n_el * pick_percentage)))
    # pickPercentage == 1.0 -> n_pick == n_el -> every eligible trial selected;
    # RNG order is immaterial. Take all eligible (deterministic).
    picked = elig[:n_pick]
    # Convert 1-based trial indices to 0-based mask positions.
    max_dv_trial_mask[(picked - 1).astype(int)] = True

    return max_dv_trial_mask, max_dv_frame_cap
