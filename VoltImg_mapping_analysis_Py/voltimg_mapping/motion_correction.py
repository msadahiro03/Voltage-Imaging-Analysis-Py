"""Motion-correction adapter (Stage D).

MATLAB uses NoRMCorre (NoRMCorreSetParms / normcorre / apply_shifts), a MATLAB
toolbox with no drop-in NumPy equivalent. Two supported strategies:

  1. External MATLAB parity (recommended for exact reproduction): run the MATLAB
     NoRMCorre step, save `_mc.tif` stacks, and point this Python port at that
     Motion_Corrected_Tiffs folder. The dF/F stage then reads those exact MC
     stacks, giving numerically faithful downstream results. This is the default
     assumption of `pipeline.run_dff_from_mc_folder`.

  2. Python rigid MC via CaImAn's NoRMCorre port (`caiman.motion_correction`),
     which is the same algorithm. Results will be close but NOT bit-identical to
     MATLAB NoRMCorre (different FFT/subpixel implementations). Use only if
     MATLAB parity is not required; validate against MATLAB.

The MATLAB NoRMCorre parameters (for whoever runs strategy 2 or the MATLAB step):
    NoRMCorreSetParms('d1',H,'d2',W,'bin_width',15,'max_shift',4,'us_fac',50,
                      'init_batch',1)  -- rigid, single global template.

`normcorre_fn` in run_motion_correction lets you inject either a CaImAn-backed
callable or a stub; if None, the function raises to force an explicit choice.
"""

from __future__ import annotations

import numpy as np


def build_global_template(
    trial_stacks_iter,
    exclude_trials,           # set of 1-based tt
    use_laser_filter=False,
    laser_filter_fn=None,
):
    """Pass 1: accumulate per-trial mean images into one global template.

    trial_stacks_iter yields (tt_1based, stack_hwt) for ALL trials in order.
    Returns (global_template (H, W) float32, n_template_trials).
    """
    accum = None
    n_template = 0
    for tt, stack in trial_stacks_iter:
        if tt in exclude_trials:
            continue
        s = stack
        if use_laser_filter and laser_filter_fn is not None:
            s, _ = laser_filter_fn(s)
        mean_img = np.mean(s.astype(np.float32), axis=2)
        if accum is None:
            accum = np.zeros_like(mean_img, dtype=np.float32)
        accum += mean_img
        n_template += 1
    if n_template == 0:
        raise RuntimeError(
            "No non-excluded trials available to build a global MC template."
        )
    return accum / n_template, n_template
