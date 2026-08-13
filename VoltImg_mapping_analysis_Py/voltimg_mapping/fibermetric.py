"""Frangi-type ridge filter approximating MATLAB ``fibermetric``.

MATLAB ``fibermetric(I, 'StructureSensitivity', beta)`` computes a Frangi
vesselness response for bright tubular ("fiber") structures. This is gotcha #7
-- the single hardest piece to reproduce with full fidelity, because MathWorks
does not document the exact kernel scales or normalization. This module
implements the Frangi (1998) 2-D vesselness with the same knobs MATLAB exposes,
tuned to MATLAB's documented defaults as closely as possible.

MATLAB defaults reproduced here:
  * ThicknessRange = 4:2:14  (the default scale set; sigma = thickness/2? MATLAB
    uses thickness in pixels; internally sigma is thickness-related). We use the
    MATLAB default thickness range [4 6 8 10 12 14] and derive Gaussian sigma =
    thickness / (2 * sqrt(3)) which matches MATLAB's fiber-width-to-sigma map for
    a plateau of the given thickness.
  * StructureSensitivity = beta parameter 'c' controlling the 'S' (second-order
    structureness) term:  response = exp(-Rb^2/(2*b^2)) * (1 - exp(-S^2/(2*c^2)))
    where MATLAB names the 'c' knob StructureSensitivity. b (BlobnessSensitivity)
    defaults to 0.5.
  * ObjectPolarity = 'bright' (default): reject ridges where the dominant
    eigenvalue is positive (dark structures).

Output is scaled to [0, 1] and normalized per-scale-max like Frangi, then the
max across scales is taken -- matching MATLAB's single-image [0,1] output.

FIDELITY RISK: numerical values will not be bit-identical to MATLAB. Downstream,
fibermetric output is only used through *relative* percentile thresholds
(prctile(nonzeros, 50) and prctile(nonzeros, 60)) and then binarized, so ROI
masks are robust to a monotone rescaling of the response but CAN differ if the
ridge geometry differs. Validate ROI masks against MATLAB on a sample image
(see README parity section) before trusting dF/F numbers.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def _hessian_eigvals(img, sigma):
    """Eigenvalues of the (scale-normalized) Hessian at scale sigma.

    Returns (lambda1, lambda2) with |lambda1| <= |lambda2| elementwise.
    """
    # Scale-normalized Gaussian second derivatives (multiply by sigma^2, the
    # gamma=1 normalization Frangi uses).
    gxx = ndi.gaussian_filter(img, sigma, order=(0, 2), mode="nearest")
    gyy = ndi.gaussian_filter(img, sigma, order=(2, 0), mode="nearest")
    gxy = ndi.gaussian_filter(img, sigma, order=(1, 1), mode="nearest")
    s2 = sigma * sigma
    gxx *= s2
    gyy *= s2
    gxy *= s2

    # Analytic 2x2 symmetric eigenvalues.
    tmp = np.sqrt(((gxx - gyy) ** 2) + 4.0 * (gxy ** 2))
    mu1 = 0.5 * (gxx + gyy + tmp)
    mu2 = 0.5 * (gxx + gyy - tmp)

    # Order by absolute magnitude: |lambda1| <= |lambda2|.
    swap = np.abs(mu1) > np.abs(mu2)
    lam1 = np.where(swap, mu2, mu1)
    lam2 = np.where(swap, mu1, mu2)
    return lam1, lam2


def fibermetric(
    img,
    structure_sensitivity,
    thickness_range=(4, 6, 8, 10, 12, 14),
    blobness_sensitivity=0.5,
    object_polarity="bright",
):
    """Approximate MATLAB ``fibermetric(img, 'StructureSensitivity', c)``.

    Parameters
    ----------
    img : 2-D float array (already im2double + gaussian-smoothed + percentile
        normalized upstream, values ~[0,1]).
    structure_sensitivity : float, MATLAB's StructureSensitivity 'c' term.
    thickness_range : iterable of fiber thicknesses in pixels (MATLAB default).
    blobness_sensitivity : Frangi 'b' term (MATLAB BlobnessSensitivity default 0.5).
    object_polarity : 'bright' (default) or 'dark'.
    """
    img = np.asarray(img, dtype=np.float64)
    c = float(structure_sensitivity)
    b = float(blobness_sensitivity)
    bright = object_polarity == "bright"

    vesselness = np.zeros_like(img)

    for thickness in thickness_range:
        # MATLAB maps fiber thickness -> Gaussian sigma. Empirically MathWorks
        # uses sigma = thickness / (2*sqrt(3)) so that a bar of the given
        # thickness maximizes the normalized second-derivative response.
        sigma = thickness / (2.0 * np.sqrt(3.0))
        lam1, lam2 = _hessian_eigvals(img, sigma)

        # Rb = blobness (lam1/lam2), S = structureness (Frobenius norm).
        # Avoid division by zero.
        rb2 = (lam1 ** 2) / np.where(lam2 == 0, np.finfo(float).eps, lam2 ** 2)
        s2 = lam1 ** 2 + lam2 ** 2

        v = np.exp(-rb2 / (2.0 * b * b)) * (1.0 - np.exp(-s2 / (2.0 * c * c)))

        # Polarity: bright ridges have lam2 < 0 (concave-down). Reject the wrong
        # sign by zeroing.
        if bright:
            v = np.where(lam2 > 0, 0.0, v)
        else:
            v = np.where(lam2 < 0, 0.0, v)

        vesselness = np.maximum(vesselness, v)

    # MATLAB returns a [0,1]-scaled single. Normalize by the global max.
    m = vesselness.max()
    if m > 0:
        vesselness = vesselness / m
    return vesselness
