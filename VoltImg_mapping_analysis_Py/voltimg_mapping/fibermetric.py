"""Frangi-type ridge filter reproducing MATLAB ``fibermetric``.

MATLAB ``fibermetric(I, 'StructureSensitivity', c)`` computes a Frangi (1998)
vesselness response for bright tubular ("fiber") structures. Gotcha #7: the
final vesselness kernel is a C++ builtin (``images.internal.builtins
.fibermetric``), but the surrounding structure IS documented in fibermetric.m
(verified against the R2025b source):

  * default Thickness = 4:2:14
  * per scale:  sigma = thickness/6            <- NOT thickness/(2*sqrt(3))
                Ig = imgaussfilt(V, sigma, 'FilterSize', 2*ceil(3*sigma)+1)
                out = builtin(Ig, c, isBright, sigma)
                B = max(B, out)                <- no global-max rescale
  * StructureSensitivity default = diff(getrangefromclass(V))/100 (0.01 for
    float input); the pipeline always passes it explicitly (2).
  * ObjectPolarity 'bright' (default) / 'dark'. There is NO BlobnessSensitivity
    knob in MATLAB; Frangi's beta is fixed at 0.5.

The builtin evaluates the Frangi 2-D vesselness on the pre-smoothed image at
scale sigma: scale-normalized Hessian (sigma^2 * second derivatives), ordered
eigenvalues |l1| <= |l2|, Rb = l1/l2, S = sqrt(l1^2+l2^2),
V = exp(-Rb^2/(2*0.25)) * (1 - exp(-S^2/(2*c^2))), zeroed where l2 has the
wrong sign for the requested polarity.

FIDELITY: the builtin's derivative discretization was identified empirically
(see _hessian_eigvals_presmoothed) and this implementation matches MATLAB
R2025b fibermetric output to ~1e-16 max abs difference on ridge and random
test images, borders included. Bit-level agreement on other MATLAB releases
is not guaranteed if MathWorks changes the builtin.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .matlab_compat import imgaussfilt

_BETA = 0.5  # Frangi blobness; fixed in MATLAB (no user knob).


_D2_KERNEL = np.array([1.0, 0.0, -2.0, 0.0, 1.0]) / 4.0  # transfer -sin(k)^2
_D1_KERNEL = np.array([1.0, 0.0, -1.0]) / 2.0


def _hessian_eigvals_presmoothed(img_smoothed, sigma):
    """Scale-normalized Hessian eigenvalues of an ALREADY-smoothed image.

    MATLAB smooths with imgaussfilt first and hands the smoothed image to the
    C++ builtin. The builtin's derivative operator was identified empirically
    against R2025b (sinusoidal-grating transfer probe + full-image diff):
    diagonal terms are the 5-tap stencil [1 0 -2 0 1]/4 (central first
    difference applied twice, gain sin(k)^2 -- NOT the [1,-2,1] stencil and
    NOT a Gaussian derivative), the cross term is [1 0 -1]/2 along each axis,
    and the boundary is half-sample symmetric reflection (scipy 'reflect').
    With that combination the output matches MATLAB to ~1e-16 everywhere,
    borders included. Multiplied by sigma^2 (gamma=1 Frangi scale
    normalization; the builtin receives sigma for exactly this).

    Returns (lambda1, lambda2) with |lambda1| <= |lambda2| elementwise.
    """
    img = np.asarray(img_smoothed, dtype=np.float64)
    gxx = ndi.correlate1d(img, _D2_KERNEL, axis=1, mode="reflect")
    gyy = ndi.correlate1d(img, _D2_KERNEL, axis=0, mode="reflect")
    gx = ndi.correlate1d(img, _D1_KERNEL, axis=1, mode="reflect")
    gxy = ndi.correlate1d(gx, _D1_KERNEL, axis=0, mode="reflect")

    s2 = sigma * sigma
    gxx = gxx * s2
    gyy = gyy * s2
    gxy = gxy * s2

    tmp = np.sqrt(((gxx - gyy) ** 2) + 4.0 * (gxy ** 2))
    mu1 = 0.5 * (gxx + gyy + tmp)
    mu2 = 0.5 * (gxx + gyy - tmp)

    swap = np.abs(mu1) > np.abs(mu2)
    lam1 = np.where(swap, mu2, mu1)
    lam2 = np.where(swap, mu1, mu2)
    return lam1, lam2


def fibermetric(
    img,
    structure_sensitivity,
    thickness_range=(4, 6, 8, 10, 12, 14),
    object_polarity="bright",
):
    """MATLAB ``fibermetric(img, 'StructureSensitivity', c)``.

    Parameters
    ----------
    img : 2-D float array (already im2double + percentile-normalized upstream).
    structure_sensitivity : float, MATLAB's StructureSensitivity 'c' term.
    thickness_range : iterable of fiber thicknesses in pixels (MATLAB default
        4:2:14).
    object_polarity : 'bright' (default) or 'dark'.
    """
    img = np.asarray(img, dtype=np.float64)
    c = float(structure_sensitivity)
    bright = object_polarity == "bright"

    vesselness = np.zeros_like(img)

    for thickness in thickness_range:
        sigma = float(thickness) / 6.0  # fibermetric.m:103 (R2025b)
        # fibermetric.m:105: imgaussfilt(V, sigma, 'FilterSize', 2*ceil(3*sigma)+1)
        filter_size = 2 * int(np.ceil(3.0 * sigma)) + 1
        ig = imgaussfilt(img, sigma, filter_size=filter_size)

        lam1, lam2 = _hessian_eigvals_presmoothed(ig, sigma)

        rb2 = (lam1 ** 2) / np.where(lam2 == 0, np.finfo(float).eps, lam2 ** 2)
        s2 = lam1 ** 2 + lam2 ** 2

        v = np.exp(-rb2 / (2.0 * _BETA * _BETA)) * (1.0 - np.exp(-s2 / (2.0 * c * c)))

        # Bright ridges have lam2 < 0; zero the wrong-sign response.
        if bright:
            v = np.where(lam2 >= 0, 0.0, v)
        else:
            v = np.where(lam2 <= 0, 0.0, v)

        vesselness = np.maximum(vesselness, v)

    # MATLAB takes max across scales with NO global-max rescale
    # (fibermetric.m:110-111: B = max(B, out) only).
    return vesselness
