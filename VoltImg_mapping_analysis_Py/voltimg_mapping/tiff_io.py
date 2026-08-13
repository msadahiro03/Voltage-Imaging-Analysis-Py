"""Multi-page TIFF I/O and channel detection (Stage A support).

Uses tifffile. Stacks are returned as (H, W, T) to match MATLAB's frame-last
convention throughout the port.
"""

from __future__ import annotations

import numpy as np

try:
    import tifffile
except ImportError:  # pragma: no cover
    tifffile = None


def _require_tifffile():
    if tifffile is None:
        raise ImportError(
            "tifffile is required for TIFF I/O. Install with `pip install tifffile`."
        )


def read_stack(path, keep_odd_pages=False):
    """Read a multi-page TIFF -> (H, W, T) array (native dtype).

    keep_odd_pages: if True (2-color interleaved), keep MATLAB pages 1,3,5,...
    i.e. 0-based indices 0,2,4,... (gotcha #13).
    """
    _require_tifffile()
    arr = tifffile.imread(path)  # (T, H, W) or (H, W) for single page
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if keep_odd_pages:
        arr = arr[0::2, ...]
    # (T, H, W) -> (H, W, T)
    return np.transpose(arr, (1, 2, 0))


def n_pages(path):
    """Number of pages/directories in a TIFF."""
    _require_tifffile()
    with tifffile.TiffFile(path) as tf:
        return len(tf.pages)


def detect_channels(test_tiff_path):
    """Auto-detect single vs 2-color interleaved (MATLAB Stage A, 94-193).

    Returns rawImgNChannels in {1, 2}.
    """
    _require_tifffile()
    n_dirs = n_pages(test_tiff_path)
    if n_dirs < 4:
        return 1

    n_sample = min(120, n_dirs)
    with tifffile.TiffFile(test_tiff_path) as tf:
        frames = [tf.pages[pp].asarray().astype(np.float32)
                  for pp in range(n_sample)]

    frame_means = np.array([f.mean() for f in frames])
    # Odd/even (1-based) -> 0-based even index = MATLAB odd page.
    odd_frames = frames[0::2]   # MATLAB pages 1,3,5...
    even_frames = frames[1::2]  # MATLAB pages 2,4,6...
    odd_mean_img = np.mean(np.stack(odd_frames, axis=0), axis=0) if odd_frames else None
    even_mean_img = np.mean(np.stack(even_frames, axis=0), axis=0) if even_frames else None

    if odd_mean_img is not None and even_mean_img is not None:
        r = np.corrcoef(odd_mean_img.ravel(), even_mean_img.ravel())
        odd_even_img_corr = r[0, 1] if r.size >= 4 else 1.0
    else:
        odd_even_img_corr = 1.0

    if frame_means.size >= 3:
        lag1 = np.corrcoef(frame_means[:-1], frame_means[1:])
        lag2 = np.corrcoef(frame_means[:-2], frame_means[2:])
        lag1_corr = lag1[0, 1] if lag1.size >= 4 else 0.0
        lag2_corr = lag2[0, 1] if lag2.size >= 4 else 0.0
        alt_step_diff = np.mean(np.abs(np.diff(frame_means)))
        same_chan_diff = np.mean(np.abs(frame_means[2:] - frame_means[:-2]))
    else:
        lag1_corr = lag2_corr = alt_step_diff = same_chan_diff = 0.0

    eps = np.finfo(float).eps
    is_interleaved = (odd_even_img_corr < 0.90) and (
        (lag2_corr > lag1_corr + 0.10)
        or (alt_step_diff > 1.15 * max(same_chan_diff, eps))
    )
    return 2 if is_interleaved else 1


def write_stack_uint16_rescaled(path, stack_hwt):
    """Save (H, W, T) stack as LZW uint16, per-trial rescaled (MATLAB 637-666).

    imageStack_mc_uint16 = uint16((x-min)/(max-min)*65535). If max<=min -> zeros.
    """
    _require_tifffile()
    x = stack_hwt.astype(np.float64)
    mc_min = x.min()
    mc_max = x.max()
    if mc_max > mc_min:
        out = np.uint16((x - mc_min) / (mc_max - mc_min) * 65535.0)
    else:
        out = np.zeros_like(x, dtype=np.uint16)
    # (H, W, T) -> (T, H, W) pages.
    tifffile.imwrite(path, np.transpose(out, (2, 0, 1)), compression="lzw")
