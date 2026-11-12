"""Enhancement of the visibility of indications in radiographs.

Sizes are given in pixels along each axis (anisotropic sizes are allowed):
callers convert physical lengths to pixels with the magnification of the
object plane of interest.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import ndimage

BACKGROUND_GAUSSIAN = "gaussian"
BACKGROUND_MORPHOLOGICAL = "morphological"
BACKGROUND_METHODS = (BACKGROUND_GAUSSIAN, BACKGROUND_MORPHOLOGICAL)
FLATTEN_DIVIDE = "divide"
FLATTEN_SUBTRACT = "subtract"
FLATTEN_MODES = (FLATTEN_DIVIDE, FLATTEN_SUBTRACT)
AVERAGE_MEAN = "mean"
AVERAGE_MEDIAN = "median"
AVERAGE_CLIPPED = "clipped"
AVERAGE_METHODS = (AVERAGE_MEAN, AVERAGE_MEDIAN, AVERAGE_CLIPPED)

Size = float | Sequence[float]


def _sizes(size: Size) -> tuple[float, float]:
    """Return (rows, columns) sizes from a scalar or an (x, y) pair."""
    if isinstance(size, (int, float)):
        return float(size), float(size)
    sx, sy = size
    return float(sy), float(sx)


def _finite(data: np.ndarray) -> np.ndarray:
    """Return floating point data whose non-finite values are replaced."""
    array = np.asarray(data, dtype=float)
    if not np.isfinite(array).all():
        array = np.where(np.isfinite(array), array, np.nanmedian(array))
    return array


def _odd_padded(array: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """Pad an image by odd reflection, which extends linear trends exactly."""
    return np.pad(
        array, ((rows, rows), (cols, cols)), mode="reflect", reflect_type="odd"
    )


def background(data: np.ndarray, size: Size, method: str = BACKGROUND_GAUSSIAN):
    """Return the large-scale background of an image.

    The image is extended by odd reflection, so that gradients such as the heel
    effect are followed up to the borders.

    Args:
        data: image
        size: size of the features removed from the background, in pixels
         (scalar, or X and Y sizes): Gaussian sigma, or morphological size
        method: ``gaussian`` (smooth variations), or ``morphological``
         (opening then closing: removes small bright and dark features while
         keeping steps of the background)

    Returns:
        Background image
    """
    array = _finite(data)
    rows, cols = _sizes(size)
    if min(rows, cols) <= 0:
        raise ValueError("The background size must be strictly positive")
    if method == BACKGROUND_GAUSSIAN:
        pad_r, pad_c = int(np.ceil(4 * rows)), int(np.ceil(4 * cols))
        padded = _odd_padded(array, pad_r, pad_c)
        result = ndimage.gaussian_filter(padded, (rows, cols), mode="nearest")
    elif method == BACKGROUND_MORPHOLOGICAL:
        shape = (max(int(round(rows)) | 1, 3), max(int(round(cols)) | 1, 3))
        pad_r, pad_c = 2 * shape[0], 2 * shape[1]
        padded = _odd_padded(array, pad_r, pad_c)
        opened = ndimage.grey_opening(padded, size=shape, mode="nearest")
        closed = ndimage.grey_closing(opened, size=shape, mode="nearest")
        result = ndimage.uniform_filter(closed, shape, mode="nearest")
    else:
        raise ValueError(f"Unknown background method {method!r}")
    return result[pad_r : pad_r + array.shape[0], pad_c : pad_c + array.shape[1]]


def flatten_background(
    data: np.ndarray,
    size: Size,
    method: str = BACKGROUND_GAUSSIAN,
    mode: str = FLATTEN_DIVIDE,
) -> np.ndarray:
    """Remove the large-scale background of an image.

    The result keeps the mean level of the image: the background is divided
    (thickness variations, heel effect) or subtracted, then the mean of the
    background is restored.

    Args:
        data: image
        size: size of the background features, in pixels (see :func:`background`)
        method: background estimation method
        mode: ``divide`` or ``subtract``

    Returns:
        Flattened image
    """
    if mode not in FLATTEN_MODES:
        raise ValueError(f"Unknown flattening mode {mode!r}")
    array = _finite(data)
    estimate = background(array, size, method)
    level = float(np.mean(estimate))
    if mode == FLATTEN_SUBTRACT:
        return array - estimate + level
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(estimate != 0, array / estimate * level, np.nan)


def unsharp_mask(data: np.ndarray, sigma: Size, amount: float = 1.0) -> np.ndarray:
    """Enhance details smaller than sigma: I + amount (I - G_sigma(I)).

    Args:
        data: image
        sigma: Gaussian sigma, in pixels (scalar, or X and Y values)
        amount: gain of the details
    """
    array = _finite(data)
    rows, cols = _sizes(sigma)
    if min(rows, cols) <= 0:
        raise ValueError("The detail size must be strictly positive")
    blurred = ndimage.gaussian_filter(array, (rows, cols), mode="nearest")
    return array + float(amount) * (array - blurred)


def robust_average(
    frames: Sequence[np.ndarray], method: str = AVERAGE_CLIPPED, threshold: float = 3.0
) -> np.ndarray:
    """Average frames, rejecting outliers such as direct photon hits.

    Args:
        frames: frames of the same shape
        method: ``mean``, ``median``, or ``clipped`` (mean of the values within
         ``threshold`` robust standard deviations of the median of each pixel)
        threshold: rejection threshold of the clipped mean

    Raises:
        ValueError: if there are no frames, shapes differ or the method is unknown
    """
    if not frames:
        raise ValueError("At least one frame is required")
    shape = np.shape(frames[0])
    if any(np.shape(frame) != shape for frame in frames):
        raise ValueError("All frames must have the same shape")
    stack = np.stack([np.asarray(frame, dtype=float) for frame in frames])
    if method == AVERAGE_MEAN:
        return stack.mean(axis=0)
    median = np.median(stack, axis=0)
    if method == AVERAGE_MEDIAN:
        return median
    if method != AVERAGE_CLIPPED:
        raise ValueError(f"Unknown averaging method {method!r}")
    if len(frames) < 3:
        return stack.mean(axis=0)
    deviation = np.abs(stack - median)
    sigma = 1.4826 * np.median(deviation, axis=0)
    sigma = np.where(sigma > 0, sigma, np.inf)
    keep = deviation <= threshold * sigma
    keep |= deviation == 0
    return np.sum(np.where(keep, stack, 0.0), axis=0) / np.maximum(keep.sum(axis=0), 1)


def normalize(data: np.ndarray, low: float, high: float) -> np.ndarray:
    """Map values from [low, high] to [0, 1], clipping the others."""
    if not high > low:
        raise ValueError("Normalization needs low < high")
    return np.clip((_finite(data) - low) / (high - low), 0.0, 1.0)
