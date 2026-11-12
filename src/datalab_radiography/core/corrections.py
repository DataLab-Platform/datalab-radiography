"""Detector corrections: offset and gain (dark and flat), defective pixels."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

OUTPUT_CORRECTED = "corrected"
OUTPUT_TRANSMISSION = "transmission"
OUTPUTS = (OUTPUT_CORRECTED, OUTPUT_TRANSMISSION)


def mean_frame(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Return the mean of frames of the same shape, as floating point values.

    Raises:
        ValueError: if there is no frame or the shapes differ
    """
    if not frames:
        raise ValueError("At least one frame is required")
    shape = np.shape(frames[0])
    total = np.zeros(shape, dtype=float)
    for frame in frames:
        if np.shape(frame) != shape:
            raise ValueError("All frames must have the same shape")
        total += np.asarray(frame, dtype=float)
    return total / len(frames)


@dataclasses.dataclass(frozen=True)
class FlatFieldReport:
    """Quality indicators of an offset and gain correction.

    Attributes:
        dark_level: median of the mean dark frame
        gain_level: median of the mean flat minus dark frame
        invalid_gain_fraction: fraction of pixels whose flat is not above dark
        negative_fraction: fraction of pixels of the raw frame below dark
        saturated_fraction: fraction of pixels of the raw frame at its maximum,
         when it is frequent (saturation)
    """

    dark_level: float
    gain_level: float
    invalid_gain_fraction: float
    negative_fraction: float
    saturated_fraction: float


def flat_field_correct(
    raw: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    output: str = OUTPUT_CORRECTED,
    min_gain_fraction: float = 0.05,
) -> tuple[np.ndarray, FlatFieldReport]:
    """Correct the offset and gain non-uniformity of a raw frame.

    The transmission is T = (I - D) / (F - D). The corrected frame is
    T × median(F - D): it keeps the scale of digital numbers.

    Args:
        raw: raw frame I
        dark: mean dark frame D (beam off)
        flat: mean flat frame F (beam on, without object)
        output: ``corrected`` or ``transmission``
        min_gain_fraction: pixels whose gain F - D is below this fraction of its
         median are invalid (NaN in the result)

    Returns:
        Corrected frame (or transmission) and quality indicators

    Raises:
        ValueError: if the shapes differ or the flat is not above the dark
    """
    if output not in OUTPUTS:
        raise ValueError(f"Unknown output {output!r}")
    raw = np.asarray(raw, dtype=float)
    dark = np.asarray(dark, dtype=float)
    flat = np.asarray(flat, dtype=float)
    if not raw.shape == dark.shape == flat.shape:
        raise ValueError("Raw, dark and flat frames must have the same shape")
    gain = flat - dark
    gain_level = float(np.median(gain))
    if not gain_level > 0:
        raise ValueError("The flat frames are not brighter than the dark frames")
    invalid = gain < min_gain_fraction * gain_level
    signal = raw - dark
    with np.errstate(divide="ignore", invalid="ignore"):
        transmission = np.where(invalid, np.nan, signal / np.where(invalid, 1, gain))
    raw_max = raw.max()
    saturated = raw == raw_max
    saturated_fraction = float(saturated.mean())
    if saturated_fraction < 1e-3:
        saturated_fraction = 0.0
    report = FlatFieldReport(
        dark_level=float(np.median(dark)),
        gain_level=gain_level,
        invalid_gain_fraction=float(invalid.mean()),
        negative_fraction=float((signal < 0).mean()),
        saturated_fraction=saturated_fraction,
    )
    if output == OUTPUT_TRANSMISSION:
        return transmission, report
    return transmission * gain_level, report


def robust_sigma(values: np.ndarray) -> float:
    """Return the standard deviation estimated from the median absolute deviation."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return 0.0
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def defective_pixel_map(
    image: np.ndarray, size: int = 5, threshold: float = 6.0
) -> np.ndarray:
    """Return the pixels deviating from their local median.

    A pixel is defective when it differs from the median of its neighborhood
    by more than ``threshold`` robust standard deviations of these differences.

    Args:
        image: image
        size: size of the neighborhood (odd number of pixels)
        threshold: detection threshold, in robust standard deviations

    Returns:
        Boolean mask, True for defective pixels
    """
    data = np.asarray(image, dtype=float)
    finite = np.isfinite(data)
    if not finite.all():
        data = np.where(finite, data, np.nanmedian(data))
    residual = data - ndimage.median_filter(data, size=size, mode="nearest")
    sigma = robust_sigma(residual)
    if sigma == 0:
        sigma = float(np.std(residual)) or 1.0
    return (np.abs(residual) > threshold * sigma) | ~finite


def detector_defect_map(
    dark: np.ndarray, flat: np.ndarray, threshold: float = 6.0
) -> np.ndarray:
    """Return the defective pixels of a detector from its dark and flat frames.

    Hot pixels have a dark level far above the others; dead or weak pixels have
    a gain (flat minus dark) far from the local gain.
    """
    dark = np.asarray(dark, dtype=float)
    gain = np.asarray(flat, dtype=float) - dark
    hot = (dark - np.median(dark)) > threshold * max(robust_sigma(dark), 1e-12)
    return hot | defective_pixel_map(gain, threshold=threshold)


def correct_defective_pixels(
    image: np.ndarray, mask: np.ndarray, size: int = 5
) -> np.ndarray:
    """Replace defective pixels by the median of their valid neighbors.

    Args:
        image: image
        mask: defective pixels (True)
        size: size of the neighborhood (odd number of pixels)

    Returns:
        Corrected image (floating point values)
    """
    data = np.asarray(image, dtype=float).copy()
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return data
    half = size // 2
    work = data.copy()
    work[mask] = np.nan
    for _iteration in range(8):
        rows, cols = np.nonzero(np.isnan(work))
        if not rows.size:
            break
        padded = np.pad(work, half, mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))
        neighborhoods = windows[rows, cols].reshape(rows.size, -1)
        with np.errstate(all="ignore"):
            valid = np.isfinite(neighborhoods)
            sorted_values = np.sort(np.where(valid, neighborhoods, np.inf), axis=1)
            counts = valid.sum(axis=1)
            has_values = counts > 0
            lower = sorted_values[np.arange(rows.size), np.maximum(counts - 1, 0) // 2]
            upper = sorted_values[np.arange(rows.size), np.maximum(counts, 1) // 2]
            medians = np.where(has_values, 0.5 * (lower + upper), np.nan)
        work[rows, cols] = medians
    still_missing = np.isnan(work)
    if still_missing.any():
        work[still_missing] = np.nanmedian(work)
    data[mask] = work[mask]
    return data
