"""Radiographic display: window (level and width) and gray-scale inversion.

The display settings never change the data: the window only sets the range of
values mapped to the gray scale.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

#: Largest number of values used to compute percentiles (larger images are
#: sampled with a regular stride)
MAX_SAMPLES = 1_000_000
#: Minimum fraction of the values equal to an extreme value for these values to
#: be considered as saturated (free beam) or clipped background
SATURATION_FRACTION = 1e-3

PHOTOMETRIC_MONOCHROME1 = "MONOCHROME1"
PHOTOMETRIC_MONOCHROME2 = "MONOCHROME2"


@dataclasses.dataclass(frozen=True)
class Window:
    """Range of values mapped from black to white.

    Args:
        low: value displayed black (white if the gray scale is inverted)
        high: value displayed white (black if the gray scale is inverted)
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        """Validate the window."""
        low, high = float(self.low), float(self.high)
        if not (math.isfinite(low) and math.isfinite(high)) or high <= low:
            raise ValueError("A window needs finite bounds with low < high")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def center(self) -> float:
        """Window level (center)."""
        return 0.5 * (self.low + self.high)

    @property
    def width(self) -> float:
        """Window width."""
        return self.high - self.low

    @classmethod
    def from_center_width(cls, center: float, width: float) -> Window:
        """Return the window of a level and a width."""
        if not width > 0:
            raise ValueError("The window width must be strictly positive")
        return cls(center - 0.5 * width, center + 0.5 * width)


def _sample(values: np.ndarray) -> np.ndarray:
    """Return at most :data:`MAX_SAMPLES` regularly spaced values."""
    if values.size > MAX_SAMPLES:
        values = values[:: int(math.ceil(values.size / MAX_SAMPLES))]
    return values


def _without_saturation(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Remove saturated values (free beam) and dead or clipped zero values.

    The maximum value is removed when it is frequent (detector saturation) or
    equal to the largest value of the data type. The minimum value is removed
    only when it is the smallest value of the data type or a frequent zero.
    """
    if values.size == 0:
        return values
    keep = np.ones(values.shape, dtype=bool)
    type_min = type_max = None
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        type_min, type_max = info.min, info.max
    vmin, vmax = values.min(), values.max()
    if vmax > vmin:
        is_max = values == vmax
        if vmax == type_max or is_max.mean() >= SATURATION_FRACTION:
            keep &= ~is_max
        is_min = values == vmin
        if vmin == type_min or (vmin == 0 and is_min.mean() >= SATURATION_FRACTION):
            keep &= ~is_min
    remaining = values[keep]
    return remaining if remaining.size else values


def percentile_window(
    data: np.ndarray,
    low: float = 0.5,
    high: float = 99.5,
    mask: np.ndarray | None = None,
    exclude_saturation: bool = True,
) -> Window:
    """Return a window between two percentiles of the image values.

    Args:
        data: image data
        low: lower percentile (0 to 100)
        high: upper percentile (0 to 100)
        mask: boolean array, True for the values to use (default: all)
        exclude_saturation: ignore the values equal to a saturated (free beam)
         or clipped extreme value, so that they do not compress the window

    Returns:
        Window

    Raises:
        ValueError: if the percentiles are invalid or no finite value is used
    """
    if not 0.0 <= low < high <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= low < high <= 100")
    array = np.asarray(data)
    if np.iscomplexobj(array):
        array = np.abs(array)
    values = array[mask] if mask is not None else array.ravel()
    values = _sample(values)
    values = values[np.isfinite(values)] if values.dtype.kind == "f" else values
    if values.size == 0:
        raise ValueError("No finite value to compute the window")
    if exclude_saturation:
        values = _without_saturation(values, array.dtype)
    lo, hi = (float(value) for value in np.percentile(values, [low, high]))
    if hi <= lo:
        span = max(abs(lo) * 1e-6, 0.5)
        lo, hi = lo - span, hi + span
    return Window(lo, hi)


def window_from_dicom(
    center: float, width: float, slope: float = 1.0, intercept: float = 0.0
) -> Window:
    """Return the window of DICOM attributes, in stored pixel values.

    DICOM windows apply to rescaled values (slope × stored + intercept), with
    the linear function of PS3.3 C.11.2.1.2.1.

    Args:
        center: Window Center attribute
        width: Window Width attribute (at least 1)
        slope: Rescale Slope attribute
        intercept: Rescale Intercept attribute
    """
    if not width >= 1.0:
        raise ValueError("DICOM window width must be at least 1")
    if slope == 0:
        raise ValueError("DICOM rescale slope must not be zero")
    low = center - 0.5 - 0.5 * (width - 1.0)
    high = center - 0.5 + 0.5 * (width - 1.0)
    if high <= low:
        high = low + 1.0
    stored = sorted(((low - intercept) / slope, (high - intercept) / slope))
    return Window(*stored)


def dicom_inversion(
    photometric: str | None, presentation_lut_shape: str | None = None
) -> bool | None:
    """Return whether DICOM attributes require an inverted gray scale.

    ``MONOCHROME1`` images display their minimum value white. A Presentation
    LUT Shape ``INVERSE`` inverts ``MONOCHROME2`` images (with ``MONOCHROME1``
    images, it only confirms their inversion).

    Returns:
        True or False, or None if the attributes are not monochrome DICOM ones
    """
    photometric = (photometric or "").strip().upper()
    shape = (presentation_lut_shape or "").strip().upper()
    if photometric == PHOTOMETRIC_MONOCHROME1:
        return True
    if photometric == PHOTOMETRIC_MONOCHROME2:
        return shape == "INVERSE"
    return None


def display_inversion(
    film_convention: bool,
    photometric: str | None = None,
    presentation_lut_shape: str | None = None,
) -> bool:
    """Return whether to invert the gray scale of a radiograph.

    DICOM images follow their attributes, whatever the film convention
    preference. Other images, whose values grow with the dose, are inverted
    with the film convention (dense matter bright).
    """
    dicom = dicom_inversion(photometric, presentation_lut_shape)
    return bool(film_convention) if dicom is None else dicom
