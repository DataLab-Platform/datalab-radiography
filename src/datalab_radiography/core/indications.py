"""Detection and sizing of indications (pores, cracks, inclusions).

Indications are found as regions deviating from the local background by more
than a number of noise standard deviations (hysteresis: seeds at a detection
threshold, extent at a lower sizing threshold), then sized with their Feret
diameters in a plane of known scale. No acceptance criterion is applied.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import ndimage, spatial

from . import enhancement

POLARITY_BRIGHTER = "brighter"
POLARITY_DARKER = "darker"
POLARITY_BOTH = "both"
POLARITIES = (POLARITY_BRIGHTER, POLARITY_DARKER, POLARITY_BOTH)

_CORNERS = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]])


@dataclasses.dataclass(frozen=True)
class Region:
    """Pixels of a detected indication.

    Attributes:
        label: label in the label image
        rows: row indices
        cols: column indices
        contrast: mean relative contrast (signed, in noise units)
    """

    label: int
    rows: np.ndarray
    cols: np.ndarray
    contrast: float


@dataclasses.dataclass(frozen=True)
class Size:
    """Size of an indication in a plane.

    Attributes:
        length: largest Feret diameter
        width: smallest Feret diameter
        area: area
    """

    length: float
    width: float
    area: float

    @property
    def equivalent_diameter(self) -> float:
        """Diameter of the disk of the same area."""
        return 2.0 * math.sqrt(self.area / math.pi)


def contrast_map(
    data: np.ndarray,
    size: enhancement.Size,
    method: str = enhancement.BACKGROUND_MORPHOLOGICAL,
    mask: np.ndarray | None = None,
    despeckle: bool = True,
) -> tuple[np.ndarray, float]:
    """Return the contrast to the background, in noise standard deviations.

    The contrast is relative (ratio to the background minus 1) for positive
    backgrounds, absolute otherwise.

    Args:
        data: image
        size: background feature size, in pixels (see
         :func:`.enhancement.background`): larger than the indications
        method: background estimation method
        mask: True for the pixels used to estimate the noise (default: all)
        despeckle: filter the contrast with a 3×3 median, so that isolated
         pixels (defective pixels, direct photon hits) are not detected; the
         noise is estimated before filtering

    Returns:
        Contrast map and noise standard deviation of the contrast
    """
    array = np.asarray(data, dtype=float)
    estimate = enhancement.background(array, size, method)
    region = np.ones(array.shape, dtype=bool) if mask is None else mask
    if not region.any():
        raise ValueError("The detection region is empty")
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.all(estimate[region] > 0):
            residual = array / estimate - 1.0
        else:
            residual = array - estimate
    values = residual[region & np.isfinite(residual)]
    # Morphological backgrounds follow the lower envelope of the noise: center
    # the residual so that the background has a zero contrast
    center = float(np.median(values))
    residual = residual - center
    values = values - center
    noise = float(1.4826 * np.median(np.abs(values)))
    if not noise > 0:
        noise = float(np.std(values)) or 1.0
    contrast = residual / noise
    if despeckle:
        contrast = ndimage.median_filter(contrast, size=3, mode="nearest")
    return contrast, noise


def detect(
    contrast: np.ndarray,
    mask: np.ndarray | None = None,
    polarity: str = POLARITY_BRIGHTER,
    detection: float = 5.0,
    sizing: float = 2.5,
    min_pixels: int = 4,
) -> tuple[np.ndarray, list[Region]]:
    """Detect indications on a contrast map (in noise units).

    Args:
        contrast: contrast map, see :func:`contrast_map`
        mask: True where indications are searched (default: everywhere)
        polarity: ``brighter`` (voids on raw or transmission images),
         ``darker`` (dense inclusions, or voids on attenuation images) or ``both``
        detection: detection threshold (an indication has at least one pixel
         beyond it)
        sizing: sizing threshold, which gives the extent of the indications
        min_pixels: smallest indication, in pixels

    Returns:
        Label image (0: background) and regions, by decreasing area
    """
    if polarity not in POLARITIES:
        raise ValueError(f"Unknown polarity {polarity!r}")
    if not 0 < sizing <= detection:
        raise ValueError("Thresholds must satisfy 0 < sizing <= detection")
    region = np.ones(contrast.shape, dtype=bool) if mask is None else mask
    values = np.nan_to_num(contrast, nan=0.0)
    labels = np.zeros(contrast.shape, dtype=np.int32)
    regions: list[Region] = []
    signs = {POLARITY_BRIGHTER: (1,), POLARITY_DARKER: (-1,), POLARITY_BOTH: (1, -1)}
    for sign in signs[polarity]:
        signed = sign * values
        extent, count = ndimage.label((signed > sizing) & region)
        if not count:
            continue
        seeded = np.unique(extent[(signed > detection) & region])
        for index in seeded[seeded > 0]:
            rows, cols = np.nonzero(extent == index)
            if rows.size < min_pixels:
                continue
            mean = float(np.mean(values[rows, cols]))
            regions.append(Region(0, rows, cols, mean))
    regions.sort(key=lambda item: item.rows.size, reverse=True)
    numbered = []
    for label, item in enumerate(regions, start=1):
        labels[item.rows, item.cols] = label
        numbered.append(dataclasses.replace(item, label=label))
    return labels, numbered


def feret(points: np.ndarray) -> tuple[float, float]:
    """Return the largest and smallest Feret diameters of a set of points."""
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3:
        return (
            (float(spatial.distance.pdist(points).max()), 0.0)
            if len(points) > 1
            else (0.0, 0.0)
        )
    try:
        hull = points[spatial.ConvexHull(points).vertices]
    except spatial.QhullError:
        span = float(spatial.distance.pdist(points).max())
        return span, 0.0
    largest = float(spatial.distance.pdist(hull).max())
    smallest = math.inf
    for index, start in enumerate(hull):
        edge = hull[(index + 1) % len(hull)] - start
        length = float(np.hypot(*edge))
        if length == 0:
            continue
        normal = np.array([-edge[1], edge[0]]) / length
        smallest = min(smallest, float(np.ptp((hull - start) @ normal)))
    return largest, smallest


def region_size(region: Region, scale: tuple[float, float]) -> Size:
    """Return the size of an indication, each pixel spanning ``scale`` (X, Y)."""
    sx, sy = scale
    centers = np.column_stack([region.cols, region.rows]).astype(float)
    corners = (centers[:, None, :] + _CORNERS[None, :, :]).reshape(-1, 2)
    length, width = feret(corners * [sx, sy])
    return Size(length, width, region.rows.size * sx * sy)
