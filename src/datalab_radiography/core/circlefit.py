"""Sub-pixel fit of the edge of a ball or a disk in a radiograph.

The edge is located along rays cast from a first estimate of the center, at
the steepest slope of each intensity profile (with a parabolic sub-sample
refinement), then a circle is fitted to the edge points: algebraic fit, outlier
rejection, then geometric least squares. Rays are cast again from the fitted
center until it is stable.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import ndimage, optimize

#: Number of rays cast from the center
RAY_COUNT = 180
#: Sampling step along the rays, in pixels
RAY_STEP = 0.25

SHAPE_EDGE = "edge"
SHAPE_SPHERE = "sphere"
SHAPE_DISK = "disk"
SHAPES = (SHAPE_EDGE, SHAPE_SPHERE, SHAPE_DISK)


@dataclasses.dataclass(frozen=True)
class CircleFit:
    """Fitted circle, in the coordinates of the given axes.

    Args:
        xc: X coordinate of the center
        yc: Y coordinate of the center
        radius: radius
        rms_residual: RMS distance of the edge points to the circle
        point_count: number of edge points kept for the fit
        points: edge points (N × 2 array of X and Y coordinates)
    """

    xc: float
    yc: float
    radius: float
    rms_residual: float
    point_count: int
    points: np.ndarray = dataclasses.field(repr=False, compare=False)

    @property
    def diameter(self) -> float:
        """Diameter of the circle."""
        return 2.0 * self.radius


class CircleFitError(ValueError):
    """Raised when no circle can be fitted."""


def _initial_disk(data: np.ndarray) -> tuple[float, float, float]:
    """Return a first estimate (row, column, radius) of the dominant disk."""
    low, high = np.percentile(data, [2.0, 98.0])
    if not high > low:
        raise CircleFitError("The region has no contrast")
    threshold = 0.5 * (low + high)
    border = np.concatenate([data[0], data[-1], data[:, 0], data[:, -1]])
    # The background surrounds the disk: the disk is on the other side
    mask = data < threshold if np.median(border) > threshold else data > threshold
    labels, count = ndimage.label(mask)
    if count == 0:
        raise CircleFitError("No disk found in the region")
    sizes = ndimage.sum(mask, labels, index=np.arange(1, count + 1))
    label = int(np.argmax(sizes)) + 1
    area = float(sizes[label - 1])
    if area < 12:
        raise CircleFitError("The disk is too small (fewer than 12 pixels)")
    row, col = ndimage.center_of_mass(labels == label)
    return float(row), float(col), math.sqrt(area / math.pi)


def _edge_points(data: np.ndarray, row: float, col: float, radius: float) -> np.ndarray:
    """Return sub-pixel edge points (row, column) along rays."""
    angles = np.linspace(0.0, 2.0 * math.pi, RAY_COUNT, endpoint=False)
    distances = np.arange(0.5 * radius, 1.5 * radius + RAY_STEP, RAY_STEP)
    rows = row + np.outer(np.sin(angles), distances)
    cols = col + np.outer(np.cos(angles), distances)
    profiles = ndimage.map_coordinates(
        data, [rows.ravel(), cols.ravel()], order=1, mode="nearest"
    ).reshape(rows.shape)
    slopes = np.abs(np.gradient(profiles, axis=1))
    points = []
    for index in range(RAY_COUNT):
        ray_rows, ray_cols = rows[index], cols[index]
        inside = (
            (ray_rows >= 0)
            & (ray_rows <= data.shape[0] - 1)
            & (ray_cols >= 0)
            & (ray_cols <= data.shape[1] - 1)
        )
        if not inside.all():
            continue
        slope = slopes[index]
        peak = int(np.argmax(slope))
        if peak in (0, slope.size - 1):
            continue
        before, center, after = slope[peak - 1 : peak + 2]
        denominator = before - 2.0 * center + after
        shift = 0.5 * (before - after) / denominator if denominator else 0.0
        distance = distances[peak] + float(np.clip(shift, -0.5, 0.5)) * RAY_STEP
        points.append(
            (
                row + math.sin(angles[index]) * distance,
                col + math.cos(angles[index]) * distance,
            )
        )
    return np.asarray(points, dtype=float).reshape(-1, 2)


def _algebraic_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return the algebraic (Kåsa) circle fit of points."""
    matrix = np.column_stack([x, y, np.ones_like(x)])
    rhs = x**2 + y**2
    (a, b, c), *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    xc, yc = a / 2.0, b / 2.0
    return xc, yc, math.sqrt(max(c + xc**2 + yc**2, 0.0))


def _geometric_fit(
    x: np.ndarray, y: np.ndarray, start: tuple[float, float, float]
) -> tuple[float, float, float]:
    """Return the geometric least-squares circle fit of points."""

    def residuals(params: np.ndarray) -> np.ndarray:
        return np.hypot(x - params[0], y - params[1]) - params[2]

    result = optimize.least_squares(residuals, np.asarray(start), method="lm")
    xc, yc, radius = result.x
    return float(xc), float(yc), abs(float(radius))


def _fit_points(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Fit a circle with outlier rejection; return it and the kept points mask."""
    keep = np.ones(x.shape, dtype=bool)
    circle = _algebraic_fit(x, y)
    for _iteration in range(3):
        residual = np.hypot(x - circle[0], y - circle[1]) - circle[2]
        mad = np.median(np.abs(residual[keep] - np.median(residual[keep])))
        new_keep = np.abs(residual - np.median(residual[keep])) <= max(
            4.0 * 1.4826 * mad, 1e-9
        )
        if new_keep.sum() < 12:
            break
        keep = new_keep
        circle = _algebraic_fit(x[keep], y[keep])
    circle = _geometric_fit(x[keep], y[keep], circle)
    return (*circle, keep)


def fit_circle(
    data: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.0,
    dx: float = 1.0,
    dy: float = 1.0,
    smoothing: float = 0.7,
    shape: str = "edge",
) -> CircleFit:
    """Fit the edge of the dominant ball or disk of an image region.

    Args:
        data: image region holding one ball or disk on a background
        x0: X coordinate of the first column
        y0: Y coordinate of the first row
        dx: spacing of the columns
        dy: spacing of the rows
        smoothing: standard deviation of the Gaussian smoothing, in pixels
        shape: ``edge`` for the edge fit only, ``sphere`` or ``disk`` to refine
         it with a model of the projected ball or disk (see :func:`refine_model`)

    Returns:
        Fitted circle, in the coordinates defined by ``x0``, ``y0``, ``dx``, ``dy``

    Raises:
        CircleFitError: if no circle can be fitted
    """
    if shape not in SHAPES:
        raise ValueError(f"Unknown shape {shape!r}")
    raw = np.asarray(data, dtype=float)
    if raw.ndim != 2 or min(raw.shape) < 8:
        raise CircleFitError("The region must be an image of at least 8 × 8 pixels")
    if not np.isfinite(raw).all():
        raw = np.nan_to_num(raw, nan=float(np.nanmedian(raw)))
    array = ndimage.gaussian_filter(raw, smoothing) if smoothing > 0 else raw
    row, col, radius = _initial_disk(array)
    points = np.empty((0, 2))
    for _iteration in range(3):
        points = _edge_points(array, row, col, radius)
        if len(points) < 12:
            raise CircleFitError(
                "Too few edge points: the disk must lie entirely in the region"
            )
        new_col, new_row, new_radius, keep = _fit_points(points[:, 1], points[:, 0])
        moved = math.hypot(new_row - row, new_col - col) + abs(new_radius - radius)
        row, col, radius = new_row, new_col, new_radius
        if moved < 0.01:
            break
    rms = None
    if shape != SHAPE_EDGE:
        row, col, radius, rms = refine_model(raw, row, col, radius, shape)
    # Edge points in physical coordinates, fitted again for anisotropic spacing
    x = x0 + points[keep, 1] * dx
    y = y0 + points[keep, 0] * dy
    if shape == SHAPE_EDGE:
        xc, yc, radius_phys = _geometric_fit(
            x, y, (x0 + col * dx, y0 + row * dy, radius * math.sqrt(dx * dy))
        )
        residual = np.hypot(x - xc, y - yc) - radius_phys
        rms = float(np.sqrt(np.mean(residual**2)))
    else:
        xc, yc = x0 + col * dx, y0 + row * dy
        radius_phys = radius * math.sqrt(dx * dy)
        rms = rms * math.sqrt(dx * dy)
    return CircleFit(
        xc, yc, radius_phys, float(rms), int(keep.sum()), np.column_stack([x, y])
    )


def _supersampled_grid(
    shape: tuple[int, int], factor: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return sub-pixel row and column coordinates of a pixel grid."""
    offsets = (np.arange(factor) + 0.5) / factor - 0.5
    rows = (np.arange(shape[0])[:, None] + offsets[None, :]).ravel()
    cols = (np.arange(shape[1])[:, None] + offsets[None, :]).ravel()
    return np.meshgrid(rows, cols, indexing="ij")


def refine_model(
    data: np.ndarray, row: float, col: float, radius: float, shape: str
) -> tuple[float, float, float, float]:
    """Refine a circle by fitting a model of the projected ball or disk.

    The thickness of a sphere of radius R at distance r from its center is
    proportional to sqrt(R² - r²); a disk has a constant thickness. The model
    attenuates a linear background (Beer-Lambert law) when the object is darker
    than its background, or adds a contrast proportional to the thickness when
    it is brighter (attenuation image), then blurs it with a Gaussian.

    Args:
        data: image region (not smoothed)
        row: first estimate of the center row
        col: first estimate of the center column
        radius: first estimate of the radius, in pixels
        shape: ``sphere`` or ``disk``

    Returns:
        Center row, center column, radius and RMS residual of the edge fit, in
        pixels

    Raises:
        CircleFitError: if the model fit fails
    """
    margin = int(math.ceil(0.4 * radius + 6))
    r0 = max(int(math.floor(row - radius - margin)), 0)
    r1 = min(int(math.ceil(row + radius + margin)) + 1, data.shape[0])
    c0 = max(int(math.floor(col - radius - margin)), 0)
    c1 = min(int(math.ceil(col + radius + margin)) + 1, data.shape[1])
    region = data[r0:r1, c0:c1]
    factor = 4 if radius < 40 else 2
    rows, cols = _supersampled_grid(region.shape, factor)
    radial = np.hypot(
        np.arange(region.shape[0])[:, None] - (row - r0),
        np.arange(region.shape[1])[None, :] - (col - c0),
    )
    background = float(np.median(region[radial > radius + 2]))
    inside = float(np.median(region[radial < 0.3 * radius]))
    # Exponential model (raw intensities of a darker object, with a detector
    # offset), or linear model (attenuation images: brighter object, background
    # near zero or negative)
    exponential = 0 < inside < background
    if exponential:
        contrast = math.log(background / inside)
    else:
        contrast = inside - background
    start = np.array(
        [row - r0, col - c0, radius, background, 0.0, 0.0, contrast, 1.0, 0.0],
        dtype=float,
    )

    def model(params: np.ndarray) -> np.ndarray:
        prow, pcol, prad, base, slope_y, slope_x, amplitude, sigma, offset = params
        distance = np.hypot(rows - prow, cols - pcol)
        if shape == SHAPE_SPHERE:
            thickness = np.sqrt(np.clip(1.0 - (distance / prad) ** 2, 0.0, None))
        else:
            # Linear ramp over one sub-sample: continuous in the parameters
            thickness = np.clip((prad - distance) * factor + 0.5, 0.0, 1.0)
        thickness = thickness.reshape(
            region.shape[0], factor, region.shape[1], factor
        ).mean(axis=(1, 3))
        plane = (
            base
            + slope_y * (np.arange(region.shape[0])[:, None] - prow)
            + slope_x * (np.arange(region.shape[1])[None, :] - pcol)
        )
        if exponential:
            image = offset + plane * np.exp(-amplitude * thickness)
        else:
            image = plane + amplitude * thickness
        return ndimage.gaussian_filter(image, abs(sigma))

    def residuals(params: np.ndarray) -> np.ndarray:
        return (model(params) - region).ravel()

    lower = [-np.inf, -np.inf, 2.0, -np.inf, -np.inf, -np.inf, -np.inf, 0.05, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 20.0, np.inf]
    if not exponential or shape != SHAPE_SPHERE:
        # The offset is part of the background plane of the linear model, and
        # cannot be told from the amplitude for a disk (two levels only)
        lower[-1], upper[-1] = -1e-12, 1e-12
    try:
        result = optimize.least_squares(
            residuals, start, bounds=(lower, upper), x_scale="jac", diff_step=1e-4
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise CircleFitError(f"The model fit failed: {exc}") from exc
    if not result.success:
        raise CircleFitError(f"The model fit failed: {result.message}")
    prow, pcol, prad = result.x[:3]
    rms = float(np.sqrt(np.mean(result.fun**2)))
    return float(prow + r0), float(pcol + c0), float(prad), rms
