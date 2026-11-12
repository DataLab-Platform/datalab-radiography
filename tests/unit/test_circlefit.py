"""Tests of the circle fit used to calibrate with a ball or a disk."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from datalab_radiography.core import circlefit


def projected_object(
    radius: float,
    shape: str = "sphere",
    blur: float = 0.8,
    noise: float = 0.01,
    bright: bool = False,
    seed: int = 0,
) -> tuple[np.ndarray, float, float]:
    """Return a radiograph of a ball or disk, with its true center."""
    size = int(4 * radius) + 24
    xc, yc = size / 2 + 0.37, size / 2 - 0.21
    factor = 4
    offsets = (np.arange(size * factor) + 0.5) / factor - 0.5
    yy, xx = np.meshgrid(offsets, offsets, indexing="ij")
    squared = (xx - xc) ** 2 + (yy - yc) ** 2
    if shape == "sphere":
        thickness = 2.0 * np.sqrt(np.clip(radius**2 - squared, 0.0, None))
    else:
        thickness = np.where(squared <= radius**2, 5.0, 0.0)
    attenuation = 0.08 * thickness
    image = attenuation if bright else np.exp(-attenuation)
    image = image.reshape(size, factor, size, factor).mean(axis=(1, 3))
    image = ndimage.gaussian_filter(image, blur)
    image += np.random.default_rng(seed).normal(0.0, noise, image.shape)
    return image, xc, yc


@pytest.mark.parametrize(
    ("radius", "shape", "blur", "bright", "tolerance"),
    [
        (6.0, "sphere", 0.5, False, 1e-2),
        (10.0, "sphere", 0.7, False, 5e-3),
        (20.3, "sphere", 1.0, False, 5e-3),
        (35.7, "sphere", 1.5, False, 5e-3),
        (12.5, "sphere", 0.9, True, 5e-3),
        (8.3, "disk", 0.6, False, 1e-2),
        (15.2, "disk", 0.8, False, 5e-3),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_model_fit_is_accurate(radius, shape, blur, bright, tolerance, seed) -> None:
    """The model fit finds the radius (within 0.5 % from 10 pixels) and center."""
    image, xc, yc = projected_object(radius, shape, blur, bright=bright, seed=seed)
    fit = circlefit.fit_circle(image, shape=shape)
    assert fit.radius == pytest.approx(radius, rel=tolerance)
    assert fit.xc == pytest.approx(xc, abs=0.05)
    assert fit.yc == pytest.approx(yc, abs=0.05)
    assert fit.diameter == pytest.approx(2 * fit.radius)


def test_edge_fit_and_physical_coordinates() -> None:
    """The edge fit is exact for disks, and follows the axis coordinates."""
    image, xc, yc = projected_object(15.2, "disk", 0.8)
    fit = circlefit.fit_circle(image, x0=10.0, y0=-5.0, dx=0.2, dy=0.2)
    assert fit.radius == pytest.approx(15.2 * 0.2, rel=1e-2)
    assert fit.xc == pytest.approx(10.0 + xc * 0.2, abs=0.02)
    assert fit.yc == pytest.approx(-5.0 + yc * 0.2, abs=0.02)
    assert fit.point_count >= 100
    assert fit.points.shape == (fit.point_count, 2)


def test_fit_errors() -> None:
    """Regions without a disk are refused."""
    with pytest.raises(circlefit.CircleFitError, match="contrast"):
        circlefit.fit_circle(np.ones((20, 20)))
    with pytest.raises(circlefit.CircleFitError, match="at least 8"):
        circlefit.fit_circle(np.ones((5, 20)))
    tiny = np.ones((30, 30))
    tiny[15, 15] = 0.0
    with pytest.raises(circlefit.CircleFitError):
        circlefit.fit_circle(tiny, smoothing=0)
    with pytest.raises(ValueError):
        circlefit.fit_circle(np.ones((20, 20)), shape="cube")
