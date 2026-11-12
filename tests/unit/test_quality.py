"""Tests of the image quality, wall thickness and indication cores."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import ndimage

from datalab_radiography.core import indications as cind
from datalab_radiography.core import quality, wall
from datalab_radiography.core import simulation as sim


def test_region_statistics_snr_and_cnr() -> None:
    """SNR and CNR are exact on analytic regions; detrending removes slopes."""
    rng = np.random.default_rng(0)
    rows, cols = np.mgrid[0:60, 0:80]
    noise = rng.normal(0.0, 2.0, rows.shape)
    flat = 100.0 + noise
    sloped = flat + 0.5 * cols + 0.2 * rows
    stats = quality.region_statistics(flat, rows, cols)
    assert stats.mean == pytest.approx(100.0, abs=0.1)
    assert stats.std == pytest.approx(2.0, rel=0.05)
    assert stats.snr == pytest.approx(50.0, rel=0.05)
    raw = quality.region_statistics(sloped, rows, cols)
    detrended = quality.region_statistics(sloped, rows, cols, detrend=True)
    assert raw.std > 5 * detrended.std
    assert detrended.std == pytest.approx(2.0, rel=0.05)
    assert quality.normalized_snr(100.0, 177.2) == pytest.approx(50.0)
    with pytest.raises(quality.QualityError):
        quality.normalized_snr(100.0, 0.0)
    with pytest.raises(quality.QualityError, match="fewer than 3"):
        quality.region_statistics(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="positions"):
        quality.region_statistics(flat, detrend=True)
    indication = quality.RegionStatistics(120.0, 3.0, 10)
    background = quality.RegionStatistics(100.0, 4.0, 100)
    assert quality.contrast_to_noise(indication, background) == (20.0, 5.0)
    assert quality.RegionStatistics(1.0, 0.0, 3).snr == math.inf


def test_profiles_and_baseline() -> None:
    """Robust profiles ignore defective pixels; baselines ignore features."""
    region = np.full((20, 50), 100.0)
    region[3, 10] = 16383.0
    region[7, 20] = 0.0
    assert np.allclose(quality.robust_profile(region, axis=0), 100.0)
    x = np.arange(200.0)
    profile = 1000.0 + 2.0 * x
    profile[80:90] -= 300.0
    assert np.allclose(quality.baseline(profile), 1000.0 + 2.0 * x)
    features, darker = quality.feature_profile(profile)
    assert darker and features[85] == pytest.approx(300.0)
    features, darker = quality.feature_profile(-profile)
    assert not darker and features[85] == pytest.approx(300.0)
    with pytest.raises(ValueError):
        quality.feature_profile(profile, "inverted")


def duplex_profile(
    focal_spot: float = 0.4, noise: bool = False, seed: int = 0
) -> tuple[np.ndarray, float]:
    """Return a profile across a simulated duplex IQI and its step in the IQI plane."""
    scene = sim.Scene(
        "duplex",
        {"IQI": 800.0},
        (
            sim.Box("Plate", "IQI", "steel", depth=8.0, supersampling=1),
            sim.duplex_iqi("Duplex", "IQI", -12.0, -3.0, length=6.0),
        ),
    )
    detector = sim.Detector(rows=60, cols=360, dead_fraction=0, hot_fraction=0)
    simulation = sim.simulate(
        scene,
        sim.Geometry(focal_spot_mm=focal_spot),
        detector,
        sim.Exposure(photons_per_pixel=20000.0, frames=4),
        seed=seed,
        noise=noise,
    )
    profile = quality.robust_profile(simulation.radiographs[0][10:50], axis=0)
    return profile, detector.pitch_mm / (1000.0 / 800.0)


def test_duplex_iqi_resolution() -> None:
    """The duplex analysis finds the elements in both directions."""
    profile, step = duplex_profile()
    result = quality.analyze_duplex(profile, step)
    assert not result.reversed
    numbers = [element.number for element in result.elements]
    assert numbers == list(range(1, len(numbers) + 1))
    assert all(element.resolved for element in result.elements[:-1])
    assert not result.elements[-1].resolved
    assert result.isrb == result.elements[-1].number
    previous = result.elements[-2]
    assert result.elements[-1].diameter_mm < result.srb_mm < previous.diameter_mm
    backward = quality.analyze_duplex(profile[::-1], step)
    assert backward.reversed and backward.isrb == result.isrb
    assert backward.srb_mm == pytest.approx(result.srb_mm)
    blurred = quality.analyze_duplex(duplex_profile(focal_spot=1.5)[0], step)
    assert blurred.isrb < result.isrb and blurred.srb_mm > result.srb_mm
    late = quality.analyze_duplex(profile[60:], step, first_element=3)
    assert late.elements[0].number == 3 and late.isrb == result.isrb


def test_duplex_errors_and_interpolation() -> None:
    """Profiles without IQI are refused; the SRb is interpolated at 20 %."""
    with pytest.raises(quality.QualityError, match="No duplex"):
        quality.analyze_duplex(np.random.default_rng(0).normal(0, 1, 300), 0.08)
    with pytest.raises(quality.QualityError, match="too short"):
        quality.analyze_duplex(np.ones(5), 0.08)
    with pytest.raises(ValueError):
        quality.analyze_duplex(np.ones(50), 0.0)
    with pytest.raises(ValueError):
        quality.analyze_duplex(np.ones(50), 0.1, first_element=14)
    elements = [
        quality.DuplexElement(8, 0.16, 60.0, 0, 1),
        quality.DuplexElement(9, 0.13, 40.0, 2, 3),
        quality.DuplexElement(10, 0.10, 0.0, 4, None),
    ]
    isrb, srb = quality.interpolate_srb(elements)
    assert isrb == 10 and srb == pytest.approx(0.13 * (0.10 / 0.13) ** 0.5)
    assert quality.interpolate_srb(elements[:2]) == (None, None)
    assert quality.interpolate_srb(elements[2:]) == (10, None)
    assert quality.element_dip(np.array([1.0, 0.0, 1.0]), 0, 2) == 100.0
    assert quality.element_dip(np.array([0.0, 0.0, 0.0]), 0, 2) == 0.0


def pipe_profile(
    outer: float = 14.0, inner: float = 10.5, step: float = 0.07, blur: float = 1.0
) -> np.ndarray:
    """Return an analytic profile across a steel pipe (raw intensities)."""
    u = (np.arange(600) - 300.3) * step
    thickness = 2 * np.sqrt(np.clip(outer**2 - u**2, 0, None))
    thickness -= 2 * np.sqrt(np.clip(inner**2 - u**2, 0, None))
    profile = 400.0 + 3000.0 * (1 + 0.05 * u / 20) * np.exp(-0.15 * thickness)
    return ndimage.gaussian_filter1d(profile, blur, mode="nearest")


def test_wall_thickness_methods() -> None:
    """Both methods measure the wall; the model fit corrects the blur."""
    step = 0.07
    profile = pipe_profile(step=step)
    features, darker = wall.attenuation_profile(profile)
    assert darker
    edges = wall.find_edges(features)
    left, right = edges.walls(step)
    assert left == pytest.approx(3.5, rel=0.03) and right == pytest.approx(
        3.5, rel=0.03
    )
    assert edges.outer_diameter(step) == pytest.approx(28.0, rel=0.01)
    fit = wall.fit_pipe(profile, edges)
    assert fit.wall(step) == pytest.approx(3.5, rel=0.005)
    assert 2 * fit.outer_radius * step == pytest.approx(28.0, rel=0.002)
    assert fit.center == pytest.approx(300.3, abs=0.05)
    assert fit.fitted.shape == profile.shape
    attenuation = -np.log((profile - 400.0) / 3000.0)
    bright = wall.fit_pipe(attenuation)
    assert bright.wall(step) == pytest.approx(3.5, rel=0.01)


def test_wall_errors() -> None:
    """Profiles that do not cross a whole pipe are refused."""
    with pytest.raises(quality.QualityError, match="No pipe"):
        wall.find_edges(np.zeros(100))
    profile = pipe_profile()
    features, _darker = wall.attenuation_profile(profile)
    with pytest.raises(quality.QualityError, match="whole pipe"):
        wall.find_edges(features[150:])
    with pytest.raises(quality.QualityError, match="free beam on both sides"):
        wall.attenuation_profile(profile[:300])


def disks_image(seed: int = 0) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """Return a noisy image with a gradient and bright disks (x, y, diameter)."""
    rng = np.random.default_rng(seed)
    rows, cols = np.mgrid[0:200, 0:300].astype(float)
    disks = [(60.0, 50.0, 16.0), (150.0, 120.0, 10.0), (240.0, 80.0, 6.0)]
    image = 1000.0 * (1 + cols / 600.0)
    for x, y, diameter in disks:
        inside = (cols - x) ** 2 + (rows - y) ** 2 <= (diameter / 2) ** 2
        image = np.where(inside, image * 1.15, image)
    image = image + rng.normal(0.0, 10.0, image.shape)
    image[30, 200] = 16383.0
    return image, disks


def test_indications_are_found_and_sized() -> None:
    """Bright disks are found once each and sized by their Feret diameters."""
    image, disks = disks_image()
    contrast, noise = cind.contrast_map(image, 30.0)
    assert noise == pytest.approx(0.01, rel=0.2)
    labels, regions = cind.detect(contrast)
    assert len(regions) == len(disks)
    assert labels.max() == len(disks)
    for (x, y, diameter), region in zip(disks, regions):
        assert region.cols.mean() == pytest.approx(x, abs=0.5)
        assert region.rows.mean() == pytest.approx(y, abs=0.5)
        size = cind.region_size(region, (0.1, 0.1))
        assert size.length == pytest.approx(0.1 * diameter, rel=0.2)
        assert size.equivalent_diameter == pytest.approx(0.1 * diameter, rel=0.2)
        assert region.contrast > 5
    assert not cind.detect(contrast, polarity=cind.POLARITY_DARKER)[1]
    assert len(cind.detect(-contrast, polarity=cind.POLARITY_BOTH)[1]) == 3
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, :100] = True
    assert len(cind.detect(contrast, mask)[1]) == 1
    noisy, _noise = cind.contrast_map(image, 30.0, despeckle=False)
    assert len(cind.detect(noisy)[1]) > len(disks)
    with pytest.raises(ValueError):
        cind.detect(contrast, polarity="up")
    with pytest.raises(ValueError):
        cind.detect(contrast, detection=2.0, sizing=3.0)
    with pytest.raises(ValueError, match="empty"):
        cind.contrast_map(image, 30.0, mask=np.zeros(image.shape, dtype=bool))


def test_feret_diameters() -> None:
    """Feret diameters of rectangles and degenerate sets."""
    corners = np.array([[0, 0], [4, 0], [4, 1], [0, 1], [2, 0.5]], dtype=float)
    length, width = cind.feret(corners)
    assert length == pytest.approx(math.hypot(4, 1)) and width == pytest.approx(1.0)
    assert cind.feret(np.array([[0.0, 0.0]])) == (0.0, 0.0)
    assert cind.feret(np.array([[0.0, 0.0], [3.0, 4.0]])) == (5.0, 0.0)
    assert cind.feret(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))[1] == 0.0
    region = cind.Region(1, np.array([0, 0, 0]), np.array([0, 1, 2]), 6.0)
    size = cind.region_size(region, (0.2, 0.1))
    assert size.length == pytest.approx(math.hypot(0.6, 0.1))
    assert size.area == pytest.approx(0.06)
