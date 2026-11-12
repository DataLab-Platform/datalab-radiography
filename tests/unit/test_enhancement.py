"""Tests of the enhancement of radiographs."""

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects import create_image

from datalab_radiography.core import enhancement as cenh
from datalab_radiography.core import magnification as mag
from datalab_radiography.workflow import computations, objects
from datalab_radiography.workflow import enhancement as wenh


def pore_image(
    shape: tuple[int, int] = (120, 160), noise: float = 0.0, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return a radiograph with a strong heel gradient and a small pore."""
    rows, cols = np.mgrid[: shape[0], : shape[1]].astype(float)
    background = 1000.0 * (1.0 + 0.6 * cols / shape[1]) * (1.0 + 0.2 * rows / shape[0])
    pore = np.exp(-((rows - 60.0) ** 2 + (cols - 80.0) ** 2) / (2 * 1.5**2))
    image = background * (1.0 + 0.05 * pore)
    image += np.random.default_rng(seed).normal(0.0, noise, shape)
    return image, pore > 0.5


def magnified_image(data: np.ndarray, scale_mm: float = 0.05) -> object:
    """Return an image with an active magnification of ``scale_mm`` mm per pixel."""
    image = create_image("radiograph", data)
    objects.set_magnifications(
        image,
        mag.MagnificationSet((mag.Magnification("Weld", (scale_mm, scale_mm)),)),
    )
    return image


def test_flatten_background_removes_gradients() -> None:
    """Flattening removes the heel gradient and keeps the pore and the mean level."""
    image, pore = pore_image()
    rows, cols = np.mgrid[: image.shape[0], : image.shape[1]]
    far = np.hypot(rows - 60.0, cols - 80.0) > 12.0
    for method in cenh.BACKGROUND_METHODS:
        for mode in cenh.FLATTEN_MODES:
            flat = cenh.flatten_background(image, 15.0, method, mode)
            assert np.std(flat[far]) < 0.002 * np.std(image), (method, mode)
            assert np.mean(flat) == pytest.approx(np.mean(image), rel=0.02)
            assert flat[pore].mean() > np.median(flat) * 1.02, (method, mode)
    anisotropic = cenh.flatten_background(image, (15.0, 5.0))
    assert anisotropic.shape == image.shape
    with pytest.raises(ValueError, match="strictly positive"):
        cenh.background(image, 0.0)
    with pytest.raises(ValueError, match="method"):
        cenh.background(image, 5.0, "rolling ball")
    with pytest.raises(ValueError, match="mode"):
        cenh.flatten_background(image, 5.0, mode="ratio")
    holed = image.copy()
    holed[0, 0] = np.nan
    assert np.isfinite(cenh.flatten_background(holed, 15.0)[1:, 1:]).all()


def test_morphological_background_ignores_small_features() -> None:
    """The morphological background does not follow pores smaller than its size."""
    image, pore = pore_image()
    background = cenh.background(image, 9.0, cenh.BACKGROUND_MORPHOLOGICAL)
    assert background[pore].mean() < image[pore].mean() * 0.99


def test_unsharp_mask_increases_detail_contrast() -> None:
    """The unsharp mask increases the contrast of details smaller than sigma."""
    image, pore = pore_image()
    flat = cenh.flatten_background(image, 15.0)
    level = np.median(flat)
    sharp = cenh.unsharp_mask(flat, 4.0, 2.0)
    before = flat[pore].mean() - level
    after = sharp[pore].mean() - level
    assert after > 2.0 * before
    assert np.array_equal(cenh.unsharp_mask(flat, 4.0, 0.0), flat)
    with pytest.raises(ValueError):
        cenh.unsharp_mask(flat, 0.0)


def test_robust_average_rejects_direct_hits() -> None:
    """Outliers of one frame are rejected by the median and the clipped mean."""
    rng = np.random.default_rng(3)
    frames = [rng.normal(500.0, 5.0, (40, 40)) for _ in range(8)]
    frames[2][10, 10] = 16000.0
    frames[5][20:22, 30] = 9000.0
    mean = cenh.robust_average(frames, cenh.AVERAGE_MEAN)
    assert mean[10, 10] > 2000.0
    for method in (cenh.AVERAGE_MEDIAN, cenh.AVERAGE_CLIPPED):
        result = cenh.robust_average(frames, method)
        assert abs(result[10, 10] - 500.0) < 15.0
        assert np.all(np.abs(result[20:22, 30] - 500.0) < 15.0)
    clipped = cenh.robust_average(frames)
    assert np.std(clipped) < np.std(cenh.robust_average(frames, cenh.AVERAGE_MEDIAN))
    assert np.allclose(cenh.robust_average(frames[:2]), mean_of(frames[:2]))
    with pytest.raises(ValueError, match="At least one"):
        cenh.robust_average([])
    with pytest.raises(ValueError, match="same shape"):
        cenh.robust_average([frames[0], frames[1][:5]])
    with pytest.raises(ValueError, match="method"):
        cenh.robust_average(frames, "mode")


def mean_of(frames: list[np.ndarray]) -> np.ndarray:
    """Return the plain mean of frames."""
    return np.mean(np.stack(frames), axis=0)


def test_normalize() -> None:
    """Normalization maps a window to [0, 1]."""
    assert np.allclose(
        cenh.normalize(np.array([0.0, 5.0, 20.0]), 0.0, 10.0), [0, 0.5, 1]
    )
    with pytest.raises(ValueError):
        cenh.normalize(np.zeros(3), 1.0, 1.0)


def test_pixel_sizes_follow_the_active_plane() -> None:
    """Physical sizes use the active magnification, else the detector pitch."""
    data = np.ones((20, 30))
    image = magnified_image(data, 0.05)
    assert wenh.pixel_sizes(image, 3.0, wenh.PIXELS) == (3.0, 3.0)
    assert wenh.pixel_sizes(image, 1.0, "mm") == pytest.approx((20.0, 20.0))
    assert wenh.pixel_sizes(image, 50.0, "µm") == pytest.approx((1.0, 1.0))
    image.dx, image.dy = 2.0, 4.0
    assert wenh.pixel_sizes(image, 1.0, "mm") == pytest.approx((10.0, 5.0))
    plain = create_image("detector", data)
    with pytest.raises(mag.MagnificationError, match="pixel pitch"):
        wenh.pixel_sizes(plain, 1.0, "mm")
    objects.set_detector_pitch(plain, 0.1)
    assert wenh.pixel_sizes(plain, 1.0, "mm") == pytest.approx((10.0, 10.0))
    with pytest.raises(ValueError, match="strictly positive"):
        wenh.pixel_sizes(plain, 0.0, wenh.PIXELS)
    nonuniform = create_image("nonuniform", data)
    nonuniform.set_coords(np.arange(30) ** 1.5, np.arange(20.0))
    with pytest.raises(ValueError, match="uniform"):
        wenh.pixel_sizes(nonuniform, 1.0, "mm")


def test_enhancement_computations_on_images() -> None:
    """Enhancement features convert physical sizes and keep the image frame."""
    data, pore = pore_image(noise=8.0)
    image = magnified_image(data, 0.1)
    param = wenh.FlattenBackgroundParam.create(size=1.5, unit="mm")
    flat = computations.flatten_background(image, param)
    assert np.std(flat.data[10:-10, 10:40]) < 0.1 * np.std(data)
    assert objects.get_magnifications(flat).active == "Weld"
    sharp = computations.unsharp_mask(
        flat, wenh.UnsharpMaskParam.create(sigma=0.4, unit="mm", amount=2.0)
    )
    level = np.median(flat.data)
    assert sharp.data[pore].mean() - level > 1.5 * (flat.data[pore].mean() - level)
    clahe = computations.radiographic_clahe(
        image, wenh.RadiographicClaheParam.create(tile_size=4.0, unit="mm")
    )
    assert 0.0 <= np.nanmin(clahe.data) and np.nanmax(clahe.data) <= 1.0
    assert clahe.zunit == ""
    noise = np.std(flat.data[10:-10, 10:40])
    for method in (
        wenh.DENOISE_MEDIAN,
        wenh.DENOISE_GAUSSIAN,
        wenh.DENOISE_TV,
        wenh.DENOISE_BILATERAL,
    ):
        param = wenh.RadiographicDenoiseParam.create(method=method, size=3.0)
        param.weight = 0.05
        denoised = computations.radiographic_denoise(flat, param)
        assert np.std(denoised.data[10:-10, 10:40]) < 0.8 * noise, method
    frames = [magnified_image(data + 10.0 * k, 0.1) for k in range(3)]
    average = computations.robust_average(frames, wenh.RobustAverageParam())
    assert np.allclose(average.data, data + 10.0)
    with pytest.raises(ValueError, match="Select"):
        computations.robust_average([], wenh.RobustAverageParam())
