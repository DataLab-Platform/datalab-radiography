"""Tests of the radiographic display (window, inversion, profile)."""

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects import create_image, create_image_roi

from datalab_radiography.core import display
from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import display as wdisplay
from datalab_radiography.workflow.parameters import WindowParam, apply_window_param


def _radiograph(seed: int = 0) -> np.ndarray:
    """Return a 12-bit radiograph with a saturated free-beam region."""
    rng = np.random.default_rng(seed)
    data = rng.normal(1500.0, 50.0, (200, 300))
    data[:, 250:] = 4095.0  # free beam, clipped by the detector
    data[80:120, 50:100] = 600.0  # dense part
    return np.clip(data, 0, 4095).astype(np.uint16)


def test_window_validation() -> None:
    """Windows need finite, increasing bounds."""
    window = display.Window.from_center_width(100.0, 50.0)
    assert (window.low, window.high) == (75.0, 125.0)
    assert (window.center, window.width) == (100.0, 50.0)
    for low, high in ((1.0, 1.0), (2.0, 1.0), (np.nan, 1.0)):
        with pytest.raises(ValueError):
            display.Window(low, high)
    with pytest.raises(ValueError):
        display.Window.from_center_width(0.0, 0.0)


def test_percentile_window_ignores_saturation() -> None:
    """The free beam does not compress the window."""
    data = _radiograph()
    window = display.percentile_window(data, 0.5, 99.5)
    assert window.high < 2000.0
    assert window.low <= 600.0
    with_saturation = display.percentile_window(data, exclude_saturation=False)
    assert with_saturation.high == pytest.approx(4095.0)
    mask = np.zeros(data.shape, dtype=bool)
    mask[:60, :200] = True
    in_mask = display.percentile_window(data, 1.0, 99.0, mask=mask)
    assert 1300.0 < in_mask.low < in_mask.high < 1700.0
    flat = np.full((10, 10), 7.0)
    assert (
        display.percentile_window(flat).low < 7.0 < display.percentile_window(flat).high
    )
    nan_data = np.full((4, 4), np.nan)
    with pytest.raises(ValueError):
        display.percentile_window(nan_data)
    with pytest.raises(ValueError):
        display.percentile_window(data, 60.0, 40.0)


def test_percentile_window_samples_large_images() -> None:
    """Large images are sampled, with a result close to the full computation."""
    data = np.random.default_rng(1).normal(0.0, 1.0, 3_000_000)
    window = display.percentile_window(data, 1.0, 99.0, exclude_saturation=False)
    assert window.low == pytest.approx(-2.326, abs=0.02)
    assert window.high == pytest.approx(2.326, abs=0.02)


def test_dicom_window_and_inversion() -> None:
    """DICOM windows and photometric interpretations are honored."""
    window = display.window_from_dicom(511.0, 1023.0)
    assert (window.low, window.high) == (-0.5, 1021.5)
    rescaled = display.window_from_dicom(1000.0, 201.0, slope=2.0, intercept=-100.0)
    assert rescaled.low == pytest.approx((1000 - 0.5 - 100 + 100) / 2.0)
    assert display.dicom_inversion("MONOCHROME1") is True
    assert display.dicom_inversion("MONOCHROME1", "INVERSE") is True
    assert display.dicom_inversion("MONOCHROME2") is False
    assert display.dicom_inversion("MONOCHROME2", "INVERSE") is True
    assert display.dicom_inversion("RGB") is None
    assert display.display_inversion(True) is True
    assert display.display_inversion(False) is False
    assert display.display_inversion(True, "MONOCHROME2") is False
    assert display.display_inversion(False, "MONOCHROME1") is True


def test_apply_profile_on_raw_image() -> None:
    """The profile sets gray scale, inversion and window, not the data."""
    data = _radiograph()
    image = create_image("raw", data.copy())
    profile = wdisplay.apply_profile(image, film_convention=True)
    assert np.array_equal(image.data, data)
    assert image.get_metadata_option("colormap") == "gray"
    assert image.get_metadata_option("invert_colormap") is True
    assert profile.window_source == wdisplay.WINDOW_AUTO
    assert (image.zscalemin, image.zscalemax) == (
        profile.window.low,
        profile.window.high,
    )
    assert wdisplay.has_profile(image)
    stored = md.loads(image.metadata[md.DISPLAY_PROFILE_KEY])
    assert stored["colormap"] == "gray" and stored["invert"] is True


def test_apply_profile_on_dicom_image() -> None:
    """DICOM images use their window and inversion, whatever the convention."""
    image = create_image("dicom", _radiograph())
    image.metadata[md.DICOM_KEY] = md.dumps(
        {
            "PhotometricInterpretation": "MONOCHROME1",
            "WindowCenter": 2047.0,
            "WindowWidth": 4096.0,
        }
    )
    profile = wdisplay.apply_profile(image, film_convention=False)
    assert profile.invert is True
    assert profile.window_source == wdisplay.WINDOW_DICOM
    assert profile.window.low == pytest.approx(-1.0)


def test_window_parameters_and_roi() -> None:
    """Window parameters apply percentiles, ROIs, DICOM or level and width."""
    image = create_image("roi", _radiograph())
    wdisplay.apply_profile(image)
    param = WindowParam()
    param.method = wdisplay.WINDOW_MANUAL
    param.level, param.width = 1000.0, 400.0
    assert apply_window_param(image, param) == display.Window(800.0, 1200.0)
    assert md.loads(image.metadata[md.DISPLAY_PROFILE_KEY])["window_source"] == (
        wdisplay.WINDOW_MANUAL
    )
    param.method = wdisplay.WINDOW_ROI
    with pytest.raises(ValueError, match="region of interest"):
        apply_window_param(image, param)
    image.roi = create_image_roi("rectangle", [55, 85, 40, 30])
    window = apply_window_param(image, param)
    assert window.low < 600.0 < window.high < 700.0
    param.method = wdisplay.WINDOW_DICOM
    with pytest.raises(ValueError, match="DICOM"):
        apply_window_param(image, param)


def test_explicit_display_rule() -> None:
    """Default display options of the host are not explicit settings."""
    image = create_image("new", _radiograph())
    assert not wdisplay.is_explicit_display(image, False, "viridis")
    image.set_metadata_option("colormap", "viridis")
    assert not wdisplay.is_explicit_display(image, True, "viridis")
    assert wdisplay.is_explicit_display(image, False, "viridis")
    image.set_metadata_option("colormap", "jet")
    assert wdisplay.is_explicit_display(image, True, "viridis")


def test_processing_results_inherit_the_display() -> None:
    """Results of the plugin features keep the radiographic display."""
    from datalab_radiography.workflow import computations
    from datalab_radiography.workflow import enhancement as wenh

    image = create_image("radiograph", _radiograph())
    wdisplay.apply_profile(image, film_convention=True)
    result = computations.flatten_background(
        image, wenh.FlattenBackgroundParam.create(size=20.0, unit=wenh.PIXELS)
    )
    assert result.get_metadata_option("colormap") == wdisplay.COLORMAP
    assert result.get_metadata_option("invert_colormap") is True
    assert result.zscalemin is not None and result.zscalemax > result.zscalemin
    plain = create_image("plain", _radiograph())
    other = computations.flatten_background(
        plain, wenh.FlattenBackgroundParam.create(size=20.0, unit=wenh.PIXELS)
    )
    assert "__colormap" not in other.metadata and other.zscalemin is None
