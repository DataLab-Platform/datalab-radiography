"""Tests of magnifications and coordinate frames."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sigima.objects import create_image

from datalab_radiography.core import magnification as mag
from datalab_radiography.workflow import objects


def test_magnification_from_distances() -> None:
    """M = SDD / SOD, from SOD or from ODD."""
    assert mag.magnification_from_distances(199.006, 66.0014) == pytest.approx(
        3.015181, rel=1e-5
    )
    assert mag.magnification_from_distances(1000.0, odd_mm=200.0) == pytest.approx(1.25)
    with pytest.raises(mag.MagnificationError):
        mag.magnification_from_distances(100.0, 150.0)
    with pytest.raises(mag.MagnificationError):
        mag.magnification_from_distances(100.0)
    with pytest.raises(mag.MagnificationError):
        mag.magnification_from_distances(100.0, 50.0, 50.0)


def test_unsharpness() -> None:
    """Geometric and total unsharpness, and optimal magnification."""
    assert mag.geometric_unsharpness(0.4, 3.0) == pytest.approx(0.8)
    total = mag.image_unsharpness(0.4, 3.0, 0.1)
    assert total == pytest.approx((0.8**3 + 0.2**3) ** (1 / 3) / 3.0)
    v_opt = mag.optimal_magnification(0.4, 0.1)
    assert v_opt == pytest.approx(1.0 + 0.5**1.5)
    # The optimal magnification minimizes the total unsharpness
    values = [mag.image_unsharpness(0.4, v, 0.1) for v in (v_opt * 0.9, v_opt * 1.1)]
    assert min(values) > mag.image_unsharpness(0.4, v_opt, 0.1)


def test_definition_methods() -> None:
    """Every definition method gives the expected object-plane scale."""
    pitch = (0.1496, 0.1496)
    by_value = mag.from_value("Value", 3.0, pitch)
    assert by_value.scale_mm == pytest.approx((0.1496 / 3.0,) * 2)
    by_geometry = mag.from_geometry(
        "Geometry", 199.006188, pitch, sod_mm=66.001402, focal_spot_mm=0.05
    )
    assert by_geometry.factor == pytest.approx(3.015181, rel=1e-6)
    assert by_geometry.details["geometric_unsharpness_mm"] == pytest.approx(
        0.05 * (by_geometry.factor - 1)
    )
    # A 10 mm object imaged with M = 2.5 on a 0.2 mm pitch spans 125 pixels
    by_length = mag.from_length("Length", (75.0, 100.0), 10.0, (0.2, 0.2))
    assert by_length.factor == pytest.approx(2.5)
    assert by_length.length_mm(75.0, 100.0) == pytest.approx(10.0)
    unknown_pitch = mag.from_length("Length", (3.0, 4.0), 10.0)
    assert unknown_pitch.factor is None
    assert unknown_pitch.scale_mm == pytest.approx((2.0, 2.0))
    assert unknown_pitch.with_factor_from((0.5, 0.5)).factor == pytest.approx(0.25)
    by_diameter = mag.from_diameter("Ball", 40.0, 4.0, (0.2, 0.2), fit_details={"r": 1})
    assert by_diameter.factor == pytest.approx(2.0)
    assert by_diameter.details["r"] == 1
    with pytest.raises(mag.MagnificationError, match="pitch is unknown"):
        mag.from_value("Value", 2.0, None)
    with pytest.raises(mag.MagnificationError):
        mag.from_value("", 2.0, pitch)
    with pytest.raises(mag.MagnificationError):
        mag.from_value("Value", -2.0, pitch)


def test_anisotropic_length() -> None:
    """Lengths along each axis use their own scale."""
    item = mag.Magnification("Plane", (0.1, 0.2))
    assert item.length_mm(3.0, 4.0) == pytest.approx(math.hypot(0.3, 0.8))
    assert item.area_mm2(10.0) == pytest.approx(0.2)
    assert item.rescaled(2.0, 4.0).scale_mm == pytest.approx((0.05, 0.05))


def test_magnification_set() -> None:
    """Sets keep unique names, an active item and survive JSON."""
    first = mag.from_value("Weld", 1.2, (0.1, 0.1))
    second = mag.from_value("Ball", 1.1, (0.1, 0.1))
    mset = mag.MagnificationSet().with_item(first).with_item(second)
    assert mset.names == ("Weld", "Ball") and mset.active == "Weld"
    mset = mset.with_active("Ball")
    assert mset.active_item is second
    renamed = mag.from_value("Ball (fitted)", 1.15, (0.1, 0.1))
    mset = mset.with_item(renamed, replace="Ball")
    assert mset.names == ("Weld", "Ball (fitted)")
    assert mset.active == "Ball (fitted)"
    restored = mag.MagnificationSet.from_json(mset.to_json())
    assert restored == mset
    assert mset.without("Ball (fitted)").active == "Weld"
    with pytest.raises(mag.MagnificationError):
        mag.MagnificationSet((first, first))
    with pytest.raises(KeyError):
        mset.get("Pipe")
    assert len(mag.MagnificationSet.from_json(None)) == 0


def test_coordinate_frames() -> None:
    """Frames give the detector length of a coordinate unit."""
    assert mag.default_frame("", "").frame == mag.FRAME_PIXEL
    detector = mag.default_frame("cm", "cm")
    assert detector == mag.CoordinateFrame(mag.FRAME_DETECTOR, "cm")
    assert detector.detector_mm_per_unit(None) == (10.0, 10.0)
    pixel = mag.CoordinateFrame()
    assert pixel.detector_mm_per_unit(None) is None
    assert pixel.detector_mm_per_unit((0.2, 0.3)) == (0.2, 0.3)
    obj_frame = mag.CoordinateFrame(mag.FRAME_OBJECT, "mm", "Weld", 2.0)
    assert obj_frame.detector_mm_per_unit(None) == (2.0, 2.0)
    assert "object plane 'Weld'" in obj_frame.describe()
    assert mag.CoordinateFrame.from_json(obj_frame.to_json()) == obj_frame
    with pytest.raises(mag.MagnificationError):
        mag.CoordinateFrame(mag.FRAME_DETECTOR)
    with pytest.raises(mag.MagnificationError):
        mag.CoordinateFrame(mag.FRAME_PIXEL, "mm")


def test_express_factors_roundtrip() -> None:
    """Expressing coordinates in a plane and back restores them."""
    pitch = (0.2, 0.2)
    pixel = mag.CoordinateFrame()
    weld = mag.from_value("Weld", 2.0, pitch)
    factors, frame = mag.express_factors(pixel, pitch, mag.FRAME_OBJECT, "mm", weld)
    assert factors == pytest.approx((0.1, 0.1))
    assert frame == mag.CoordinateFrame(mag.FRAME_OBJECT, "mm", "Weld", 2.0)
    weld_in_mm = weld.rescaled(*factors)
    assert weld_in_mm.scale_mm == pytest.approx((1.0, 1.0))
    back, back_frame = mag.express_factors(frame, pitch, mag.FRAME_PIXEL)
    assert back == pytest.approx((10.0, 10.0))
    assert back_frame == pixel
    to_detector, _frame = mag.express_factors(frame, None, mag.FRAME_DETECTOR, "cm")
    assert to_detector == pytest.approx((0.2, 0.2))
    with pytest.raises(mag.MagnificationError, match="pitch is unknown"):
        mag.express_factors(pixel, None, mag.FRAME_DETECTOR, "mm")


def test_image_metadata_roundtrip() -> None:
    """Plugin metadata are read back from Sigima images."""
    image = create_image("radiograph", np.zeros((8, 8), dtype=np.uint16))
    assert objects.get_detector_pitch(image) is None
    assert objects.get_coordinate_frame(image).frame == mag.FRAME_PIXEL
    assert not objects.has_coordinate_frame(image)
    objects.set_detector_pitch(image, 0.1496)
    assert objects.get_detector_pitch(image) == pytest.approx((0.1496, 0.1496))
    assert objects.detector_mm_per_unit(image) == pytest.approx((0.1496, 0.1496))
    mset = mag.MagnificationSet().with_item(mag.from_value("Plane", 3.0, (0.15, 0.15)))
    objects.set_magnifications(image, mset)
    assert objects.check_magnifications(image).names == ("Plane",)
    image.xunit = "mm"
    with pytest.raises(mag.MagnificationError, match="coordinate unit"):
        objects.check_magnifications(image)
    objects.set_magnifications(image, mag.MagnificationSet())
    assert not len(objects.get_magnifications(image))
    objects.set_frame_role(image, "flat")
    assert objects.get_frame_role(image) == "flat"
    with pytest.raises(ValueError):
        objects.set_frame_role(image, "bright")
    objects.set_geometry(image, {"sdd_mm": 1000.0})
    assert objects.get_geometry(image) == {"sdd_mm": 1000.0}
    copy = image.copy()
    assert objects.get_geometry(copy) == {"sdd_mm": 1000.0}
