"""Tests of the magnification operations and of the coordinate conversion."""

from __future__ import annotations

import numpy as np
import pytest
import sigima.params
import sigima.proc.image as sipi
from sigima.objects import create_image, create_image_roi
from sigima.objects.annotations import (
    CircleAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    annotation_from_dict,
    annotation_to_dict,
)

from datalab_radiography.core import magnification as mag
from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import computations, features, objects
from datalab_radiography.workflow import magnification as wmag

from .test_circlefit import projected_object


def _image(pitch: float | None = 0.2):
    """Return a radiograph with an optional detector pitch."""
    image = create_image("radiograph", np.random.default_rng(0).random((100, 120)))
    if pitch is not None:
        objects.set_detector_pitch(image, pitch)
    return image


def _value_param(name: str, factor: float) -> wmag.MagnificationValueParam:
    param = wmag.MagnificationValueParam()
    param.name, param.factor = name, factor
    return param


def test_definitions_need_a_detector_scale() -> None:
    """Value and geometry need the pitch; a known length does not."""
    image = _image(pitch=None)
    with pytest.raises(mag.MagnificationError, match="pitch is unknown"):
        wmag.define_value(image, _value_param("Weld", 2.0))
    length = wmag.MagnificationLengthParam()
    length.name = "Ruler"
    length.x0, length.y0, length.x1, length.y1 = 10.0, 10.0, 60.0, 10.0
    length.true_length, length.unit = 1.0, "cm"
    item = wmag.define_length(image, length)
    assert item.factor is None
    assert item.scale_mm == pytest.approx((0.2, 0.2))
    assert item.details["segment"] == [10.0, 10.0, 60.0, 10.0]
    pitch = wmag.DetectorPitchParam()
    pitch.pitch_x = pitch.pitch_y = 100.0
    pitch.unit = "µm"
    assert wmag.set_detector_pitch(image, pitch) == pytest.approx((0.1, 0.1))
    assert objects.detector_mm_per_unit(image) == pytest.approx((0.1, 0.1))
    param = wmag.detector_pitch_param(image, None)
    assert (param.pitch_x, param.unit) == (pytest.approx(0.1), "mm")


def test_value_geometry_and_management() -> None:
    """Magnifications are added, made active, removed and copied."""
    image = _image()
    wmag.define_value(image, _value_param("Weld", 2.0))
    geometry = wmag.MagnificationGeometryParam()
    geometry.name = "Pipe axis"
    geometry.sdd, geometry.distance_kind, geometry.distance = 100.0, "odd", 20.0
    geometry.focal_spot, geometry.unit = 0.04, "cm"
    item = wmag.define_geometry(image, geometry)
    assert item.factor == pytest.approx(1.25)
    assert item.details["geometric_unsharpness_mm"] == pytest.approx(0.1)
    mset = objects.get_magnifications(image)
    assert mset.names == ("Weld", "Pipe axis") and mset.active == "Pipe axis"
    assert wmag.set_active_magnification(image, "Weld").active == "Weld"
    other = _image()
    assert wmag.copy_magnifications(image, [image, other]) == 1
    assert objects.get_magnifications(other).names == mset.names
    assert objects.get_detector_pitch(other) == pytest.approx((0.2, 0.2))
    other.xunit = "mm"
    with pytest.raises(ValueError, match="coordinate unit"):
        wmag.copy_magnifications(image, [other])
    assert wmag.remove_magnification(image, "Weld").names == ("Pipe axis",)
    texts = wmag.describe_magnification(item, "µm")
    assert texts["factor"] == "1.25" and texts["pixel"] == "160 µm"


def test_ball_definition_from_roi_and_box() -> None:
    """A calibration ball gives the magnification within 0.5 %."""
    data, xc, yc = projected_object(20.0, "sphere", 1.0)
    image = create_image("ball", data)
    objects.set_detector_pitch(image, 0.15)
    param = wmag.MagnificationBallParam()
    param.true_diameter, param.unit, param.roi_index = 2.0, "mm", 1
    with pytest.raises(ValueError, match="no region of interest number 1"):
        wmag.define_ball(image, param)
    image.roi = create_image_roi("circle", [xc, yc, 32.0])
    item, fit = wmag.define_ball(image, param)
    # 40 pixels of 0.15 mm for a 2 mm ball: M = 3
    assert item.factor == pytest.approx(3.0, rel=5e-3)
    assert fit.xc == pytest.approx(xc, abs=0.1)
    assert item.details["shape"] == "sphere"
    box_item, _fit = wmag.define_ball(
        image, param, box=(xc - 30, yc - 30, xc + 30, yc + 30)
    )
    assert box_item.factor == pytest.approx(3.0, rel=5e-3)
    with pytest.raises(ValueError, match="8 × 8"):
        wmag.define_ball(image, param, box=(0, 0, 3, 3))


def test_magnifications_from_metadata() -> None:
    """Geometry metadata and DICOM attributes define magnifications."""
    image = _image(pitch=0.1496)
    with pytest.raises(ValueError, match="no magnification"):
        wmag.define_from_metadata(image)
    objects.set_geometry(
        image,
        {
            "sdd_mm": 199.006188,
            "focal_spot_mm": 0.05,
            "planes": [{"name": "Walnut", "sod_mm": 66.001402}],
        },
    )
    image.metadata[md.DICOM_KEY] = md.dumps(
        {"DistanceSourceToDetector": 1000.0, "DistanceSourceToPatient": 800.0}
    )
    items = wmag.define_from_metadata(image)
    assert [item.name for item in items] == ["Walnut", "DICOM"]
    assert items[0].factor == pytest.approx(3.015181, rel=1e-6)
    assert items[0].method == mag.METHOD_METADATA
    assert items[1].factor == pytest.approx(1.25)


def test_express_coordinates_keeps_shapes_and_magnifications() -> None:
    """Coordinates convert back and forth with ROIs and annotations."""
    image = _image(pitch=0.2)
    wmag.define_value(image, _value_param("Weld", 2.0))
    image.roi = create_image_roi("rectangle", [10, 20, 30, 40])
    image.set_annotations(
        [
            annotation_to_dict(SegmentAnnotation(x0=0, y0=0, x1=50, y1=0)),
            annotation_to_dict(CircleAnnotation(cx=60, cy=50, radius=10)),
        ]
    )
    param = wmag.ExpressCoordinatesParam()
    param.target, param.unit = mag.FRAME_OBJECT, "mm"
    in_mm = computations.express_coordinates(image, param)
    assert (in_mm.dx, in_mm.dy, in_mm.xunit) == (pytest.approx(0.1), 0.1, "mm")
    assert objects.get_coordinate_frame(in_mm) == mag.CoordinateFrame(
        mag.FRAME_OBJECT, "mm", "Weld", 2.0
    )
    assert np.allclose(in_mm.roi.single_rois[0].coords, [1, 2, 3, 4])
    segment = annotation_from_dict(in_mm.get_annotations()[0])
    assert segment.x1 == pytest.approx(5.0)
    assert objects.get_magnifications(in_mm).get("Weld").scale_mm == pytest.approx(
        (1.0, 1.0)
    )
    assert np.array_equal(in_mm.data, image.data)
    param.target = mag.FRAME_PIXEL
    back = computations.express_coordinates(in_mm, param)
    assert back.dx == pytest.approx(1.0) and back.xunit == ""
    assert np.allclose(back.roi.single_rois[0].coords, [10, 20, 30, 40])
    circle = annotation_from_dict(back.get_annotations()[1])
    assert circle.radius == pytest.approx(10.0)
    param.target, param.magnification = mag.FRAME_OBJECT, "Pipe"
    with pytest.raises(KeyError):
        computations.express_coordinates(image, param)


def test_anisotropic_conversion_of_shapes() -> None:
    """Circles become polygons or ellipses with anisotropic factors."""
    image = _image(pitch=None)
    image.roi = create_image_roi("circle", [50, 50, 10])
    image.set_annotations(
        [
            annotation_to_dict(CircleAnnotation(cx=60, cy=50, radius=10)),
            annotation_to_dict(
                RectangleAnnotation(x=10, y=10, width=10, height=5, angle=0.3)
            ),
            {"legacy": "plotpy item"},
        ]
    )
    assert objects.scale_coordinates(image, 2.0, 1.0) == 1
    assert type(image.roi.single_rois[0]).__name__ == "PolygonalROI"
    kinds = [item.get("kind") for item in image.get_annotations()]
    assert kinds == ["ellipse", "polygon", None]
    assert image.dx == pytest.approx(2.0)


def test_magnifications_survive_binning() -> None:
    """Binning that updates the pixel size keeps the magnifications valid."""
    image = _image(pitch=0.2)
    wmag.define_value(image, _value_param("Weld", 2.0))
    binned = sipi.binning(
        image, sigima.params.BinningParam.create(sx=2, sy=2, change_pixel_size=True)
    )
    item = objects.check_magnifications(binned).get("Weld")
    # Two binned pixels span 4 detector pixels: 0.8 mm, 0.4 mm in the object
    assert item.length_mm(2 * binned.dx, 0.0) == pytest.approx(0.4)


def test_geometry_report_and_catalog() -> None:
    """The calculator derives the geometry quantities; the catalog is valid."""
    param = wmag.GeometryCalculatorParam()
    param.sdd, param.sod, param.focal_spot, param.pitch = 1000.0, 500.0, 0.4, 0.1
    report = dict(wmag.geometry_report(param))
    assert report["Magnification M = SDD / SOD"] == "2"
    assert report["Geometric unsharpness Ug = f (M - 1)"] == "0.4 mm"
    assert report["Pixel size in the object plane"] == "0.05 mm"
    assert features.features_for(features.HOST_DESKTOP)
    feature = features.FEATURES[0]
    assert feature.feature_id == "org.datalab.radiography:express-coordinates"
    assert feature.menu_path.startswith("Plugins/X-ray Radiography & NDT/")
    with pytest.raises(ValueError):
        features.check_catalog((feature, feature))
