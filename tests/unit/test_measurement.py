"""Tests of the measurements in several planes."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sigima.objects import create_image, create_image_roi
from sigima.objects.annotations import (
    PointAnnotation,
    PolylineAnnotation,
    SegmentAnnotation,
    annotation_to_dict,
)

from datalab_radiography.core import measurement as meas
from datalab_radiography.workflow import computations, objects
from datalab_radiography.workflow import magnification as wmag
from datalab_radiography.workflow import measurements as wmeas


def _values(shape: meas.Shape, scale=(1.0, 1.0)) -> dict[str, float]:
    return {name: value for name, _dim, value in shape.measure(scale)}


def test_shapes_with_isotropic_scale() -> None:
    """Shapes give their usual lengths, areas and angles."""
    assert _values(meas.Segment("s", 0, 0, 3, 4))["Length"] == pytest.approx(5.0)
    polyline = _values(meas.Polyline("p", ((1, 0), (0, 0), (0, 2))), (2.0, 2.0))
    assert polyline["Length"] == pytest.approx(6.0)
    assert polyline["Angle"] == pytest.approx(90.0)
    square = _values(meas.Polygon("q", ((0, 0), (2, 0), (2, 2), (0, 2))))
    assert (square["Perimeter"], square["Area"]) == pytest.approx((8.0, 4.0))
    circle = _values(meas.Circle("c", 5, 5, 3), (0.5, 0.5))
    assert circle["Diameter"] == pytest.approx(3.0)
    assert circle["Area"] == pytest.approx(math.pi * 2.25)
    rectangle = _values(meas.Rectangle("r", 0, 0, 4, 2, math.radians(30)))
    assert (rectangle["Width"], rectangle["Height"]) == pytest.approx((4.0, 2.0))
    point = _values(meas.Point("pt", 2, 3, 42.0), (0.1, 0.1))
    assert (point["X"], point["Y"], point["Value"]) == pytest.approx((0.2, 0.3, 42.0))


def test_shapes_with_anisotropic_scale() -> None:
    """Anisotropic scales are applied along each axis."""
    scale = (1.0, 2.0)
    assert _values(meas.Segment("s", 0, 0, 0, 3), scale)["Length"] == pytest.approx(6)
    ellipse = _values(meas.Circle("c", 0, 0, 1), scale)
    assert ellipse["Major axis"] == pytest.approx(4.0)
    assert ellipse["Minor axis"] == pytest.approx(2.0)
    assert ellipse["Area"] == pytest.approx(2.0 * math.pi)
    rotated = _values(meas.Ellipse("e", 0, 0, 2, 1, math.pi / 2), scale)
    assert rotated["Major axis"] == pytest.approx(8.0)
    rectangle = _values(meas.Rectangle("r", 0, 0, 3, 1, math.pi / 2), scale)
    assert (rectangle["Width"], rectangle["Height"]) == pytest.approx((6.0, 1.0))


def test_measure_shapes_in_planes() -> None:
    """Rows hold one value per plane, in the display unit."""
    planes = [
        meas.Plane("Image (px)"),
        meas.Plane("Detector", (0.2, 0.2), 1.0),
        meas.Plane("Weld", (0.1, 0.1), 2.0),
    ]
    rows = meas.measure_shapes(
        [meas.Segment("Segment 1", 0, 0, 100, 0), meas.Point("P", 1, 1, 7.0)],
        planes,
        unit_mm=10.0,
    )
    assert rows[0].values == pytest.approx((100.0, 2.0, 1.0))
    assert rows[0].quantity == "Length" and rows[0].kind == "segment"
    value_row = [row for row in rows if row.quantity == "Value"][0]
    assert value_row.values[0] == 7.0 and math.isnan(value_row.values[1])


def _calibrated_image():
    """Return an image with a 0.2 mm pitch and a magnification of 2."""
    image = create_image(
        "radiograph", np.arange(200 * 200, dtype=float).reshape(200, 200)
    )
    objects.set_detector_pitch(image, 0.2)
    param = wmag.MagnificationValueParam()
    param.name, param.factor = "Weld", 2.0
    wmag.define_value(image, param)
    return image


def test_shapes_of_annotations_and_rois() -> None:
    """Annotations and ROIs become measured shapes with default titles."""
    image = _calibrated_image()
    image.set_annotations(
        [
            annotation_to_dict(SegmentAnnotation(x0=0, y0=0, x1=100, y1=0)),
            annotation_to_dict(SegmentAnnotation(x0=0, y0=0, x1=0, y1=50, title="Gap")),
            annotation_to_dict(PolylineAnnotation(points=((0, 0), (10, 0), (10, 10)))),
            annotation_to_dict(PointAnnotation(x=3, y=2)),
        ]
    )
    shapes = wmeas.shapes_from_annotations(image)
    assert [shape.title for shape in shapes] == [
        "Segment 1",
        "Gap",
        "Polyline 1",
        "Point 1",
    ]
    assert shapes[-1].value == pytest.approx(2 * 200 + 3)
    image.roi = create_image_roi("rectangle", [10, 20, 30, 40])
    image.roi.add_roi(create_image_roi("circle", [50, 50, 10]).single_rois[0])
    rois = wmeas.shapes_from_rois(image)
    assert [type(shape).__name__ for shape in rois] == ["Rectangle", "Circle"]
    assert rois[0].x == pytest.approx(25.0) and rois[0].width == pytest.approx(30.0)
    assert wmeas.pixel_value(image, 1000, 1000) != wmeas.pixel_value(image, 1, 1)


def test_measurement_report_function() -> None:
    """The report gives values in every plane of the image."""
    image = _calibrated_image()
    param = wmeas.MeasurementReportParam()
    assert computations.measurement_report(image, param) is None
    image.set_annotations(
        [annotation_to_dict(SegmentAnnotation(x0=0, y0=0, x1=100, y1=0))]
    )
    image.roi = create_image_roi("circle", [50, 50, 10])
    param.unit = "mm"
    table = computations.measurement_report(image, param)
    assert table.headers == [
        "Shape",
        "Quantity",
        "Image (px)",
        "Detector (mm, M = 1)",
        "Weld (mm, M = 2)",
    ]
    assert table.data[0][:2] == ["Segment 1", "Length"]
    assert table.data[0][2:] == pytest.approx([100.0, 20.0, 10.0])
    diameter = [row for row in table.data if row[1] == "Diameter"][0]
    assert diameter[2:] == pytest.approx([20.0, 4.0, 2.0])
    area = [row for row in table.data if row[1].startswith("Area")][0]
    assert area[1] == "Area (squared unit)"
    assert area[4] == pytest.approx(math.pi)
    param.sources, param.planes = wmeas.SOURCE_ANNOTATIONS, wmeas.PLANES_ACTIVE
    single = computations.measurement_report(image, param)
    assert len(single.data) == 1 and len(single.headers) == 5


def test_format_value() -> None:
    """Values are formatted with their unit."""
    row = meas.MeasurementRow("s", "segment", "Length", meas.LENGTH, (1.0,))
    assert wmeas.format_value(1.23456, row, "cm") == "1.235 cm"
    area = meas.MeasurementRow("c", "circle", "Area", meas.AREA, (1.0,))
    assert wmeas.format_value(2.0, area, "mm") == "2 mm²"
    angle = meas.MeasurementRow("p", "polyline", "Angle", meas.ANGLE, (1.0,))
    assert wmeas.format_value(90.0, angle, "mm") == "90°"
    assert wmeas.format_value(float("nan"), row, "cm") == "-"
