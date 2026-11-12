"""Measurements of the annotations and ROIs of Sigima images."""

from __future__ import annotations

import math
from collections.abc import Sequence

import guidata.dataset as gds
import numpy as np
from sigima.objects import (
    CircularROI,
    ImageObj,
    PolygonalROI,
    RectangularROI,
    TableResult,
)
from sigima.objects.annotations import (
    CircleAnnotation,
    EllipseAnnotation,
    GraphicalAnnotation,
    PointAnnotation,
    PolygonAnnotation,
    PolylineAnnotation,
    RectangleAnnotation,
    SegmentAnnotation,
    annotation_from_dict,
    is_graphical_annotation_dict,
)

from ..core import magnification as mag
from ..core import measurement as meas
from ..core import units
from . import objects

UNIT_CHOICES = tuple((unit, unit) for unit in units.LENGTH_UNITS_MM)
PLANES_ALL = "all"
PLANES_ACTIVE = "active"
SOURCE_BOTH = "both"
SOURCE_ANNOTATIONS = "annotations"
SOURCE_ROIS = "rois"
SHAPE_TITLES = {
    "point": "Point",
    "segment": "Segment",
    "polyline": "Polyline",
    "polygon": "Polygon",
    "circle": "Circle",
    "ellipse": "Ellipse",
    "rectangle": "Rectangle",
}


class MeasurementReportParam(gds.DataSet):
    """Measurement report"""

    sources = gds.ChoiceItem(
        "Shapes",
        (
            (SOURCE_BOTH, "Annotations and regions of interest"),
            (SOURCE_ANNOTATIONS, "Annotations"),
            (SOURCE_ROIS, "Regions of interest"),
        ),
        default=SOURCE_BOTH,
    )
    planes = gds.ChoiceItem(
        "Magnifications",
        ((PLANES_ALL, "All magnifications"), (PLANES_ACTIVE, "Active magnification")),
        default=PLANES_ALL,
    )
    unit = gds.ChoiceItem("Length unit", UNIT_CHOICES, default=units.DEFAULT_UNIT)


def pixel_value(obj: ImageObj, x: float, y: float) -> float:
    """Return the image value at the nearest pixel of a point (NaN outside)."""
    if obj.is_uniform_coords:
        col = int(round((x - obj.x0) / obj.dx))
        row = int(round((y - obj.y0) / obj.dy))
    else:
        col = int(np.argmin(np.abs(np.asarray(obj.xcoords) - x)))
        row = int(np.argmin(np.abs(np.asarray(obj.ycoords) - y)))
    if 0 <= row < obj.data.shape[0] and 0 <= col < obj.data.shape[1]:
        return float(np.real(obj.data[row, col]))
    return float("nan")


def shape_from_annotation(
    obj: ImageObj, annotation: GraphicalAnnotation, title: str
) -> meas.Shape | None:
    """Return the measured shape of an annotation (None if not measurable)."""
    title = annotation.title or title
    if isinstance(annotation, SegmentAnnotation):
        return meas.Segment(
            title, annotation.x0, annotation.y0, annotation.x1, annotation.y1
        )
    if isinstance(annotation, PolylineAnnotation):
        return meas.Polyline(title, tuple(map(tuple, annotation.points)))
    if isinstance(annotation, PolygonAnnotation):
        return meas.Polygon(title, tuple(map(tuple, annotation.points)))
    if isinstance(annotation, CircleAnnotation):
        return meas.Circle(title, annotation.cx, annotation.cy, annotation.radius)
    if isinstance(annotation, EllipseAnnotation):
        return meas.Ellipse(
            title,
            annotation.cx,
            annotation.cy,
            annotation.radius_x,
            annotation.radius_y,
            annotation.angle,
        )
    if isinstance(annotation, RectangleAnnotation):
        return meas.Rectangle(
            title,
            annotation.x,
            annotation.y,
            annotation.width,
            annotation.height,
            annotation.angle,
        )
    if isinstance(annotation, PointAnnotation):
        return meas.Point(
            title,
            annotation.x,
            annotation.y,
            pixel_value(obj, annotation.x, annotation.y),
        )
    return None


def _numbered(kind: str, counts: dict[str, int], prefix: str = "") -> str:
    """Return a numbered default title for a shape kind."""
    counts[kind] = counts.get(kind, 0) + 1
    return f"{prefix}{SHAPE_TITLES.get(kind, kind.capitalize())} {counts[kind]}"


def shapes_from_annotations(
    obj: ImageObj, annotations: Sequence[GraphicalAnnotation] | None = None
) -> list[meas.Shape]:
    """Return the measurable shapes of annotations (by default, of the image)."""
    if annotations is None:
        annotations = [
            annotation_from_dict(data)
            for data in obj.get_annotations()
            if is_graphical_annotation_dict(data)
        ]
    shapes, counts = [], {}
    for annotation in annotations:
        kind = annotation.KIND.value if annotation.KIND is not None else ""
        shape = shape_from_annotation(obj, annotation, _numbered(kind, counts))
        if shape is not None:
            shapes.append(shape)
    return shapes


def shapes_from_rois(obj: ImageObj) -> list[meas.Shape]:
    """Return the shapes of the regions of interest of an image."""
    if obj.roi is None or obj.roi.is_empty():
        return []
    shapes = []
    for index, roi in enumerate(obj.roi.single_rois, start=1):
        coords = np.asarray(roi.get_physical_coords(obj), dtype=float)
        title = roi.title or f"ROI {index}"
        if isinstance(roi, RectangularROI):
            x0, y0, width, height = coords
            shapes.append(
                meas.Rectangle(title, x0 + width / 2, y0 + height / 2, width, height)
            )
        elif isinstance(roi, CircularROI):
            shapes.append(meas.Circle(title, *coords))
        elif isinstance(roi, PolygonalROI):
            shapes.append(meas.Polygon(title, tuple(map(tuple, coords.reshape(-1, 2)))))
    return shapes


def image_unit(obj: ImageObj) -> str:
    """Return the unit of image coordinates ("px" for pixels)."""
    return obj.xunit or "px"


PLANE_DETECTOR = "Detector"


def plane_named(obj: ImageObj, name: str = "") -> meas.Plane:
    """Return a plane of an image by name.

    Args:
        obj: image
        name: name of a magnification, empty for the active magnification, or
         ``Detector`` for the detector plane

    Raises:
        MagnificationError: if the plane is unknown or its scale is not known
    """
    name = (name or "").strip()
    detector = objects.detector_mm_per_unit(obj)
    if name.lower() == PLANE_DETECTOR.lower():
        if detector is None:
            raise mag.MagnificationError(
                "The detector pixel pitch of the image is not known"
            )
        return meas.Plane(PLANE_DETECTOR, detector, 1.0)
    mset = objects.check_magnifications(obj)
    if name:
        try:
            item = mset.get(name)
        except KeyError as exc:
            known = ", ".join(mset.names) or "none"
            raise mag.MagnificationError(
                f"No magnification named {name!r} (magnifications: {known})"
            ) from exc
    else:
        item = mset.active_item
        if item is None:
            raise mag.MagnificationError(
                "The image has no magnification: define the magnification of "
                f"the plane, or use the {PLANE_DETECTOR!r} plane"
            )
    factor = item.factor
    if factor is None and detector is not None:
        factor = item.with_factor_from(detector).factor
    return meas.Plane(item.name, item.scale_mm, factor)


def planes_for(obj: ImageObj, which: str = PLANES_ALL) -> list[meas.Plane]:
    """Return the measurement planes of an image.

    The first plane gives values in image coordinates; then come the detector
    plane (if its scale is known) and the object planes of the magnifications.
    """
    planes = [meas.Plane(f"Image ({image_unit(obj)})")]
    detector = objects.detector_mm_per_unit(obj)
    if detector is not None:
        planes.append(meas.Plane("Detector", detector, 1.0))
    mset = objects.check_magnifications(obj)
    for item in mset:
        if which == PLANES_ACTIVE and item.name != mset.active:
            continue
        factor = item.factor
        if factor is None and detector is not None:
            factor = item.with_factor_from(detector).factor
        planes.append(meas.Plane(item.name, item.scale_mm, factor))
    return planes


def plane_header(plane: meas.Plane, unit: str) -> str:
    """Return the header of a plane column."""
    if plane.scale is None:
        return plane.title
    factor = "" if plane.factor is None else f", M = {plane.factor:.4g}"
    return f"{plane.title} ({unit}{factor})"


def measure(
    obj: ImageObj,
    shapes: Sequence[meas.Shape],
    which: str = PLANES_ALL,
    unit: str = units.DEFAULT_UNIT,
) -> tuple[list[meas.Plane], list[meas.MeasurementRow]]:
    """Measure shapes on an image in all its planes."""
    planes = planes_for(obj, which)
    return planes, meas.measure_shapes(shapes, planes, units.unit_mm(unit))


def quantity_label(row: meas.MeasurementRow) -> str:
    """Return the quantity label of a row, with its unit when it is not a length."""
    if row.dimension == meas.AREA:
        return f"{row.quantity} (squared unit)"
    if row.dimension == meas.ANGLE:
        return f"{row.quantity} (°)"
    return row.quantity


def measurement_table(
    planes: Sequence[meas.Plane],
    rows: Sequence[meas.MeasurementRow],
    unit: str,
    title: str = "Measurements",
) -> TableResult:
    """Return a Sigima table of measurements."""
    headers = ["Shape", "Quantity"] + [plane_header(plane, unit) for plane in planes]
    data = [
        [row.shape, quantity_label(row)]
        + [
            float(value) if math.isfinite(value) else float("nan")
            for value in row.values
        ]
        for row in rows
    ]
    return TableResult.from_rows(
        title, headers, data, [-1] * len(data), func_name="measurement_report"
    )


def shapes_for_report(obj: ImageObj, sources: str) -> list[meas.Shape]:
    """Return the shapes measured by a report."""
    shapes: list[meas.Shape] = []
    if sources in (SOURCE_BOTH, SOURCE_ANNOTATIONS):
        shapes.extend(shapes_from_annotations(obj))
    if sources in (SOURCE_BOTH, SOURCE_ROIS):
        shapes.extend(shapes_from_rois(obj))
    return shapes


def format_value(value: float, row: meas.MeasurementRow, unit: str) -> str:
    """Format a measured value with its unit."""
    if not math.isfinite(value):
        return "-"
    if row.dimension == meas.ANGLE:
        return f"{value:.4g}°"
    if row.dimension == meas.VALUE:
        return f"{value:.6g}"
    suffix = f" {unit}²" if row.dimension == meas.AREA else f" {unit}"
    return f"{value:.4g}{suffix}"
