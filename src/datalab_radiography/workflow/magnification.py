"""Magnification operations on Sigima images, shared by DataLab Desktop and Web.

Each operation reads its inputs from a parameter set, changes the plugin
metadata of the image in place and returns what it defined.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import guidata.dataset as gds
import numpy as np
from sigima.objects import ImageObj

from ..core import circlefit, units
from ..core import magnification as mag
from . import display as wdisplay
from . import objects

UNIT_CHOICES = tuple((unit, unit) for unit in units.LENGTH_UNITS_MM)
DEFAULT_ENTRY_UNIT = "mm"


def _name_item(default: str) -> gds.StringItem:
    """Return the name item of a magnification definition."""
    return gds.StringItem(
        "Name", default=default, notempty=True, help="Name of the object plane"
    )


def _remarks_item() -> gds.StringItem:
    """Return the remarks item of a magnification definition."""
    return gds.StringItem("Remarks", default="")


class DetectorPitchParam(gds.DataSet):
    """Detector pixel pitch"""

    pitch_x = gds.FloatItem("Pitch along X", default=0.1, min=0.0, nonzero=True)
    pitch_y = gds.FloatItem("Pitch along Y", default=0.1, min=0.0, nonzero=True)
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT)


class MagnificationValueParam(gds.DataSet):
    """Magnification from its value"""

    name = _name_item("Object plane")
    factor = gds.FloatItem("Magnification M", default=1.0, min=0.0, nonzero=True)
    remarks = _remarks_item()


_DISTANCE_KIND = gds.ValueProp("sod")


class MagnificationGeometryParam(gds.DataSet):
    """Magnification from the acquisition distances"""

    name = _name_item("Object plane")
    sdd = gds.FloatItem(
        "Source-to-detector distance (SDD)", default=1000.0, min=0.0, nonzero=True
    )
    distance_kind = gds.ChoiceItem(
        "Object position",
        (
            ("sod", "Source-to-object distance (SOD)"),
            ("odd", "Object-to-detector distance (ODD)"),
        ),
        default="sod",
    ).set_prop("display", store=_DISTANCE_KIND)
    distance = gds.FloatItem("Distance", default=800.0, min=0.0)
    focal_spot = gds.FloatItem(
        "Focal spot size",
        default=0.0,
        min=0.0,
        help="Optical focal spot size, to compute the geometric unsharpness "
        "(0: unknown)",
    )
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT)
    remarks = _remarks_item()


class MagnificationLengthParam(gds.DataSet):
    """Magnification from a known length"""

    name = _name_item("Object plane")
    _segment = gds.BeginGroup("Segment, in image coordinates")
    x0 = gds.FloatItem("X<sub>0</sub>", default=0.0)
    y0 = gds.FloatItem("Y<sub>0</sub>", default=0.0).set_pos(col=1)
    x1 = gds.FloatItem("X<sub>1</sub>", default=100.0)
    y1 = gds.FloatItem("Y<sub>1</sub>", default=0.0).set_pos(col=1)
    _segment_end = gds.EndGroup("Segment, in image coordinates")
    true_length = gds.FloatItem("True length", default=10.0, min=0.0, nonzero=True)
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT)
    remarks = _remarks_item()


class MagnificationBallParam(gds.DataSet):
    """Magnification from a calibration ball or disk"""

    name = _name_item("Calibration ball")
    true_diameter = gds.FloatItem("True diameter", default=5.0, min=0.0, nonzero=True)
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT)
    shape = gds.ChoiceItem(
        "Object",
        (
            (circlefit.SHAPE_SPHERE, "Ball (sphere)"),
            (circlefit.SHAPE_DISK, "Disk or cylinder seen end-on"),
            (circlefit.SHAPE_EDGE, "Other round object (edge fit only)"),
        ),
        default=circlefit.SHAPE_SPHERE,
    )
    roi_index = gds.IntItem(
        "Region of interest",
        default=1,
        min=0,
        help="Number of the region of interest around the ball (0: whole image)",
    )
    remarks = _remarks_item()


class MagnificationNameParam(gds.DataSet):
    """Magnification"""

    name = gds.StringItem("Name", default="", notempty=True)


_EXPRESS_TARGET = gds.ValueProp(mag.FRAME_OBJECT)


class ExpressCoordinatesParam(gds.DataSet):
    """Express coordinates in a plane"""

    target = gds.ChoiceItem(
        "Coordinates",
        (
            (mag.FRAME_OBJECT, "Object plane of a magnification"),
            (mag.FRAME_DETECTOR, "Detector plane"),
            (mag.FRAME_PIXEL, "Detector pixels"),
        ),
        default=mag.FRAME_OBJECT,
    ).set_prop("display", store=_EXPRESS_TARGET)
    magnification = gds.StringItem(
        "Magnification",
        default="",
        help="Name of the magnification of the object plane (empty: active one)",
    ).set_prop(
        "display", active=gds.FuncProp(_EXPRESS_TARGET, lambda value: value == "object")
    )
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT).set_prop(
        "display", active=gds.FuncProp(_EXPRESS_TARGET, lambda value: value != "pixel")
    )


class GeometryCalculatorParam(gds.DataSet):
    """Radiographic geometry calculator"""

    sdd = gds.FloatItem(
        "Source-to-detector distance (SDD)", default=1000.0, min=0.0, nonzero=True
    )
    sod = gds.FloatItem(
        "Source-to-object distance (SOD)", default=800.0, min=0.0, nonzero=True
    )
    focal_spot = gds.FloatItem("Focal spot size", default=0.4, min=0.0, nonzero=True)
    pitch = gds.FloatItem("Detector pixel pitch", default=0.1, min=0.0, nonzero=True)
    detector_srb = gds.FloatItem(
        "Detector basic spatial resolution (SRb)",
        default=0.0,
        min=0.0,
        help="0: one pixel pitch",
    )
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default=DEFAULT_ENTRY_UNIT)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _to_mm(value: float, unit: str) -> float:
    """Convert a length entered in a unit to millimeters."""
    return units.to_mm(value, unit)


def set_detector_pitch(obj: ImageObj, param: DetectorPitchParam) -> mag.Pair:
    """Set the detector pixel pitch of an image."""
    pitch = (_to_mm(param.pitch_x, param.unit), _to_mm(param.pitch_y, param.unit))
    objects.set_detector_pitch(obj, pitch)
    return pitch


def detector_pitch_param(obj: ImageObj, default_mm: float | None) -> DetectorPitchParam:
    """Return a pitch parameter set filled with the pitch of an image."""
    param = DetectorPitchParam()
    pitch = objects.get_detector_pitch(obj)
    if pitch is None and default_mm:
        pitch = (default_mm, default_mm)
    if pitch is not None:
        param.pitch_x, param.pitch_y = pitch
    param.unit = "mm"
    return param


def add_magnification(
    obj: ImageObj,
    item: mag.Magnification,
    make_active: bool = True,
    replace: str | None = None,
) -> mag.MagnificationSet:
    """Add a magnification to an image (replacing one with the same name)."""
    mset = objects.check_magnifications(obj).with_item(item, make_active, replace)
    objects.set_magnifications(obj, mset)
    return mset


def define_value(obj: ImageObj, param: MagnificationValueParam) -> mag.Magnification:
    """Define a magnification of an image from its value."""
    item = mag.from_value(
        param.name, param.factor, objects.detector_mm_per_unit(obj), param.remarks
    )
    add_magnification(obj, item)
    return item


def define_geometry(
    obj: ImageObj, param: MagnificationGeometryParam
) -> mag.Magnification:
    """Define a magnification of an image from the acquisition distances."""
    distance = _to_mm(param.distance, param.unit)
    item = mag.from_geometry(
        param.name,
        _to_mm(param.sdd, param.unit),
        objects.detector_mm_per_unit(obj),
        sod_mm=distance if param.distance_kind == "sod" else None,
        odd_mm=distance if param.distance_kind == "odd" else None,
        focal_spot_mm=_to_mm(param.focal_spot, param.unit) or None,
        remarks=param.remarks,
    )
    add_magnification(obj, item)
    return item


def define_length(obj: ImageObj, param: MagnificationLengthParam) -> mag.Magnification:
    """Define a magnification of an image from a segment of known length."""
    item = mag.from_length(
        param.name,
        (param.x1 - param.x0, param.y1 - param.y0),
        _to_mm(param.true_length, param.unit),
        objects.detector_mm_per_unit(obj),
        param.remarks,
    )
    item = dataclasses.replace(
        item,
        details={**item.details, "segment": [param.x0, param.y0, param.x1, param.y1]},
    )
    add_magnification(obj, item)
    return item


def region_of(
    obj: ImageObj, roi_index: int
) -> tuple[np.ndarray, float, float, float, float]:
    """Return the data and coordinates of the bounding box of a ROI.

    Args:
        obj: image with uniform coordinates
        roi_index: number of the ROI (1 for the first one), 0 for the whole image

    Returns:
        Data of the region, X and Y of its first pixel, and the pixel spacing

    Raises:
        ValueError: if the image coordinates are not uniform or the ROI is unknown
    """
    if not obj.is_uniform_coords:
        raise ValueError("The image must have uniform coordinates")
    if roi_index == 0:
        return obj.data, obj.x0, obj.y0, obj.dx, obj.dy
    rois = [] if obj.roi is None else list(obj.roi.single_rois)
    if not 1 <= roi_index <= len(rois):
        raise ValueError(
            f"The image has no region of interest number {roi_index} "
            f"({len(rois)} defined)"
        )
    mask = ~rois[roi_index - 1].to_mask(obj)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        raise ValueError("The region of interest is empty")
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    return (
        obj.data[r0:r1, c0:c1],
        obj.x0 + c0 * obj.dx,
        obj.y0 + r0 * obj.dy,
        obj.dx,
        obj.dy,
    )


def region_from_box(
    obj: ImageObj, box: tuple[float, float, float, float]
) -> tuple[np.ndarray, float, float, float, float]:
    """Return the data and coordinates of a box given in image coordinates.

    Args:
        obj: image with uniform coordinates
        box: X and Y of two opposite corners of the box

    Raises:
        ValueError: if the image coordinates are not uniform or the box is empty
    """
    if not obj.is_uniform_coords:
        raise ValueError("The image must have uniform coordinates")
    xa, ya, xb, yb = box
    cols = sorted(int(round((x - obj.x0) / obj.dx)) for x in (xa, xb))
    rows = sorted(int(round((y - obj.y0) / obj.dy)) for y in (ya, yb))
    c0, c1 = max(cols[0], 0), min(cols[1] + 1, obj.data.shape[1])
    r0, r1 = max(rows[0], 0), min(rows[1] + 1, obj.data.shape[0])
    if c1 - c0 < 8 or r1 - r0 < 8:
        raise ValueError("The region must cover at least 8 × 8 pixels of the image")
    return (
        obj.data[r0:r1, c0:c1],
        obj.x0 + c0 * obj.dx,
        obj.y0 + r0 * obj.dy,
        obj.dx,
        obj.dy,
    )


def define_ball(
    obj: ImageObj,
    param: MagnificationBallParam,
    box: tuple[float, float, float, float] | None = None,
) -> tuple[mag.Magnification, circlefit.CircleFit]:
    """Define a magnification of an image from a calibration ball or disk.

    Args:
        obj: image
        param: definition parameters
        box: region around the ball (two opposite corners, in image
         coordinates); by default, the region of interest of ``param``

    Raises:
        ValueError: if the region is invalid or no circle can be fitted
    """
    if box is None:
        data, x0, y0, dx, dy = region_of(obj, param.roi_index)
    else:
        data, x0, y0, dx, dy = region_from_box(obj, box)
    fit = circlefit.fit_circle(data, x0, y0, dx, dy, shape=param.shape)
    details = {
        "center": [fit.xc, fit.yc],
        "rms_residual": fit.rms_residual,
        "shape": param.shape,
        "roi_index": param.roi_index,
    }
    item = mag.from_diameter(
        param.name,
        fit.diameter,
        _to_mm(param.true_diameter, param.unit),
        objects.detector_mm_per_unit(obj),
        param.remarks,
        details,
    )
    add_magnification(obj, item)
    return item, fit


def magnifications_from_metadata(obj: ImageObj) -> list[mag.Magnification]:
    """Return the magnifications given by the acquisition metadata of an image.

    The plugin geometry metadata (examples, simulator) give the distances; the
    DICOM attributes give an estimated magnification or the distances.
    """
    detector = objects.detector_mm_per_unit(obj)
    items: list[mag.Magnification] = []
    geometry = objects.get_geometry(obj)
    planes = geometry.get("planes") or []
    if not planes and geometry.get("sod_mm"):
        planes = [{"name": "Acquisition", "sod_mm": geometry["sod_mm"]}]
    if geometry.get("sdd_mm") and detector is not None:
        for plane in planes:
            item = mag.from_geometry(
                plane["name"],
                geometry["sdd_mm"],
                detector,
                sod_mm=plane["sod_mm"],
                focal_spot_mm=geometry.get("focal_spot_mm"),
                remarks="From the acquisition metadata",
            )
            items.append(dataclasses.replace(item, method=mag.METHOD_METADATA))
    attrs = wdisplay.get_dicom_attributes(obj)
    if attrs and detector is not None:
        factor = attrs.get("EstimatedRadiographicMagnificationFactor")
        sdd, sod = (
            attrs.get("DistanceSourceToDetector"),
            attrs.get("DistanceSourceToPatient"),
        )
        details: dict[str, Any] = {"source": "DICOM"}
        if not factor and sdd and sod and 0 < float(sod) <= float(sdd):
            factor = float(sdd) / float(sod)
            details.update(sdd_mm=float(sdd), sod_mm=float(sod))
        if factor and float(factor) > 0:
            item = mag.from_value("DICOM", float(factor), detector, "From DICOM")
            items.append(
                dataclasses.replace(item, method=mag.METHOD_METADATA, details=details)
            )
    return items


def define_from_metadata(obj: ImageObj) -> list[mag.Magnification]:
    """Add the magnifications given by the acquisition metadata of an image.

    Raises:
        ValueError: if the metadata give no magnification
    """
    items = magnifications_from_metadata(obj)
    if not items:
        raise ValueError(
            "The image metadata give no magnification (or the detector pixel "
            "pitch is unknown)"
        )
    for item in items:
        add_magnification(obj, item, make_active=False)
    return items


def remove_magnification(obj: ImageObj, name: str) -> mag.MagnificationSet:
    """Remove a magnification of an image."""
    mset = objects.check_magnifications(obj).without(name)
    objects.set_magnifications(obj, mset)
    return mset


def set_active_magnification(obj: ImageObj, name: str) -> mag.MagnificationSet:
    """Set the active magnification of an image."""
    mset = objects.check_magnifications(obj).with_active(name)
    objects.set_magnifications(obj, mset)
    return mset


def copy_magnifications(source: ImageObj, targets: Sequence[ImageObj]) -> int:
    """Copy the magnifications, frame and pitch of an image to other images.

    The target images must have the same coordinate unit.

    Returns:
        Number of images changed

    Raises:
        ValueError: if a target image has another coordinate unit
    """
    mset = objects.check_magnifications(source)
    pitch = objects.get_detector_pitch(source)
    frame = objects.get_coordinate_frame(source)
    changed = 0
    for target in targets:
        if target is source:
            continue
        if (target.xunit or "") != (source.xunit or ""):
            raise ValueError(
                f"Image '{target.title}' has another coordinate unit than "
                f"'{source.title}'"
            )
        objects.set_detector_pitch(target, pitch)
        objects.set_coordinate_frame(target, frame)
        objects.set_magnifications(target, mset)
        changed += 1
    return changed


def geometry_report(param: GeometryCalculatorParam) -> list[tuple[str, str]]:
    """Return the quantities derived from an acquisition geometry.

    Returns:
        List of (quantity, formatted value) pairs
    """
    unit = param.unit
    sdd, sod = _to_mm(param.sdd, unit), _to_mm(param.sod, unit)
    focal = _to_mm(param.focal_spot, unit)
    pitch = _to_mm(param.pitch, unit)
    srb = _to_mm(param.detector_srb, unit) if param.detector_srb > 0 else pitch
    factor = mag.magnification_from_distances(sdd, sod)
    ug = mag.geometric_unsharpness(focal, factor)
    total = mag.image_unsharpness(focal, factor, srb)
    optimal = mag.optimal_magnification(focal, srb)

    def length(value_mm: float) -> str:
        return units.format_length(value_mm, unit)

    return [
        ("Magnification M = SDD / SOD", f"{factor:.4g}"),
        ("Object-to-detector distance", length(sdd - sod)),
        ("Pixel size in the object plane", length(pitch / factor)),
        ("Geometric unsharpness Ug = f (M - 1)", length(ug)),
        ("Detector unsharpness (2 SRb)", length(2.0 * srb)),
        ("Total unsharpness in the object plane", length(total)),
        ("Optimal magnification", f"{optimal:.4g}"),
        (
            "SOD at the optimal magnification",
            length(sdd / optimal) if optimal >= 1.0 else "-",
        ),
    ]


def describe_magnification(item: mag.Magnification, unit: str) -> dict[str, str]:
    """Return the display texts of a magnification."""
    factor = "-" if item.factor is None else f"{item.factor:.5g}"
    return {
        "name": item.name,
        "method": mag.METHOD_TITLES.get(item.method, item.method),
        "factor": factor,
        "pixel": units.format_length(item.object_pixel_mm, unit),
        "remarks": item.remarks,
    }
