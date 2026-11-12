"""Read and write the plugin metadata of Sigima images."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from sigima.objects import (
    CircularROI,
    ImageObj,
    ImageROI,
    RectangularROI,
    create_image_roi,
)
from sigima.objects.annotations import (
    AnnotationTransformError,
    PolygonAnnotation,
    RectangleAnnotation,
    annotation_from_dict,
    annotation_to_dict,
    is_graphical_annotation_dict,
    scale_annotation,
)

from ..core import magnification as mag
from ..core import metadata as md
from ..core.metadata import MetadataSchemaError

Pair = tuple[float, float]


def _get_json(obj: ImageObj, key: str) -> dict[str, Any] | None:
    """Return a structured plugin metadata value, or None if absent."""
    return md.loads(obj.metadata.get(key))


def get_detector_pitch(obj: ImageObj) -> Pair | None:
    """Return the detector pixel pitch (X, Y) in mm, if known."""
    value = obj.metadata.get(md.DETECTOR_PITCH_KEY)
    if value is None:
        return None
    if isinstance(value, str):
        value = json.loads(value)
    pitch = tuple(float(item) for item in value)
    if len(pitch) != 2 or min(pitch) <= 0.0:
        raise MetadataSchemaError("Invalid detector pitch metadata")
    return pitch


def set_detector_pitch(obj: ImageObj, pitch_mm: Pair | float | None) -> None:
    """Set (or remove, with None) the detector pixel pitch, in mm."""
    if pitch_mm is None:
        obj.metadata.pop(md.DETECTOR_PITCH_KEY, None)
        return
    if isinstance(pitch_mm, (int, float)):
        pitch_mm = (pitch_mm, pitch_mm)
    pitch = mag.positive_pair(pitch_mm, "Detector pitch")
    obj.metadata[md.DETECTOR_PITCH_KEY] = json.dumps(list(pitch))


def get_coordinate_frame(obj: ImageObj) -> mag.CoordinateFrame:
    """Return the coordinate frame of an image.

    Without plugin metadata, the frame is guessed from the axis units.
    """
    data = _get_json(obj, md.COORDINATE_FRAME_KEY)
    if data is None:
        return mag.default_frame(obj.xunit, obj.yunit)
    return mag.CoordinateFrame.from_dict(data)


def has_coordinate_frame(obj: ImageObj) -> bool:
    """Return True if the coordinate frame was set explicitly."""
    return md.COORDINATE_FRAME_KEY in obj.metadata


def set_coordinate_frame(obj: ImageObj, frame: mag.CoordinateFrame) -> None:
    """Set the coordinate frame of an image."""
    obj.metadata[md.COORDINATE_FRAME_KEY] = frame.to_json()


def detector_mm_per_unit(obj: ImageObj) -> Pair | None:
    """Return the detector-plane length of one coordinate unit, if known."""
    return get_coordinate_frame(obj).detector_mm_per_unit(get_detector_pitch(obj))


def get_magnifications(obj: ImageObj) -> mag.MagnificationSet:
    """Return the magnifications of an image (empty set if none)."""
    return mag.MagnificationSet.from_json(obj.metadata.get(md.MAGNIFICATIONS_KEY))


def set_magnifications(obj: ImageObj, magnifications: mag.MagnificationSet) -> None:
    """Set the magnifications of an image (removes them if the set is empty)."""
    if not len(magnifications):
        obj.metadata.pop(md.MAGNIFICATIONS_KEY, None)
        return
    magnifications = mag.MagnificationSet(
        magnifications.items, magnifications.active, obj.xunit or ""
    )
    obj.metadata[md.MAGNIFICATIONS_KEY] = magnifications.to_json()


def check_magnifications(obj: ImageObj) -> mag.MagnificationSet:
    """Return the magnifications after checking that they are still valid.

    Raises:
        MagnificationError: if the image coordinate unit changed since the
         magnifications were defined (outside the plugin)
    """
    magnifications = get_magnifications(obj)
    if len(magnifications) and magnifications.coordinate_unit != (obj.xunit or ""):
        raise mag.MagnificationError(
            "The coordinate unit of the image changed since its magnifications "
            f"were defined ({magnifications.coordinate_unit or 'none'!r} -> "
            f"{obj.xunit or 'none'!r}): define the magnifications again"
        )
    return magnifications


def get_geometry(obj: ImageObj) -> dict[str, Any]:
    """Return the acquisition geometry of an image (empty if unknown)."""
    return _get_json(obj, md.GEOMETRY_KEY) or {}


def set_geometry(obj: ImageObj, geometry: dict[str, Any]) -> None:
    """Set the acquisition geometry of an image."""
    obj.metadata[md.GEOMETRY_KEY] = md.dumps(geometry)


def get_frame_role(obj: ImageObj) -> str | None:
    """Return the role of an image in a correction campaign, if set."""
    role = obj.metadata.get(md.FRAME_ROLE_KEY)
    return role if role in md.FRAME_ROLES else None


def set_frame_role(obj: ImageObj, role: str) -> None:
    """Set the role of an image in a correction campaign."""
    if role not in md.FRAME_ROLES:
        raise ValueError(f"Frame role must be one of {md.FRAME_ROLES}")
    obj.metadata[md.FRAME_ROLE_KEY] = role


def _circle_polygon(xc: float, yc: float, rx: float, ry: float) -> np.ndarray:
    """Return the vertices of an ellipse approximated by a 64-gon."""
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    return np.column_stack([xc + rx * np.cos(angles), yc + ry * np.sin(angles)])


def _scaled_roi(obj: ImageObj, fx: float, fy: float) -> ImageROI | None:
    """Return the ROIs of an image after its coordinates are multiplied."""
    if obj.roi is None or obj.roi.is_empty():
        return obj.roi
    scaled = ImageROI()
    for roi in obj.roi.single_rois:
        if roi.indices:
            # Pixel-index ROIs keep their pixels
            scaled.add_roi(roi)
            continue
        coords = np.asarray(roi.coords, dtype=float)
        if isinstance(roi, RectangularROI):
            new = create_image_roi(
                "rectangle",
                coords * [fx, fy, fx, fy],
                title=roi.title,
                inverse=roi.inverse,
            )
        elif isinstance(roi, CircularROI):
            xc, yc, radius = coords
            if np.isclose(fx, fy):
                new = create_image_roi(
                    "circle",
                    [xc * fx, yc * fy, radius * fx],
                    title=roi.title,
                    inverse=roi.inverse,
                )
            else:
                polygon = _circle_polygon(xc * fx, yc * fy, radius * fx, radius * fy)
                new = create_image_roi(
                    "polygon", polygon.ravel(), title=roi.title, inverse=roi.inverse
                )
        else:
            points = coords.reshape(-1, 2) * [fx, fy]
            new = create_image_roi(
                "polygon", points.ravel(), title=roi.title, inverse=roi.inverse
            )
        for single in new.single_rois:
            scaled.add_roi(single)
    return scaled


def _scaled_annotations(
    annotations: list[dict], fx: float, fy: float
) -> tuple[list[dict], int]:
    """Return annotations after the coordinates are multiplied.

    Returns:
        Scaled annotations, and the number of annotations that could not be
        converted (kept unchanged)
    """
    result, skipped = [], 0
    for data in annotations:
        if not is_graphical_annotation_dict(data):
            result.append(data)
            skipped += 1
            continue
        annotation = annotation_from_dict(data)
        try:
            scaled = scale_annotation(annotation, fx, fy)
        except AnnotationTransformError:
            # A rotated rectangle scaled anisotropically becomes a parallelogram
            corners = _rectangle_corners(annotation)
            scaled = PolygonAnnotation(
                id=annotation.id,
                visible=annotation.visible,
                locked=annotation.locked,
                z_index=annotation.z_index,
                title=annotation.title,
                style=annotation.style,
                label=annotation.label,
                metadata=annotation.metadata,
                points=tuple((x * fx, y * fy) for x, y in corners),
            )
        result.append(annotation_to_dict(scaled))
    return result, skipped


def _rectangle_corners(annotation: RectangleAnnotation) -> list[tuple[float, float]]:
    """Return the corners of a rotated rectangle annotation (centered on x, y)."""
    cos_a, sin_a = np.cos(annotation.angle), np.sin(annotation.angle)
    corners = []
    for u, v in ((-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)):
        du, dv = u * annotation.width, v * annotation.height
        corners.append(
            (
                annotation.x + du * cos_a - dv * sin_a,
                annotation.y + du * sin_a + dv * cos_a,
            )
        )
    return corners


def scale_coordinates(obj: ImageObj, fx: float, fy: float) -> int:
    """Multiply the coordinates of an image, keeping its ROIs and annotations.

    Args:
        obj: image (changed in place)
        fx: factor applied to X coordinates
        fy: factor applied to Y coordinates

    Returns:
        Number of annotations that could not be converted
    """
    roi = _scaled_roi(obj, fx, fy)
    if obj.is_uniform_coords:
        obj.set_uniform_coords(obj.dx * fx, obj.dy * fy, obj.x0 * fx, obj.y0 * fy)
    else:
        obj.set_coords(np.asarray(obj.xcoords) * fx, np.asarray(obj.ycoords) * fy)
    obj.roi = roi
    annotations, skipped = _scaled_annotations(obj.get_annotations(), fx, fy)
    obj.set_annotations(annotations)
    return skipped
