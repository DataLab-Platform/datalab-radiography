"""Read and write the plugin metadata of Sigima images."""

from __future__ import annotations

import json
from typing import Any

from sigima.objects import ImageObj

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
