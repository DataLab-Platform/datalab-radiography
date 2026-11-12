"""DICOM attributes of radiographs: calibration and display (host-neutral)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sigima.objects import ImageObj

from ..core import dicom as cdicom
from ..core import metadata as md
from . import display as wdisplay
from . import magnification as wmag
from . import objects
from .settings import RadiographySettings


def matches(obj: ImageObj, attrs: Mapping[str, Any]) -> bool:
    """Return True if DICOM attributes describe the data of an image.

    Images derived from a DICOM image keep its source file: their size must
    match the size given by the attributes.
    """
    rows, columns = attrs.get("Rows"), attrs.get("Columns")
    if not isinstance(rows, int) or not isinstance(columns, int):
        return True
    return obj.data.ndim == 2 and obj.data.shape == (rows, columns)


def apply_attributes(
    obj: ImageObj,
    attrs: Mapping[str, Any],
    settings: RadiographySettings | None = None,
    force: bool = False,
) -> list[str]:
    """Store DICOM attributes in an image and apply their calibration.

    The detector pixel pitch, the acquisition geometry and the magnifications
    given by the attributes are set when the image has none (or always, with
    ``force``). The window and the gray-scale inversion are applied by the
    radiographic display profile, applied here when ``settings`` are given.

    Args:
        obj: image
        attrs: technical DICOM attributes (see :func:`.core.dicom.attributes`)
        settings: plugin preferences, to apply the display profile
        force: replace the pitch, geometry and magnifications of the image

    Returns:
        Descriptions of the changes
    """
    changes = []
    obj.metadata[md.DICOM_KEY] = md.dumps(dict(attrs))
    calibration = cdicom.calibration(attrs)
    if calibration.pitch_mm is not None and (
        force or objects.get_detector_pitch(obj) is None
    ):
        objects.set_detector_pitch(obj, calibration.pitch_mm)
        changes.append(
            "detector pixel pitch {:g} × {:g} mm".format(*calibration.pitch_mm)
        )
    geometry = cdicom.geometry(attrs)
    geometry.pop("sod_mm", None)
    if geometry and (force or not objects.get_geometry(obj)):
        objects.set_geometry(obj, geometry)
    if force or not len(objects.get_magnifications(obj)):
        items = [
            item
            for item in wmag.magnifications_from_metadata(obj)
            if item.details.get("source") == "DICOM"
        ]
        for index, item in enumerate(items):
            wmag.add_magnification(obj, item, make_active=index == 0)
            changes.append(f"magnification {item.factor or 0:.4g} ({item.name})")
    if settings is not None:
        wdisplay.apply_profile(
            obj,
            settings.film_convention,
            settings.low_percentile,
            settings.high_percentile,
        )
        changes.append("display")
    return changes
