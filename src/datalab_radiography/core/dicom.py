"""Translation of DICOM (and DICONDE) attributes for radiographs.

Only technical attributes are kept: no attribute identifying a patient, a
component or an operator is copied into the images. This module does not
depend on pydicom: hosts read the header and give plain values.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

#: Technical attributes kept with the images (DICOM keywords)
KEYWORDS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "PhotometricInterpretation",
    "PresentationLUTShape",
    "BitsStored",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "ImagerPixelSpacing",
    "PixelSpacing",
    "PixelSpacingCalibrationType",
    "EstimatedRadiographicMagnificationFactor",
    "DistanceSourceToDetector",
    "DistanceSourceToPatient",
    "FocalSpots",
    "KVP",
    "ExposureTime",
    "XRayTubeCurrent",
)
#: Relative difference above which the pixel spacing is calibrated
SPACING_TOLERANCE = 1e-3
#: Header bytes of a DICOM file after its 128-byte preamble
DICOM_MAGIC = b"DICM"


def _plain(value: Any) -> Any:
    """Return a JSON-compatible value from a header value (number, text, list)."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return None
    if isinstance(value, (list, tuple)) or (
        hasattr(value, "__iter__") and not isinstance(value, str)
    ):
        items = [_plain(item) for item in value]
        return [item for item in items if item is not None]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    text = str(value).strip()
    for kind in (int, float):
        try:
            number = kind(text)
        except ValueError:
            continue
        return number if kind is int or math.isfinite(number) else None
    return text


def attributes(header: Mapping[str, Any]) -> dict[str, Any]:
    """Return the technical attributes of a DICOM header as plain values.

    Multi-valued window attributes keep their first value.

    Args:
        header: DICOM keyword to value (pydicom values are accepted)
    """
    result: dict[str, Any] = {}
    for keyword in KEYWORDS:
        value = _plain(header.get(keyword))
        if value is None or value == "" or value == []:
            continue
        if keyword in ("WindowCenter", "WindowWidth") and isinstance(value, list):
            value = value[0]
        result[keyword] = value
    return result


def is_dicom_header(prefix: bytes) -> bool:
    """Return True if the first bytes of a file are those of a DICOM file."""
    return len(prefix) >= 132 and prefix[128:132] == DICOM_MAGIC


def _spacing(value: Any) -> tuple[float, float] | None:
    """Return (X, Y) spacing from a DICOM (row, column) spacing, in mm."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    try:
        row, column = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (row > 0 and column > 0):
        return None
    return column, row


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Spatial calibration given by DICOM attributes.

    Attributes:
        pitch_mm: detector pixel pitch (X, Y), from ImagerPixelSpacing
        object_spacing_mm: pixel spacing in the object plane (X, Y), when the
         PixelSpacing is calibrated (differs from ImagerPixelSpacing)
        factor: magnification M (estimated factor, source distances, or ratio
         of the spacings)
        calibration_type: PixelSpacingCalibrationType, if any
    """

    pitch_mm: tuple[float, float] | None
    object_spacing_mm: tuple[float, float] | None
    factor: float | None
    calibration_type: str | None = None


def calibration(attrs: Mapping[str, Any]) -> Calibration:
    """Return the spatial calibration of DICOM attributes."""
    pitch = _spacing(attrs.get("ImagerPixelSpacing"))
    spacing = _spacing(attrs.get("PixelSpacing"))
    factor = None
    estimated = attrs.get("EstimatedRadiographicMagnificationFactor")
    if isinstance(estimated, (int, float)) and estimated > 0:
        factor = float(estimated)
    else:
        sdd = attrs.get("DistanceSourceToDetector")
        sod = attrs.get("DistanceSourceToPatient")
        if isinstance(sdd, (int, float)) and isinstance(sod, (int, float)):
            if 0 < sod <= sdd:
                factor = float(sdd) / float(sod)
    object_spacing = None
    if pitch is not None and spacing is not None:
        ratio = (pitch[0] / spacing[0], pitch[1] / spacing[1])
        if max(abs(r - 1.0) for r in ratio) > SPACING_TOLERANCE:
            object_spacing = spacing
            if factor is None:
                factor = math.sqrt(ratio[0] * ratio[1])
    calibration_type = attrs.get("PixelSpacingCalibrationType")
    return Calibration(
        pitch,
        object_spacing,
        factor,
        str(calibration_type) if calibration_type else None,
    )


def geometry(attrs: Mapping[str, Any]) -> dict[str, float]:
    """Return the acquisition geometry of DICOM attributes (empty if unknown)."""
    result: dict[str, float] = {}
    for keyword, key in (
        ("DistanceSourceToDetector", "sdd_mm"),
        ("DistanceSourceToPatient", "sod_mm"),
        ("KVP", "voltage_kv"),
    ):
        value = attrs.get(keyword)
        if isinstance(value, (int, float)) and value > 0:
            result[key] = float(value)
    focal = attrs.get("FocalSpots")
    if isinstance(focal, list):
        focal = focal[0] if focal else None
    if isinstance(focal, (int, float)) and focal > 0:
        result["focal_spot_mm"] = float(focal)
    return result
