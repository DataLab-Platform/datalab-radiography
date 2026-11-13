"""Detector corrections and physical quantities on Sigima images."""

from __future__ import annotations

from collections.abc import Sequence

import guidata.dataset as gds
import numpy as np
from sigima.objects import ImageObj

from ..core import corrections
from ..core import metadata as md
from . import display as wdisplay
from . import objects


class DefectivePixelParam(gds.DataSet):
    """Defective pixels"""

    size = gds.IntItem(
        "Neighborhood size",
        default=5,
        min=3,
        max=15,
        unit="pixels",
        help="Odd size of the neighborhood used to compute the local median",
    )
    threshold = gds.FloatItem(
        "Detection threshold",
        default=6.0,
        min=1.0,
        help="Deviation from the local median, in robust standard deviations",
    )


class OffsetGainParam(gds.DataSet):
    """Offset and gain correction"""

    output = gds.ChoiceItem(
        "Output",
        (
            (corrections.OUTPUT_CORRECTED, "Corrected image (digital numbers)"),
            (corrections.OUTPUT_TRANSMISSION, "Transmission (I - D) / (F - D)"),
        ),
        default=corrections.OUTPUT_CORRECTED,
    )
    min_gain = gds.FloatItem(
        "Minimum gain",
        default=5.0,
        min=0.0,
        max=100.0,
        unit="%",
        help="Pixels whose flat minus dark is below this fraction of its median "
        "are invalid (NaN)",
    )


I0_VALUE = "value"
I0_ROI = "roi"
I0_MAXIMUM = "maximum"


class AttenuationParam(gds.DataSet):
    """Attenuation -ln(I / I0)"""

    free_beam = gds.ChoiceItem(
        "Free-beam intensity I0",
        (
            (I0_ROI, "Median in the regions of interest (air)"),
            (I0_VALUE, "Value"),
            (I0_MAXIMUM, "Upper percentile of the image"),
        ),
        default=I0_ROI,
    )
    i0 = gds.FloatItem("I0 value", default=1.0, min=0.0, nonzero=True)
    offset = gds.FloatItem(
        "Offset",
        default=0.0,
        help="Dark level subtracted from I and I0 (0 for corrected images)",
    )


class FlatAttenuationParam(gds.DataSet):
    """Attenuation with a flat image"""

    offset = gds.FloatItem(
        "Offset",
        default=0.0,
        help="Dark level subtracted from the image and the flat image",
    )


class ThicknessParam(gds.DataSet):
    """Equivalent thickness"""

    mu = gds.FloatItem(
        "Linear attenuation coefficient μ",
        default=0.15,
        min=0.0,
        nonzero=True,
        unit="1/mm",
        help="Effective coefficient of the material for the spectrum used "
        "(monochromatic approximation: beam hardening is ignored)",
    )


def derived_image(
    src: ImageObj, data: np.ndarray, title: str, zunit: str | None = None
) -> ImageObj:
    """Return a new image computed from a source image.

    The new image keeps the metadata, coordinates and ROIs of the source, and
    its radiographic display, with a window computed again on its values.
    """
    dst = src.copy(title=title)
    dst.data = data
    if zunit is not None:
        dst.zunit = zunit
    dst.zscalemin = dst.zscalemax = None
    for key in (md.FRAME_ROLE_KEY, md.OUTPUT_ROLE_KEY):
        dst.metadata.pop(key, None)
    wdisplay.inherit_display(src, dst)
    return dst


def roi_median(obj: ImageObj, data: np.ndarray | None = None) -> float:
    """Return the median of an image (or of other data) in its ROIs.

    Raises:
        ValueError: if the image has no ROI
    """
    mask = wdisplay.roi_inside_mask(obj)
    if mask is None or not mask.any():
        raise ValueError("Define a region of interest in the free beam (air) first")
    values = (obj.data if data is None else data)[mask]
    return float(np.nanmedian(values))


def free_beam_intensity(obj: ImageObj, param: AttenuationParam) -> float:
    """Return the free-beam intensity I0 (offset subtracted)."""
    if param.free_beam == I0_VALUE:
        value = param.i0 - param.offset
    elif param.free_beam == I0_ROI:
        value = roi_median(obj) - param.offset
    else:
        value = float(np.nanpercentile(obj.data, 99.5)) - param.offset
    if not value > 0:
        raise ValueError("The free-beam intensity minus the offset must be positive")
    return value


def detector_saturation_level(obj: ImageObj) -> float | None:
    """Return the saturation level of the detector of an image, if known.

    It is the full scale of the digitizer, from the DICOM attribute BitsStored
    or from the detector of the simulator.
    """
    for key, path in (
        (md.DICOM_KEY, ("BitsStored",)),
        (md.SIMULATION_TRUTH_KEY, ("detector", "bit_depth")),
    ):
        try:
            value = md.loads(obj.metadata.get(key))
        except md.MetadataSchemaError:
            continue
        for name in path:
            value = value.get(name) if isinstance(value, dict) else None
        if isinstance(value, int) and not isinstance(value, bool) and 0 < value <= 32:
            return float(corrections.full_scale(value))
    return None


def suggest_frame_bindings(
    candidates: Sequence[ImageObj],
) -> dict[str, tuple[ImageObj, ...]]:
    """Assign images to raw, dark and flat slots from their frame role.

    Images without a role are raw radiographs.
    """
    bindings: dict[str, list[ImageObj]] = {"raw": [], "dark": [], "flat": []}
    for image in candidates:
        bindings[objects.get_frame_role(image) or "raw"].append(image)
    return {slot: tuple(images) for slot, images in bindings.items() if images}
