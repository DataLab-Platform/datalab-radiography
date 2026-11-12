"""Image quality and NDT measurements: parameters, regions and tables."""

from __future__ import annotations

import guidata.dataset as gds
import numpy as np
from sigima.objects import ImageObj, RectangularROI, TableResult

from ..core import indications as cind
from ..core import measurement as meas
from ..core import quality, units
from . import measurements as wmeas
from .enhancement import PIXELS, SIZE_UNIT_CHOICES

REGION_ROI = "roi"
REGION_BOX = "box"
DIRECTION_AUTO = "auto"
DIRECTION_X = "x"
DIRECTION_Y = "y"
UNIT_CHOICES = tuple((unit, unit) for unit in units.LENGTH_UNITS_MM)
PLANE_HELP = (
    "Name of the magnification of the plane; empty: active magnification; "
    f"{wmeas.PLANE_DETECTOR!r}: detector plane"
)


class RegionError(ValueError):
    """Raised when the measurement region of an image is invalid."""


# ---------------------------------------------------------------------------
# SNR and CNR
# ---------------------------------------------------------------------------


class SNRParam(gds.DataSet):
    """Signal-to-noise ratio"""

    detrend = gds.BoolItem(
        "Remove slow variations",
        default=True,
        help="Fit a plane in each region, so that the heel effect and thickness "
        "variations are not counted as noise",
    )
    srb_um = gds.FloatItem(
        "Basic spatial resolution",
        default=0.0,
        min=0.0,
        unit="µm",
        help="Basic spatial resolution of the detector (duplex IQI), to compute "
        "the SNR normalized to 88.6 µm; 0: not computed",
    )


class CNRParam(gds.DataSet):
    """Contrast-to-noise ratio"""

    background_roi = gds.IntItem(
        "Background ROI",
        default=0,
        min=0,
        help="Number of the background region of interest (1 for the first); "
        "0: the last one. Every other region is an indication.",
    )
    detrend = gds.BoolItem(
        "Remove slow variations from the background noise", default=True
    )


def roi_regions(obj: ImageObj) -> list[tuple[str, int, np.ndarray]]:
    """Return (title, ROI index, inside mask) for each ROI (whole image if none)."""
    if obj.roi is None or obj.roi.is_empty():
        return [("Image", -1, np.ones(obj.data.shape, dtype=bool))]
    regions = []
    for index, single in enumerate(obj.roi.single_rois):
        title = single.title or f"ROI {index + 1}"
        regions.append((title, index, ~single.to_mask(obj)))
    return regions


def _statistics(
    obj: ImageObj, inside: np.ndarray, detrend: bool
) -> quality.RegionStatistics:
    """Return the statistics of an image in a region."""
    rows, cols = np.nonzero(inside)
    return quality.region_statistics(obj.data[rows, cols], rows, cols, detrend)


def snr_table(obj: ImageObj, p: SNRParam) -> TableResult:
    """Return the SNR (and normalized SNR) of each ROI of an image."""
    headers = ["Region", "Mean", "Noise", "SNR"]
    if p.srb_um > 0:
        headers.append(f"SNRN (SRb {p.srb_um:g} µm)")
    rows, indices = [], []
    for title, index, inside in roi_regions(obj):
        stats = _statistics(obj, inside, p.detrend)
        row = [title, stats.mean, stats.std, stats.snr]
        if p.srb_um > 0:
            row.append(quality.normalized_snr(stats.snr, p.srb_um))
        rows.append(row)
        indices.append(index)
    return TableResult.from_rows(
        "SNR", headers, rows, indices, func_name="radiography_snr"
    )


def cnr_table(obj: ImageObj, p: CNRParam) -> TableResult:
    """Return the CNR of each indication ROI against the background ROI.

    Raises:
        RegionError: if the image has fewer than 2 ROIs
    """
    regions = roi_regions(obj)
    if len(regions) < 2:
        raise RegionError(
            "Define at least 2 regions of interest: indications and background"
        )
    position = len(regions) - 1 if p.background_roi == 0 else p.background_roi - 1
    if not 0 <= position < len(regions):
        raise RegionError(f"There is no region of interest {p.background_roi}")
    title, _index, inside = regions[position]
    background = _statistics(obj, inside, p.detrend)
    rows, indices = [], []
    for other, (name, index, mask) in enumerate(regions):
        if other == position:
            continue
        stats = _statistics(obj, mask, False)
        contrast, cnr = quality.contrast_to_noise(stats, background)
        rows.append([name, stats.mean, background.mean, background.std, contrast, cnr])
        indices.append(index)
    headers = [
        "Indication",
        "Mean",
        f"Background mean ({title})",
        "Background noise",
        "Contrast",
        "CNR",
    ]
    return TableResult.from_rows(
        "CNR", headers, rows, indices, func_name="radiography_cnr"
    )


# ---------------------------------------------------------------------------
# Measurement regions
# ---------------------------------------------------------------------------


class RegionParam(gds.DataSet):
    """Measurement region"""

    region = gds.ChoiceItem(
        "Region",
        (
            (REGION_ROI, "Region of interest of the image"),
            (REGION_BOX, "Rectangle below (image coordinates)"),
        ),
        default=REGION_ROI,
    )
    x0 = gds.FloatItem("X min", default=0.0)
    y0 = gds.FloatItem("Y min", default=0.0)
    x1 = gds.FloatItem("X max", default=0.0)
    y1 = gds.FloatItem("Y max", default=0.0)


def box_slices(obj: ImageObj, p: RegionParam) -> tuple[slice, slice]:
    """Return the rows and columns of the measurement rectangle of an image.

    In ROI mode, the rectangle is the bounding box of the first rectangular
    ROI of the image.

    Raises:
        RegionError: if the region is missing or too small
    """
    if p.region == REGION_ROI:
        rois = [] if obj.roi is None else obj.roi.single_rois
        rectangles = [roi for roi in rois if isinstance(roi, RectangularROI)]
        if not rectangles:
            raise RegionError(
                "Draw a rectangular region of interest on the image, or give "
                "the rectangle coordinates"
            )
        rows, cols = np.nonzero(~rectangles[0].to_mask(obj))
        if rows.size == 0:
            raise RegionError("The region of interest is outside the image")
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    else:
        c0f, r0f, c1f, r1f = obj.physical_to_indices(
            [p.x0, p.y0, p.x1, p.y1], clip=True, as_float=True
        )
        r0, r1 = sorted((int(round(r0f)), int(round(r1f))))
        c0, c1 = sorted((int(round(c0f)), int(round(c1f))))
        r1, c1 = r1 + 1, c1 + 1
    if r1 - r0 < 3 or c1 - c0 < 3:
        raise RegionError("The measurement rectangle is too small")
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def region_mask(obj: ImageObj, p: RegionParam) -> np.ndarray:
    """Return the search mask of an image: ROIs (whole image if none) or rectangle."""
    if p.region == REGION_BOX:
        rows, cols = box_slices(obj, p)
        mask = np.zeros(obj.data.shape, dtype=bool)
        mask[rows, cols] = True
        return mask
    mask = np.zeros(obj.data.shape, dtype=bool)
    for _title, _index, inside in roi_regions(obj):
        mask |= inside
    return mask


def box_profile(
    obj: ImageObj, p: RegionParam, direction: str = DIRECTION_AUTO
) -> tuple[np.ndarray, np.ndarray, str]:
    """Return the robust profile along a rectangle and its coordinates.

    Args:
        obj: image (uniform coordinates)
        p: measurement region
        direction: profile direction, or ``auto`` for the longer side

    Returns:
        Profile, image coordinates along the profile, and direction (x or y)
    """
    if not obj.is_uniform_coords:
        raise RegionError("Profiles need an image with uniform coordinates")
    rows, cols = box_slices(obj, p)
    region = np.asarray(obj.data, dtype=float)[rows, cols]
    if direction == DIRECTION_AUTO:
        direction = DIRECTION_X if region.shape[1] >= region.shape[0] else DIRECTION_Y
    if direction == DIRECTION_X:
        profile = quality.robust_profile(region, axis=0)
        coords = obj.x0 + obj.dx * np.arange(cols.start, cols.stop)
    else:
        profile = quality.robust_profile(region, axis=1)
        coords = obj.y0 + obj.dy * np.arange(rows.start, rows.stop)
    return profile, coords, direction


def plane_step(obj: ImageObj, plane: meas.Plane, direction: str) -> float:
    """Return the profile sample spacing in mm in a plane."""
    if direction == DIRECTION_X:
        return abs(obj.dx) * plane.scale[0]
    return abs(obj.dy) * plane.scale[1]


DIRECTION_CHOICES = (
    (DIRECTION_AUTO, "Along the longer side"),
    (DIRECTION_X, "Along X"),
    (DIRECTION_Y, "Along Y"),
)


# ---------------------------------------------------------------------------
# Recipe parameters
# ---------------------------------------------------------------------------


class DuplexParam(RegionParam):
    """Basic spatial resolution (duplex IQI)"""

    direction = gds.ChoiceItem("Profile", DIRECTION_CHOICES, default=DIRECTION_AUTO)
    plane = gds.StringItem("IQI plane", default="", help=PLANE_HELP)
    first_element = gds.IntItem(
        "First element",
        default=1,
        min=1,
        max=13,
        help="Number of the first duplex element crossed by the profile (1: D1)",
    )


class WallThicknessParam(RegionParam):
    """Wall thickness (tangential)"""

    direction = gds.ChoiceItem("Profile", DIRECTION_CHOICES, default=DIRECTION_AUTO)
    plane = gds.StringItem("Pipe axis plane", default="", help=PLANE_HELP)
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default="mm")


class IndicationParam(RegionParam):
    """Indication detection"""

    plane = gds.StringItem("Plane of the indications", default="", help=PLANE_HELP)
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default="mm")
    polarity = gds.ChoiceItem(
        "Indications",
        (
            (cind.POLARITY_BRIGHTER, "Brighter (voids on raw images)"),
            (cind.POLARITY_DARKER, "Darker (inclusions, or voids on attenuation)"),
            (cind.POLARITY_BOTH, "Brighter and darker"),
        ),
        default=cind.POLARITY_BRIGHTER,
    )
    background_size = gds.FloatItem(
        "Background size",
        default=2.0,
        min=0.0,
        nonzero=True,
        help="About 1.5 times the largest indication",
    )
    background_unit = gds.ChoiceItem("Unit", SIZE_UNIT_CHOICES, default="mm")
    detection = gds.FloatItem(
        "Detection threshold",
        default=5.0,
        min=0.5,
        unit="σ",
        help="An indication has at least one pixel beyond this contrast",
    )
    sizing = gds.FloatItem(
        "Sizing threshold",
        default=2.5,
        min=0.5,
        unit="σ",
        help="Contrast giving the extent of the indications",
    )
    min_pixels = gds.IntItem("Smallest indication", default=4, min=1, unit="pixels")
    despeckle = gds.BoolItem(
        "Ignore isolated pixels",
        default=True,
        help="3×3 median filter of the contrast: defective pixels and direct "
        "photon hits are not detected",
    )
    label_image = gds.BoolItem("Add the label image", default=False)


def background_pixels(obj: ImageObj, p: IndicationParam) -> tuple[float, float]:
    """Return the background size of an indication search, in pixels (X, Y)."""
    if not p.background_size > 0:
        raise ValueError("The background size must be strictly positive")
    if p.background_unit == PIXELS:
        return float(p.background_size), float(p.background_size)
    length_mm = units.to_mm(p.background_size, p.background_unit)
    scale = wmeas.plane_named(obj, p.plane).scale
    return length_mm / (scale[0] * abs(obj.dx)), length_mm / (scale[1] * abs(obj.dy))
