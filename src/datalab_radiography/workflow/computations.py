"""Sigima computation functions of the radiography plugin.

Each function is registered as a processing feature of DataLab Desktop and
DataLab-Web (see :mod:`.features`). Its ``operation_id`` and
``contract_version`` identify its result in DataLab provenance: increase the
contract version whenever a change alters the result.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sigima.objects import ImageObj, TableResult
from sigima.proc.contracts import EXACT, ReplayQualification
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1, dst_2_to_1

from .. import PLUGIN_ID
from ..core import attenuation as core_attenuation
from ..core import corrections as core_corrections
from ..core import display as core_display
from ..core import enhancement as core_enhancement
from ..core import magnification as mag
from ..core import metadata as md
from . import corrections as wcorr
from . import enhancement as wenh
from . import measurements as wmeas
from . import objects
from .corrections import (
    AttenuationParam,
    DefectivePixelParam,
    FlatAttenuationParam,
    ThicknessParam,
)
from .enhancement import (
    FlattenBackgroundParam,
    RadiographicClaheParam,
    RadiographicDenoiseParam,
    RobustAverageParam,
    UnsharpMaskParam,
)
from .magnification import ExpressCoordinatesParam
from .measurements import MeasurementReportParam

OPERATION_PREFIX = PLUGIN_ID


def operation_id(name: str) -> str:
    """Return the operation identifier of a computation function."""
    return f"{OPERATION_PREFIX}.{name}"


_FRAME_METADATA = (
    md.MAGNIFICATIONS_KEY,
    md.COORDINATE_FRAME_KEY,
    md.DETECTOR_PITCH_KEY,
)


@computation_function(
    operation_id=operation_id("express_coordinates"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def express_coordinates(src: ImageObj, p: ExpressCoordinatesParam) -> ImageObj:
    """Express the image coordinates in an object plane, the detector plane or pixels.

    The data are unchanged. ROIs and annotations keep their position on the
    image, and the magnifications stay valid.

    Args:
        src: source image
        p: target coordinate frame

    Returns:
        Image with converted coordinates
    """
    frame = objects.get_coordinate_frame(src)
    pitch = objects.get_detector_pitch(src)
    mset = objects.check_magnifications(src)
    item = None
    if p.target == mag.FRAME_OBJECT:
        name = (p.magnification or "").strip() or mset.active
        if name is None:
            raise mag.MagnificationError("The image has no magnification")
        item = mset.get(name)
    unit = None if p.target == mag.FRAME_PIXEL else p.unit
    (fx, fy), new_frame = mag.express_factors(frame, pitch, p.target, unit, item)
    dst = dst_1_to_1(src, "express_coordinates", new_frame.describe())
    objects.scale_coordinates(dst, fx, fy)
    dst.xunit = dst.yunit = new_frame.unit or ""
    objects.set_coordinate_frame(dst, new_frame)
    if len(mset):
        objects.set_magnifications(dst, mset.rescaled(fx, fy))
    return dst


@computation_function(
    operation_id=operation_id("measurement_report"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def measurement_report(src: ImageObj, p: MeasurementReportParam) -> TableResult | None:
    """Measure the annotations and ROIs of an image in every plane.

    Columns give the values in image coordinates, in the detector plane (if its
    scale is known) and in the object plane of each magnification.

    Args:
        src: image
        p: report parameters

    Returns:
        Measurement table, or None if the image has no measurable shape
    """
    shapes = wmeas.shapes_for_report(src, p.sources)
    if not shapes:
        return None
    planes, rows = wmeas.measure(src, shapes, p.planes, p.unit)
    return wmeas.measurement_table(planes, rows, p.unit)


# ---------------------------------------------------------------------------
# Detector corrections and physical quantities
# ---------------------------------------------------------------------------


@computation_function(
    operation_id=operation_id("defective_pixel_map"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def defective_pixel_map(src: ImageObj, p: DefectivePixelParam) -> ImageObj:
    """Map the pixels deviating from their local median (1: defective).

    Args:
        src: image (raw radiograph, dark or flat frame)
        p: detection parameters

    Returns:
        Map of the defective pixels (unsigned 8-bit integers)
    """
    dst = dst_1_to_1(src, "defective_pixel_map", f"threshold={p.threshold:g}")
    mask = core_corrections.defective_pixel_map(src.data, p.size, p.threshold)
    dst.data = mask.astype(np.uint8)
    dst.zunit = ""
    return dst


@computation_function(
    operation_id=operation_id("correct_defective_pixels"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def correct_defective_pixels(src: ImageObj, p: DefectivePixelParam) -> ImageObj:
    """Replace the pixels deviating from their local median by this median.

    Args:
        src: image
        p: detection parameters

    Returns:
        Corrected image (floating point values)
    """
    dst = dst_1_to_1(src, "correct_defective_pixels", f"threshold={p.threshold:g}")
    mask = core_corrections.defective_pixel_map(src.data, p.size, p.threshold)
    dst.data = core_corrections.correct_defective_pixels(src.data, mask, p.size)
    return dst


@computation_function(
    operation_id=operation_id("attenuation"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def attenuation(src: ImageObj, p: AttenuationParam) -> ImageObj:
    """Compute the attenuation -ln((I - offset) / I0) of a radiograph.

    The free-beam intensity I0 is a value, the median of the image in its
    regions of interest (placed in the air), or its 99.5th percentile.

    Args:
        src: radiograph
        p: attenuation parameters

    Returns:
        Attenuation image (dimensionless)
    """
    free_beam = wcorr.free_beam_intensity(src, p)
    dst = dst_1_to_1(src, "attenuation", f"I0={free_beam:.6g}")
    dst.data = core_attenuation.attenuation(src.data - p.offset, free_beam)
    dst.zunit = ""
    return dst


@computation_function(
    operation_id=operation_id("attenuation_with_flat"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def attenuation_with_flat(
    src1: ImageObj, src2: ImageObj, p: FlatAttenuationParam
) -> ImageObj:
    """Compute the attenuation -ln((I - offset) / (F - offset)) with a flat image.

    Args:
        src1: radiograph I
        src2: flat image F (free beam, same detector and exposure)
        p: offset

    Returns:
        Attenuation image (dimensionless)
    """
    if src1.data.shape != src2.data.shape:
        raise ValueError("The radiograph and the flat image must have the same size")
    dst = dst_2_to_1(src1, src2, "attenuation_with_flat")
    flat = np.asarray(src2.data, dtype=float) - p.offset
    flat = np.where(flat > 0, flat, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        signal = np.asarray(src1.data, dtype=float) - p.offset
        dst.data = np.where(signal > 0, -np.log(signal / flat), np.nan)
    dst.zunit = ""
    return dst


@computation_function(
    operation_id=operation_id("equivalent_thickness"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def equivalent_thickness(src: ImageObj, p: ThicknessParam) -> ImageObj:
    """Convert an attenuation image to the equivalent thickness of a material.

    Monochromatic approximation t = A / mu: beam hardening is ignored (use a
    step wedge calibration to take it into account).

    Args:
        src: attenuation image
        p: linear attenuation coefficient of the material

    Returns:
        Thickness image (mm)
    """
    dst = dst_1_to_1(src, "equivalent_thickness", f"mu={p.mu:g}/mm")
    dst.data = core_attenuation.equivalent_thickness(src.data, p.mu)
    dst.zunit = "mm"
    return dst


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------


@computation_function(
    operation_id=operation_id("flatten_background"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def flatten_background(src: ImageObj, p: FlattenBackgroundParam) -> ImageObj:
    """Remove the large-scale background (thickness variations, heel effect).

    Args:
        src: radiograph
        p: background size and method

    Returns:
        Flattened radiograph, at the mean level of the source
    """
    size = wenh.pixel_sizes(src, p.size, p.unit)
    dst = dst_1_to_1(src, "flatten_background", f"size={p.size:g} {p.unit}")
    dst.data = core_enhancement.flatten_background(src.data, size, p.method, p.mode)
    return dst


@computation_function(
    operation_id=operation_id("unsharp_mask"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def unsharp_mask(src: ImageObj, p: UnsharpMaskParam) -> ImageObj:
    """Enhance small details such as pores and cracks.

    Args:
        src: radiograph
        p: detail size and amount

    Returns:
        Enhanced radiograph
    """
    sigma = wenh.pixel_sizes(src, p.sigma, p.unit)
    dst = dst_1_to_1(src, "unsharp_mask", f"sigma={p.sigma:g} {p.unit}")
    dst.data = core_enhancement.unsharp_mask(src.data, sigma, p.amount)
    return dst


@computation_function(
    operation_id=operation_id("radiographic_clahe"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def radiographic_clahe(src: ImageObj, p: RadiographicClaheParam) -> ImageObj:
    """Adaptive histogram equalization tuned for radiographs.

    The image is first normalized between two percentiles (ignoring the
    saturated free beam), then equalized by tiles of a physical size.

    Args:
        src: radiograph
        p: tile size, clipping limit and normalization percentiles

    Returns:
        Equalized radiograph (values between 0 and 1)
    """
    # pylint: disable=import-outside-toplevel
    from skimage import exposure

    sx, sy = wenh.pixel_sizes(src, p.tile_size, p.unit)
    window = core_display.percentile_window(
        src.data, p.low_percentile, p.high_percentile
    )
    normalized = core_enhancement.normalize(src.data, window.low, window.high)
    kernel = (max(int(round(sy)), 2), max(int(round(sx)), 2))
    dst = dst_1_to_1(src, "radiographic_clahe", f"clip={p.clip_limit:g}")
    dst.data = exposure.equalize_adapthist(
        normalized, kernel_size=kernel, clip_limit=p.clip_limit
    )
    dst.zunit = ""
    return dst


@computation_function(
    operation_id=operation_id("radiographic_denoise"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def radiographic_denoise(src: ImageObj, p: RadiographicDenoiseParam) -> ImageObj:
    """Reduce the noise of a radiograph with a preset suited to its origin.

    Args:
        src: radiograph
        p: method, size and strength

    Returns:
        Denoised radiograph
    """
    data = np.asarray(src.data, dtype=float)
    dst = dst_1_to_1(src, "radiographic_denoise", p.method)
    if p.method == wenh.DENOISE_MEDIAN:
        sx, sy = wenh.pixel_sizes(src, p.size, p.unit)
        size = (max(int(round(sy)) | 1, 3), max(int(round(sx)) | 1, 3))
        dst.data = ndimage.median_filter(data, size=size, mode="nearest")
    elif p.method == wenh.DENOISE_GAUSSIAN:
        sx, sy = wenh.pixel_sizes(src, p.size, p.unit)
        dst.data = ndimage.gaussian_filter(data, (sy, sx), mode="nearest")
    else:
        # pylint: disable=import-outside-toplevel
        from skimage import restoration

        window = core_display.percentile_window(
            data, 0.0, 100.0, exclude_saturation=False
        )
        span = window.high - window.low
        normalized = (data - window.low) / span
        if p.method == wenh.DENOISE_TV:
            result = restoration.denoise_tv_chambolle(normalized, weight=p.weight)
        else:
            sx, sy = wenh.pixel_sizes(src, p.size, p.unit)
            result = restoration.denoise_bilateral(
                normalized, sigma_color=p.weight, sigma_spatial=0.5 * (sx + sy)
            )
        dst.data = result * span + window.low
    return dst


@computation_function(
    operation_id=operation_id("robust_average"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
)
def robust_average(src_list: list[ImageObj], p: RobustAverageParam) -> ImageObj:
    """Average frames, rejecting outliers such as direct photon hits.

    Args:
        src_list: frames of the same size
        p: method and rejection threshold

    Returns:
        Average frame
    """
    if not src_list:
        raise ValueError("Select the frames to average")
    dst = dst_1_to_1(src_list[0], "robust_average", f"{len(src_list)} frames")
    dst.data = core_enhancement.robust_average(
        [src.data for src in src_list], p.method, p.threshold
    )
    return dst
