"""Enhancement of radiographs: parameters and physical sizes."""

from __future__ import annotations

import guidata.dataset as gds
from sigima.objects import ImageObj

from ..core import enhancement, units
from ..core import magnification as mag
from . import objects

PIXELS = "px"
SIZE_UNIT_CHOICES = ((PIXELS, "pixels"),) + tuple(
    (unit, unit) for unit in units.LENGTH_UNITS_MM
)

DENOISE_MEDIAN = "median"
DENOISE_GAUSSIAN = "gaussian"
DENOISE_TV = "tv"
DENOISE_BILATERAL = "bilateral"


def pixel_sizes(obj: ImageObj, value: float, unit: str) -> tuple[float, float]:
    """Return a size in pixels along X and Y.

    A physical length is taken in the object plane of the active magnification,
    or in the detector plane when the image has no magnification.

    Args:
        obj: image (uniform coordinates for physical lengths)
        value: size
        unit: ``px`` or a length unit

    Raises:
        ValueError: if a physical length cannot be converted
    """
    if not value > 0:
        raise ValueError("The size must be strictly positive")
    if unit == PIXELS:
        return float(value), float(value)
    if not obj.is_uniform_coords:
        raise ValueError("Physical sizes need an image with uniform coordinates")
    length_mm = units.to_mm(value, unit)
    active = objects.check_magnifications(obj).active_item
    if active is not None:
        scale = active.scale_mm
    else:
        scale = objects.detector_mm_per_unit(obj)
        if scale is None:
            raise mag.MagnificationError(
                "The image has no magnification and no detector pixel pitch: "
                "define one, or give the size in pixels"
            )
    return (
        length_mm / (scale[0] * abs(obj.dx)),
        length_mm / (scale[1] * abs(obj.dy)),
    )


class FlattenBackgroundParam(gds.DataSet):
    """Flatten background"""

    size = gds.FloatItem(
        "Background feature size",
        default=5.0,
        min=0.0,
        nonzero=True,
        help="Features larger than this size belong to the background "
        "(in the object plane of the active magnification)",
    )
    unit = gds.ChoiceItem("Unit", SIZE_UNIT_CHOICES, default="mm")
    method = gds.ChoiceItem(
        "Background",
        (
            (enhancement.BACKGROUND_GAUSSIAN, "Gaussian (smooth variations)"),
            (
                enhancement.BACKGROUND_MORPHOLOGICAL,
                "Morphological (removes small features, keeps steps)",
            ),
        ),
        default=enhancement.BACKGROUND_GAUSSIAN,
    )
    mode = gds.ChoiceItem(
        "Correction",
        (
            (enhancement.FLATTEN_DIVIDE, "Divide (multiplicative variations)"),
            (enhancement.FLATTEN_SUBTRACT, "Subtract (additive variations)"),
        ),
        default=enhancement.FLATTEN_DIVIDE,
    )


class UnsharpMaskParam(gds.DataSet):
    """Unsharp mask"""

    sigma = gds.FloatItem(
        "Detail size",
        default=0.5,
        min=0.0,
        nonzero=True,
        help="Gaussian sigma: details smaller than this size are enhanced",
    )
    unit = gds.ChoiceItem("Unit", SIZE_UNIT_CHOICES, default="mm")
    amount = gds.FloatItem("Amount", default=1.5, min=0.0, max=20.0)


class RadiographicClaheParam(gds.DataSet):
    """Radiographic adaptive histogram equalization (CLAHE)"""

    tile_size = gds.FloatItem(
        "Tile size",
        default=64.0,
        min=0.0,
        nonzero=True,
        help="Size of the contextual regions: a few times the size of the "
        "indications to reveal",
    )
    unit = gds.ChoiceItem("Unit", SIZE_UNIT_CHOICES, default=PIXELS)
    clip_limit = gds.FloatItem(
        "Clipping limit",
        default=0.01,
        min=0.0,
        max=1.0,
        help="Higher values give more contrast and more noise",
    )
    low_percentile = gds.FloatItem(
        "Lower percentile", default=0.5, min=0.0, max=50.0, unit="%"
    )
    high_percentile = gds.FloatItem(
        "Upper percentile", default=99.5, min=50.0, max=100.0, unit="%"
    )


class RadiographicDenoiseParam(gds.DataSet):
    """Radiographic denoising"""

    method = gds.ChoiceItem(
        "Method",
        (
            (DENOISE_MEDIAN, "Median (direct photon hits, impulse noise)"),
            (DENOISE_GAUSSIAN, "Gaussian"),
            (DENOISE_TV, "Total variation (keeps edges)"),
            (DENOISE_BILATERAL, "Bilateral (keeps edges)"),
        ),
        default=DENOISE_MEDIAN,
    )
    size = gds.FloatItem(
        "Size",
        default=3.0,
        min=0.0,
        nonzero=True,
        help="Median window or Gaussian sigma (median: odd number of pixels)",
    )
    unit = gds.ChoiceItem("Unit", SIZE_UNIT_CHOICES, default=PIXELS)
    weight = gds.FloatItem(
        "Strength",
        default=0.1,
        min=0.0,
        nonzero=True,
        help="Total variation weight, or bilateral range sigma, relative to "
        "the image range",
    )


class RobustAverageParam(gds.DataSet):
    """Robust frame averaging"""

    method = gds.ChoiceItem(
        "Method",
        (
            (enhancement.AVERAGE_CLIPPED, "Mean with outlier rejection"),
            (enhancement.AVERAGE_MEDIAN, "Median"),
            (enhancement.AVERAGE_MEAN, "Mean"),
        ),
        default=enhancement.AVERAGE_CLIPPED,
    )
    threshold = gds.FloatItem(
        "Rejection threshold",
        default=3.0,
        min=1.0,
        help="In robust standard deviations of each pixel",
    )
