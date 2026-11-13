"""Recipes of detector corrections and thickness calibration."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import guidata.dataset as gds
import numpy as np
from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeExecutionContext,
    RecipeInputs,
    RecipeInputSlot,
    RecipeMetadataRequirement,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
    RecipeValidationError,
)
from sigima.objects import ImageObj, TableResult, create_signal

from .. import PLUGIN_ID, __version__
from ..core import attenuation as core_attenuation
from ..core import corrections as core_corrections
from ..core import metadata as md
from ..core import units
from . import display as wdisplay
from .corrections import (
    OffsetGainParam,
    derived_image,
    detector_saturation_level,
    suggest_frame_bindings,
)

FRAME_ROLE_HINT = RecipeMetadataRequirement(
    md.FRAME_ROLE_KEY,
    "'raw', 'dark' or 'flat'; images without a role are raw radiographs",
    required=False,
)
UNIT_CHOICES = tuple((unit, unit) for unit in units.LENGTH_UNITS_MM)
#: Fraction of extrapolated pixels of a thickness map above which it is reported
EXTRAPOLATION_WARNING_FRACTION = 0.01


def _error(code: str, message: str) -> RecipeDiagnostic:
    """Return a blocking diagnostic."""
    return RecipeDiagnostic("error", code, message)


def _warning(code: str, message: str, **details) -> RecipeDiagnostic:
    """Return a warning diagnostic."""
    return RecipeDiagnostic("warning", code, message, details)


def _shape_diagnostics(inputs: RecipeInputs) -> list[RecipeDiagnostic]:
    """Require one image size across all slots."""
    shapes = {np.shape(image.data) for images in inputs.values() for image in images}
    if len(shapes) > 1:
        sizes = ", ".join("×".join(str(size) for size in shape) for shape in shapes)
        return [_error("inconsistent-image-size", f"Images differ in size: {sizes}")]
    return []


def _dark_flat_slots(required: bool) -> tuple[RecipeInputSlot, RecipeInputSlot]:
    """Return the dark and flat frame slots."""
    return (
        RecipeInputSlot(
            "dark",
            RecipeObjectType.IMAGE,
            RecipeCardinality.MANY,
            required=required,
            title="Dark frames",
            description="Frames acquired with the X-ray beam off (offset)",
            metadata=(FRAME_ROLE_HINT,),
        ),
        RecipeInputSlot(
            "flat",
            RecipeObjectType.IMAGE,
            RecipeCardinality.MANY,
            required=required,
            title="Flat frames",
            description=(
                "Frames acquired with the beam on and no object, with the same "
                "settings as the radiographs (gain)"
            ),
            metadata=(FRAME_ROLE_HINT,),
        ),
    )


# ---------------------------------------------------------------------------
# Offset and gain correction
# ---------------------------------------------------------------------------


class OffsetGainRecipeParam(OffsetGainParam):
    """Offset and gain correction"""

    defects = gds.BoolItem(
        "Correct defective pixels",
        default=True,
        help="Replace hot, dead and weak pixels found in the dark and flat frames",
    )
    saturation_level = gds.FloatItem(
        "Saturation level",
        default=0.0,
        min=0.0,
        unit="DN",
        help="Value of a saturated pixel of the raw and flat frames. 0: full "
        "scale of the detector given by the images (DICOM BitsStored, "
        "simulator), otherwise estimated from the data (heuristic)",
    )


def _saturation_level(
    image: ImageObj, parameters: OffsetGainRecipeParam
) -> float | None:
    """Return the saturation level of an image: parameter, then metadata."""
    if parameters.saturation_level > 0:
        return float(parameters.saturation_level)
    return detector_saturation_level(image)


def _saturation_text(estimate: core_corrections.SaturationEstimate) -> str:
    """Return the saturation level of the quality table."""
    if estimate.level is None:
        return "not detected (heuristic)"
    if estimate.heuristic:
        return f"≥ {estimate.level:g} DN (heuristic)"
    return f"≥ {estimate.level:g} DN"


def _saturation_message(
    title: str, estimate: core_corrections.SaturationEstimate, consequence: str
) -> str:
    """Return the warning of saturated pixels in a frame."""
    percent = f"{100 * estimate.fraction:.2g} %"
    if estimate.heuristic:
        return (
            f"{title}: possible saturation (heuristic): {percent} of the pixels "
            f"are close to the full scale of the digitizer ({estimate.level:g} DN "
            f"or more); {consequence}. Give the saturation level of the detector "
            "to confirm."
        )
    return (
        f"{title}: {percent} of the pixels are saturated ({estimate.level:g} DN "
        f"or more); {consequence}."
    )


def check_offset_gain_inputs(
    inputs: RecipeInputs, parameters: gds.DataSet | None
) -> list[RecipeDiagnostic]:
    """Check the raw, dark and flat frames of the correction."""
    return _shape_diagnostics(inputs)


def run_offset_gain(
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
    context: RecipeExecutionContext,
) -> RecipeOutcome:
    """Correct raw radiographs with dark and flat frames."""
    if not isinstance(parameters, OffsetGainRecipeParam):
        raise RecipeValidationError("Offset and gain parameters are required")
    context.raise_if_cancelled()
    context.report_progress(0.0, "Averaging dark and flat frames")
    try:
        dark = core_corrections.mean_frame([image.data for image in inputs["dark"]])
        flat = core_corrections.mean_frame([image.data for image in inputs["flat"]])
    except ValueError as exc:
        raise RecipeValidationError(str(exc)) from exc
    defects = None
    if parameters.defects:
        defects = core_corrections.detector_defect_map(dark, flat)
    diagnostics = []
    for index, image in enumerate(inputs["flat"], start=1):
        estimate = core_corrections.saturation(
            image.data, _saturation_level(image, parameters), defects
        )
        if estimate.fraction > 0:
            diagnostics.append(
                _warning(
                    "saturated-flat",
                    _saturation_message(
                        image.title or f"Flat {index}",
                        estimate,
                        "the gain is underestimated there and the corrected "
                        "values are wrong",
                    ),
                    image=image.title,
                )
            )
    raws = inputs["raw"]
    outputs, rows = [], []
    transmission = parameters.output == core_corrections.OUTPUT_TRANSMISSION
    for index, raw in enumerate(raws, start=1):
        context.raise_if_cancelled()
        context.report_progress(index / (len(raws) + 1), f"Correcting {raw.title}")
        try:
            data, report = core_corrections.flat_field_correct(
                raw.data,
                dark,
                flat,
                parameters.output,
                parameters.min_gain / 100.0,
                _saturation_level(raw, parameters),
                defects,
            )
        except ValueError as exc:
            raise RecipeValidationError(str(exc)) from exc
        if defects is not None and defects.any():
            data = core_corrections.correct_defective_pixels(data, defects)
        prefix = "Transmission" if transmission else "Corrected"
        image = derived_image(
            raw, data, f"{prefix} {raw.title}", zunit="" if transmission else None
        )
        image.metadata[md.OUTPUT_ROLE_KEY] = parameters.output
        outputs.append(RecipeObjectOutput(f"corrected-{index}", image))
        rows.append(
            [
                raw.title,
                report.dark_level,
                report.gain_level,
                100.0 * report.invalid_gain_fraction,
                100.0 * report.negative_fraction,
                100.0 * report.saturated_fraction,
                _saturation_text(report.saturation),
                0 if defects is None else int(defects.sum()),
            ]
        )
        if report.saturated_fraction > 0:
            diagnostics.append(
                _warning(
                    "saturated-radiograph",
                    _saturation_message(
                        raw.title,
                        report.saturation,
                        "the attenuation of these pixels is underestimated",
                    ),
                    image=raw.title,
                )
            )
        if report.invalid_gain_fraction > 1e-3:
            diagnostics.append(
                _warning(
                    "invalid-gain",
                    f"{100 * report.invalid_gain_fraction:.2g} % of the pixels "
                    "have a flat frame close to the dark frame (NaN)",
                )
            )
    table = TableResult.from_rows(
        "Offset and gain correction",
        [
            "Radiograph",
            "Dark level (DN)",
            "Gain level (DN)",
            "Invalid gain (%)",
            "Below dark (%)",
            "Saturated (%)",
            "Saturation level",
            "Defective pixels",
        ],
        rows,
        [-1] * len(rows),
        func_name="offset_gain_correction",
    )
    context.report_progress(1.0, "Done")
    return RecipeOutcome(
        objects=tuple(outputs),
        results=(RecipeResultOutput("report", table, "corrected-1"),),
        diagnostics=tuple(diagnostics),
    )


OFFSET_GAIN_RECIPE = RecipeDescriptor(
    recipe_id=f"{PLUGIN_ID}:offset-gain-correction",
    plugin_version=__version__,
    title="Offset and gain correction",
    version="1.0.0",
    description=(
        "Correct raw radiographs with mean dark and flat frames: detector offset "
        "and gain non-uniformity, heel effect, and defective pixels."
    ),
    inputs=(
        RecipeInputSlot(
            "raw",
            RecipeObjectType.IMAGE,
            RecipeCardinality.MANY,
            title="Radiographs",
            description="Raw radiographs acquired with the dark and flat settings",
            metadata=(FRAME_ROLE_HINT,),
        ),
        *_dark_flat_slots(required=True),
    ),
    parameter_class=OffsetGainRecipeParam,
    run=run_offset_gain,
    suggest_bindings=suggest_frame_bindings,
    check_inputs=check_offset_gain_inputs,
)


# ---------------------------------------------------------------------------
# Step wedge thickness calibration
# ---------------------------------------------------------------------------


class StepWedgeParam(gds.DataSet):
    """Step wedge calibration"""

    thicknesses = gds.StringItem(
        "Step thicknesses",
        default="2, 4, 6, 8, 10",
        help="Thickness of each step, in the order of the regions of interest",
    )
    unit = gds.ChoiceItem("Unit", UNIT_CHOICES, default="mm")
    free_beam_roi = gds.IntItem(
        "Free-beam region",
        default=0,
        min=0,
        help="Number of the region of interest in the air (0: none). Not used "
        "with dark and flat frames.",
    )
    offset = gds.FloatItem(
        "Offset",
        default=0.0,
        help="Dark level of the radiographs (not used with dark and flat frames)",
    )
    degree = gds.IntItem("Polynomial degree", default=2, min=1, max=4)


def parse_thicknesses(text: str) -> list[float]:
    """Return the thicknesses of a comma or space separated list.

    Raises:
        ValueError: if a value is not a positive number
    """
    values = [item for item in text.replace(";", ",").replace(",", " ").split()]
    try:
        thicknesses = [float(item) for item in values]
    except ValueError as exc:
        raise ValueError(f"Invalid step thicknesses: {text!r}") from exc
    if not thicknesses or any(not math.isfinite(t) or t < 0 for t in thicknesses):
        raise ValueError("Step thicknesses must be positive numbers")
    return thicknesses


def _step_rois(image: ImageObj, free_beam_roi: int) -> tuple[list, object | None]:
    """Return the step ROIs and the free-beam ROI of a step wedge image."""
    rois = [] if image.roi is None else list(image.roi.single_rois)
    free = None
    if free_beam_roi:
        if not 1 <= free_beam_roi <= len(rois):
            raise ValueError(f"There is no region of interest {free_beam_roi}")
        free = rois.pop(free_beam_roi - 1)
    return rois, free


def check_step_wedge_inputs(
    inputs: RecipeInputs, parameters: gds.DataSet | None
) -> list[RecipeDiagnostic]:
    """Check the step wedge image, its ROIs and the optional frames."""
    if not isinstance(parameters, StepWedgeParam):
        parameters = StepWedgeParam()
    diagnostics = _shape_diagnostics(inputs)
    if bool(inputs.get("dark")) != bool(inputs.get("flat")):
        diagnostics.append(
            _error("dark-flat-pair", "Give both dark and flat frames, or neither")
        )
    try:
        thicknesses = parse_thicknesses(parameters.thicknesses)
        rois, _free = _step_rois(inputs["wedge"][0], parameters.free_beam_roi)
    except ValueError as exc:
        return [*diagnostics, _error("invalid-steps", str(exc))]
    if len(rois) != len(thicknesses):
        diagnostics.append(
            _error(
                "step-count",
                f"The image has {len(rois)} step region(s) of interest for "
                f"{len(thicknesses)} thickness(es)",
            )
        )
    return diagnostics


def _attenuation_function(
    inputs: RecipeInputs, parameters: StepWedgeParam
) -> tuple[Callable[[ImageObj], np.ndarray], bool]:
    """Return a function mapping a radiograph to its attenuation image.

    Returns:
        The function and whether a zero attenuation means zero thickness
    """
    wedge = inputs["wedge"][0]
    if inputs.get("dark") and inputs.get("flat"):
        dark = core_corrections.mean_frame([image.data for image in inputs["dark"]])
        flat = core_corrections.mean_frame([image.data for image in inputs["flat"]])
        defects = core_corrections.detector_defect_map(dark, flat)

        def with_frames(image: ImageObj) -> np.ndarray:
            transmission, _report = core_corrections.flat_field_correct(
                image.data, dark, flat, core_corrections.OUTPUT_TRANSMISSION
            )
            if defects.any():
                transmission = core_corrections.correct_defective_pixels(
                    transmission, defects
                )
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(transmission > 0, -np.log(transmission), np.nan)

        return with_frames, True
    _rois, free = _step_rois(wedge, parameters.free_beam_roi)
    if free is not None:
        mask = ~free.to_mask(wedge)
        free_beam = float(np.nanmedian(wedge.data[mask])) - parameters.offset

        def with_free_beam(image: ImageObj) -> np.ndarray:
            return core_attenuation.attenuation(
                image.data - parameters.offset, free_beam
            )

        return with_free_beam, True

    def log_only(image: ImageObj) -> np.ndarray:
        signal = np.asarray(image.data, dtype=float) - parameters.offset
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(signal > 0, -np.log(signal), np.nan)

    return log_only, False


def run_step_wedge(
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
    context: RecipeExecutionContext,
) -> RecipeOutcome:
    """Calibrate the thickness from a step wedge and map thicknesses."""
    if not isinstance(parameters, StepWedgeParam):
        raise RecipeValidationError("Step wedge parameters are required")
    blocking = [
        item
        for item in check_step_wedge_inputs(inputs, parameters)
        if item.level == "error"
    ]
    if blocking:
        raise RecipeValidationError(blocking[0].message)
    context.report_progress(0.0, "Measuring the steps")
    wedge = inputs["wedge"][0]
    unit_mm = units.unit_mm(parameters.unit)
    thicknesses = [t * unit_mm for t in parse_thicknesses(parameters.thicknesses)]
    to_attenuation, through_origin = _attenuation_function(inputs, parameters)
    values = to_attenuation(wedge)
    rois, _free = _step_rois(wedge, parameters.free_beam_roi)
    steps = [float(np.nanmedian(values[~roi.to_mask(wedge)])) for roi in rois]
    try:
        calibration = core_attenuation.calibrate_thickness(
            steps, thicknesses, parameters.degree, through_origin
        )
    except ValueError as exc:
        raise RecipeValidationError(str(exc)) from exc
    context.raise_if_cancelled()
    span = max(steps) - min(0.0, min(steps))
    x = np.linspace(min(0.0, min(steps)), max(steps) + 0.1 * span, 200)
    curve = create_signal(
        "Thickness calibration",
        x,
        calibration.thickness(x),
        units=("", "mm"),
        labels=("Attenuation", "Thickness"),
    )
    curve.metadata[md.OUTPUT_ROLE_KEY] = "thickness-calibration"
    outputs = [RecipeObjectOutput("calibration", curve)]
    images = [wedge, *inputs.get("images", ())]
    diagnostics = []
    for index, image in enumerate(images, start=1):
        context.raise_if_cancelled()
        context.report_progress(index / (len(images) + 1), f"Mapping {image.title}")
        values = to_attenuation(image)
        thickness = calibration.thickness(values)
        mapped = derived_image(image, thickness, f"Thickness of {image.title}", "mm")
        mapped.metadata[md.OUTPUT_ROLE_KEY] = "thickness-map"
        if wdisplay.has_profile(mapped):
            mapped.set_metadata_option("invert_colormap", False)
        outputs.append(RecipeObjectOutput(f"thickness-{index}", mapped))
        extrapolated = calibration.extrapolated_fraction(values)
        if extrapolated > EXTRAPOLATION_WARNING_FRACTION:
            low, high = calibration.domain
            diagnostics.append(
                _warning(
                    "extrapolated-thickness",
                    f"{image.title}: the attenuation of {100 * extrapolated:.2g} % "
                    f"of the pixels is outside the calibrated range ({low:.3g} to "
                    f"{high:.3g}): their thickness is extrapolated",
                    image=image.title,
                )
            )
    rows = [
        [f"Step {index}", t, a, t + r, r]
        for index, (t, a, r) in enumerate(
            zip(
                calibration.thicknesses, calibration.attenuations, calibration.residuals
            ),
            start=1,
        )
    ]
    table = TableResult.from_rows(
        "Step wedge calibration",
        [
            "Step",
            "True thickness (mm)",
            "Attenuation",
            "Calibrated thickness (mm)",
            "Residual (mm)",
        ],
        rows,
        [-1] * len(rows),
        func_name="step_wedge_calibration",
        attrs={"coefficients": list(calibration.coefficients)},
    )
    if calibration.rms_residual > 0.02 * max(thicknesses):
        diagnostics.append(
            _warning(
                "large-residuals",
                f"RMS residual {calibration.rms_residual:.3g} mm: check the step "
                "regions and thicknesses, or change the polynomial degree",
            )
        )
    context.report_progress(1.0, "Done")
    return RecipeOutcome(
        objects=tuple(outputs),
        results=(RecipeResultOutput("steps", table, "calibration"),),
        diagnostics=tuple(diagnostics),
    )


def suggest_step_wedge_bindings(
    candidates: Sequence[ImageObj],
) -> dict[str, tuple[ImageObj, ...]]:
    """Assign the step wedge (first raw image with ROIs), frames and images."""
    bindings = suggest_frame_bindings(candidates)
    raws = list(bindings.pop("raw", ()))
    wedges = [image for image in raws if image.roi is not None and len(image.roi)]
    wedge = wedges[0] if wedges else (raws[0] if raws else None)
    if wedge is not None:
        bindings["wedge"] = (wedge,)
        others = tuple(image for image in raws if image is not wedge)
        if others:
            bindings["images"] = others
    return bindings


STEP_WEDGE_RECIPE = RecipeDescriptor(
    recipe_id=f"{PLUGIN_ID}:step-wedge-calibration",
    plugin_version=__version__,
    title="Step wedge thickness calibration",
    version="1.0.0",
    description=(
        "Calibrate the thickness of a material against the attenuation, from a "
        "radiograph of a step wedge with one region of interest per step, then "
        "map the thickness of radiographs. The calibration takes beam hardening "
        "into account."
    ),
    inputs=(
        RecipeInputSlot(
            "wedge",
            RecipeObjectType.IMAGE,
            RecipeCardinality.ONE,
            title="Step wedge radiograph",
            description="Radiograph with one region of interest on each step",
        ),
        *_dark_flat_slots(required=False),
        RecipeInputSlot(
            "images",
            RecipeObjectType.IMAGE,
            RecipeCardinality.MANY,
            required=False,
            title="Radiographs to map",
            description="Radiographs of parts of the same material",
        ),
    ),
    parameter_class=StepWedgeParam,
    run=run_step_wedge,
    suggest_bindings=suggest_step_wedge_bindings,
    check_inputs=check_step_wedge_inputs,
)

__all__ = ["OFFSET_GAIN_RECIPE", "STEP_WEDGE_RECIPE"]
