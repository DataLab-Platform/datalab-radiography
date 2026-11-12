"""Recipes of image quality and NDT measurements.

- Basic spatial resolution with a duplex wire IQI (inspired by ISO 19232-5)
- Wall thickness of pipes in tangential radiography (inspired by ISO 20769-1)
- Detection and sizing of indications (no acceptance criterion)
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import guidata.dataset as gds
import numpy as np
from datalab.plugins.recipes import (
    RecipeCardinality,
    RecipeDescriptor,
    RecipeDiagnostic,
    RecipeExecutionContext,
    RecipeInputs,
    RecipeInputSlot,
    RecipeObjectOutput,
    RecipeObjectType,
    RecipeOutcome,
    RecipeResultOutput,
    RecipeValidationError,
)
from sigima.objects import ImageObj, TableResult, create_signal
from sigima.objects.annotations import (
    AnnotationLabel,
    RangeAnnotation,
    RectangleAnnotation,
)

from .. import PLUGIN_ID, __version__
from ..core import enhancement as core_enhancement
from ..core import indications as cind
from ..core import metadata as md
from ..core import quality, units, wall
from . import measurements as wmeas
from . import objects
from . import quality as wquality
from .corrections import derived_image
from .quality import DuplexParam, IndicationParam, WallThicknessParam


def _warning(code: str, message: str) -> RecipeDiagnostic:
    """Return a warning diagnostic."""
    return RecipeDiagnostic("warning", code, message)


def suggest_radiographs(
    candidates: Sequence[ImageObj], slot: str
) -> dict[str, tuple[ImageObj, ...]]:
    """Assign the radiographs (images without a dark or flat role) to a slot."""
    images = tuple(
        image for image in candidates if objects.get_frame_role(image) in (None, "raw")
    )
    return {slot: images} if images else {}


def _image_slot(description: str) -> RecipeInputSlot:
    """Return the slot of the measured radiograph."""
    return RecipeInputSlot(
        "image",
        RecipeObjectType.IMAGE,
        RecipeCardinality.ONE,
        title="Radiograph",
        description=description,
    )


def _first(inputs: RecipeInputs) -> ImageObj:
    """Return the measured radiograph."""
    images = inputs.get("image", ())
    if not images:
        raise RecipeValidationError("Select a radiograph")
    return images[0]


def _measured(function, *args):
    """Call a measurement, converting its errors to recipe validation errors."""
    try:
        return function(*args)
    except (ValueError, KeyError) as exc:
        raise RecipeValidationError(str(exc)) from exc


def _plane_label(plane) -> str:
    """Return a plane title with its magnification."""
    return (
        plane.title
        if plane.factor is None
        else f"{plane.title}, M = {plane.factor:.4g}"
    )


# ---------------------------------------------------------------------------
# Duplex wire IQI
# ---------------------------------------------------------------------------


def run_duplex(
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
    context: RecipeExecutionContext,
) -> RecipeOutcome:
    """Measure the basic spatial resolution with a duplex wire IQI."""
    if not isinstance(parameters, DuplexParam):
        raise RecipeValidationError("Duplex IQI parameters are required")
    image = _first(inputs)
    context.report_progress(0.0, "Reading the IQI profile")
    plane = _measured(wmeas.plane_named, image, parameters.plane)
    profile, _coords, direction = _measured(
        wquality.box_profile, image, parameters, parameters.direction
    )
    step = wquality.plane_step(image, plane, direction)
    result = _measured(quality.analyze_duplex, profile, step, parameters.first_element)
    context.raise_if_cancelled()
    positions = np.arange(profile.size) * step
    curve = create_signal(
        f"Duplex IQI profile of {image.title}",
        positions,
        profile,
        units=("mm", image.zunit or ""),
        labels=(f"Position ({plane.title})", "Intensity"),
    )
    curve.metadata[md.OUTPUT_ROLE_KEY] = "duplex-profile"
    rows = []
    for element in result.elements:
        start = element.first * step
        end = start if element.second is None else element.second * step
        status = "resolved" if element.resolved else "unresolved"
        curve.add_graphical_annotation(
            RangeAnnotation(
                start=min(start, end),
                end=max(start, end),
                title=f"D{element.number}",
                label=AnnotationLabel(f"D{element.number}: {element.dip:.0f} %"),
            )
        )
        rows.append(
            [f"D{element.number}", 1000 * element.diameter_mm, element.dip, status]
        )
    elements = TableResult.from_rows(
        "Duplex IQI elements",
        ["Element", "Wire diameter (µm)", "Dip (%)", "Status"],
        rows,
        [-1] * len(rows),
        func_name="duplex_iqi_elements",
    )
    srb_plane = math.nan if result.srb_mm is None else 1000 * result.srb_mm
    factor = plane.factor if plane.factor is not None else math.nan
    srb_detector = srb_plane * factor
    summary = TableResult.from_rows(
        "Basic spatial resolution",
        [
            "iSRb",
            f"SRb in the IQI plane (µm, {_plane_label(plane)})",
            "SRb in the detector plane (µm)",
        ],
        [
            [
                "-" if result.isrb is None else f"D{result.isrb}",
                srb_plane,
                srb_detector,
            ]
        ],
        [-1],
        func_name="duplex_iqi_srb",
        attrs={"reversed": result.reversed, "noise": result.noise},
    )
    diagnostics = []
    if result.isrb is None:
        diagnostics.append(
            _warning(
                "all-elements-resolved",
                "Every element crossed by the profile is resolved: the basic "
                "spatial resolution is better than the last one",
            )
        )
    elif result.srb_mm is None:
        diagnostics.append(
            _warning(
                "first-element-unresolved",
                "The first element crossed by the profile is not resolved: check "
                "the region and the first element number",
            )
        )
    context.report_progress(1.0, "Done")
    return RecipeOutcome(
        objects=(RecipeObjectOutput("profile", curve),),
        results=(
            RecipeResultOutput("srb", summary, "profile"),
            RecipeResultOutput("elements", elements, "profile"),
        ),
        diagnostics=tuple(diagnostics),
    )


DUPLEX_RECIPE = RecipeDescriptor(
    recipe_id=f"{PLUGIN_ID}:duplex-iqi",
    plugin_version=__version__,
    title="Basic spatial resolution (duplex IQI)",
    version="1.0.0",
    description=(
        "Measure the basic spatial resolution with a duplex wire IQI: profile "
        "across the wire pairs in a rectangle, dip of each pair, first pair "
        "below 20 % (iSRb) and interpolated SRb in the IQI and detector planes "
        "(inspired by ISO 19232-5)."
    ),
    inputs=(_image_slot("Radiograph with a rectangle across the duplex IQI"),),
    parameter_class=DuplexParam,
    run=run_duplex,
    suggest_bindings=lambda candidates: suggest_radiographs(candidates, "image"),
)


# ---------------------------------------------------------------------------
# Wall thickness
# ---------------------------------------------------------------------------


def run_wall_thickness(
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
    context: RecipeExecutionContext,
) -> RecipeOutcome:
    """Measure the wall thickness of a pipe in tangential radiography."""
    if not isinstance(parameters, WallThicknessParam):
        raise RecipeValidationError("Wall thickness parameters are required")
    image = _first(inputs)
    context.report_progress(0.0, "Reading the pipe profile")
    plane = _measured(wmeas.plane_named, image, parameters.plane)
    profile, _coords, direction = _measured(
        wquality.box_profile, image, parameters, parameters.direction
    )
    step = wquality.plane_step(image, plane, direction)
    features, _darker = _measured(wall.attenuation_profile, profile)
    edges = _measured(wall.find_edges, features)
    context.raise_if_cancelled()
    context.report_progress(0.3, "Fitting the pipe model")
    fit = _measured(wall.fit_pipe, profile, edges)
    unit = parameters.unit
    scale = step / units.unit_mm(unit)
    positions = np.arange(profile.size) * scale
    curve = create_signal(
        f"Pipe profile of {image.title}",
        positions,
        profile,
        units=(unit, image.zunit or ""),
        labels=(f"Position ({plane.title})", "Intensity"),
    )
    curve.metadata[md.OUTPUT_ROLE_KEY] = "wall-profile"
    model = create_signal(
        f"Pipe model of {image.title}",
        positions,
        fit.fitted,
        units=(unit, image.zunit or ""),
        labels=(f"Position ({plane.title})", "Intensity"),
    )
    model.metadata[md.OUTPUT_ROLE_KEY] = "wall-model"
    model_wall = fit.wall(scale)
    walls = (
        (fit.center - fit.outer_radius, fit.center - fit.inner_radius),
        (fit.center + fit.inner_radius, fit.center + fit.outer_radius),
    )
    for number, (start, end) in enumerate(walls, start=1):
        curve.add_graphical_annotation(
            RangeAnnotation(
                start=start * scale,
                end=end * scale,
                title=f"Wall {number}",
                label=AnnotationLabel(f"Wall {number}: {model_wall:.3g} {unit}"),
            )
        )
    edge_walls = edges.walls(scale)
    edge_outer = edges.outer_diameter(scale)
    rows = [
        [
            "Model fit",
            model_wall,
            model_wall,
            2 * fit.outer_radius * scale,
            2 * fit.inner_radius * scale,
        ],
        [
            "Edges (steepest gradients)",
            edge_walls[0],
            edge_walls[1],
            edge_outer,
            edge_outer - edge_walls[0] - edge_walls[1],
        ],
    ]
    table = TableResult.from_rows(
        f"Wall thickness ({_plane_label(plane)})",
        [
            "Method",
            f"Wall 1 ({unit})",
            f"Wall 2 ({unit})",
            f"Outer diameter ({unit})",
            f"Inner diameter ({unit})",
        ],
        rows,
        [-1, -1],
        func_name="wall_thickness",
        attrs={"blur_samples": fit.blur, "rms": fit.rms},
    )
    diagnostics = []
    edge_mean = 0.5 * (edge_walls[0] + edge_walls[1])
    if abs(edge_mean - model_wall) > 0.1 * model_wall:
        diagnostics.append(
            _warning(
                "methods-disagree",
                "The edge and model methods differ by more than 10 %: check that "
                "the profile crosses the whole pipe perpendicularly to its axis",
            )
        )
    context.report_progress(1.0, "Done")
    return RecipeOutcome(
        objects=(
            RecipeObjectOutput("profile", curve),
            RecipeObjectOutput("model", model),
        ),
        results=(RecipeResultOutput("wall", table, "profile"),),
        diagnostics=tuple(diagnostics),
    )


WALL_THICKNESS_RECIPE = RecipeDescriptor(
    recipe_id=f"{PLUGIN_ID}:wall-thickness",
    plugin_version=__version__,
    title="Wall thickness (tangential)",
    version="1.0.0",
    description=(
        "Measure the wall thickness of a pipe seen tangentially, from a profile "
        "across the whole pipe, in the plane of its axis: fit of the projected "
        "annulus (corrects the blur), and steepest gradient edges (inspired by "
        "ISO 20769-1)."
    ),
    inputs=(_image_slot("Radiograph with a rectangle across the whole pipe"),),
    parameter_class=WallThicknessParam,
    run=run_wall_thickness,
    suggest_bindings=lambda candidates: suggest_radiographs(candidates, "image"),
)


# ---------------------------------------------------------------------------
# Indication detection
# ---------------------------------------------------------------------------


def _bounding_box(obj: ImageObj, region: cind.Region) -> tuple[float, ...]:
    """Return the center and size of the bounding box of a region (coordinates)."""
    c0, c1 = region.cols.min() - 0.5, region.cols.max() + 0.5
    r0, r1 = region.rows.min() - 0.5, region.rows.max() + 0.5
    x = obj.x0 + obj.dx * 0.5 * (c0 + c1)
    y = obj.y0 + obj.dy * 0.5 * (r0 + r1)
    return x, y, abs(obj.dx) * (c1 - c0), abs(obj.dy) * (r1 - r0)


def detect_indications(
    image: ImageObj, parameters: IndicationParam
) -> tuple[ImageObj, ImageObj, TableResult, int]:
    """Detect and size the indications of a radiograph.

    Returns:
        Annotated copy of the radiograph, label image, table and number of
        indications
    """
    plane = wmeas.plane_named(image, parameters.plane)
    if not image.is_uniform_coords:
        raise wquality.RegionError("Indications need an image with uniform coordinates")
    pixel = (plane.scale[0] * abs(image.dx), plane.scale[1] * abs(image.dy))
    mask = wquality.region_mask(image, parameters)
    size = wquality.background_pixels(image, parameters)
    contrast, _noise = cind.contrast_map(
        image.data,
        size,
        core_enhancement.BACKGROUND_MORPHOLOGICAL,
        mask,
        parameters.despeckle,
    )
    labels, regions = cind.detect(
        contrast,
        mask,
        parameters.polarity,
        parameters.detection,
        parameters.sizing,
        parameters.min_pixels,
    )
    unit = parameters.unit
    unit_mm = units.unit_mm(unit)
    annotated = derived_image(
        image, np.array(image.data, copy=True), f"Indications of {image.title}"
    )
    annotated.metadata[md.OUTPUT_ROLE_KEY] = "indications"
    rows = []
    for region in regions:
        size_mm = cind.region_size(region, pixel)
        name = f"I{region.label}"
        x = image.x0 + image.dx * float(region.cols.mean())
        y = image.y0 + image.dy * float(region.rows.mean())
        bx, by, width, height = _bounding_box(image, region)
        annotated.add_graphical_annotation(
            RectangleAnnotation(
                x=bx,
                y=by,
                width=width,
                height=height,
                title=name,
                label=AnnotationLabel(f"{name}: {size_mm.length / unit_mm:.3g} {unit}"),
            )
        )
        rows.append(
            [
                name,
                x,
                y,
                size_mm.length / unit_mm,
                size_mm.width / unit_mm,
                size_mm.area / unit_mm**2,
                size_mm.equivalent_diameter / unit_mm,
                region.contrast,
                int(region.rows.size),
            ]
        )
    table = TableResult.from_rows(
        f"Indications ({_plane_label(plane)})",
        [
            "Indication",
            "X",
            "Y",
            f"Length ({unit})",
            f"Width ({unit})",
            f"Area ({unit}²)",
            f"Equivalent diameter ({unit})",
            "Contrast (σ)",
            "Pixels",
        ],
        rows,
        [-1] * len(rows),
        func_name="indication_detection",
    )
    label_image = derived_image(
        image, labels.astype(np.float64), f"Indication labels of {image.title}", ""
    )
    label_image.metadata[md.OUTPUT_ROLE_KEY] = "indication-labels"
    return annotated, label_image, table, len(regions)


def run_indications(
    inputs: RecipeInputs,
    parameters: gds.DataSet | None,
    context: RecipeExecutionContext,
) -> RecipeOutcome:
    """Detect and size the indications of radiographs."""
    if not isinstance(parameters, IndicationParam):
        raise RecipeValidationError("Indication detection parameters are required")
    if not 0 < parameters.sizing <= parameters.detection:
        raise RecipeValidationError(
            "The sizing threshold must not exceed the detection threshold"
        )
    images = inputs.get("images", ())
    outputs, results, diagnostics = [], [], []
    for index, image in enumerate(images, start=1):
        context.raise_if_cancelled()
        context.report_progress(index / (len(images) + 1), f"Analyzing {image.title}")
        annotated, labels, table, count = _measured(
            detect_indications, image, parameters
        )
        anchor = f"indications-{index}"
        outputs.append(RecipeObjectOutput(anchor, annotated))
        if parameters.label_image:
            outputs.append(RecipeObjectOutput(f"labels-{index}", labels))
        results.append(RecipeResultOutput(f"table-{index}", table, anchor))
        if count == 0:
            diagnostics.append(
                RecipeDiagnostic(
                    "info", "no-indication", f"No indication found in {image.title}"
                )
            )
    context.report_progress(1.0, "Done")
    return RecipeOutcome(
        objects=tuple(outputs), results=tuple(results), diagnostics=tuple(diagnostics)
    )


INDICATION_RECIPE = RecipeDescriptor(
    recipe_id=f"{PLUGIN_ID}:indication-detection",
    plugin_version=__version__,
    title="Indication detection",
    version="1.0.0",
    description=(
        "Detect indications (pores, cracks, inclusions) deviating from the local "
        "background by several noise standard deviations, in the regions of "
        "interest, and size them in the plane of a magnification: length, width, "
        "area and equivalent diameter. No acceptance criterion is applied."
    ),
    inputs=(
        RecipeInputSlot(
            "images",
            RecipeObjectType.IMAGE,
            RecipeCardinality.MANY,
            title="Radiographs",
            description="Radiographs, with regions of interest on the inspected areas",
        ),
    ),
    parameter_class=IndicationParam,
    run=run_indications,
    suggest_bindings=lambda candidates: suggest_radiographs(candidates, "images"),
)

__all__ = ["DUPLEX_RECIPE", "INDICATION_RECIPE", "WALL_THICKNESS_RECIPE"]
