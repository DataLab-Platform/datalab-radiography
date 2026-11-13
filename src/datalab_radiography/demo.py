"""Simulated examples of the radiography plugin, shared by Desktop and Web.

Each example is generated in memory by the radiography simulator with fixed
seeds: its images carry their acquisition geometry, detector pixel pitch,
magnifications, frame role and ground truth as plugin metadata, and the
radiographic display profile.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping

from datalab.plugins.examples import PluginExample, PluginExampleData
from sigima.objects import ImageObj, create_image, create_image_roi

from .core import metadata as md
from .core import simulation as sim
from .workflow import display as wdisplay
from .workflow import magnification as wmag
from .workflow import objects
from .workflow.quality import REGION_BOX
from .workflow.recipes import (
    DUPLEX_RECIPE,
    INDICATION_RECIPE,
    OFFSET_GAIN_RECIPE,
    STEP_WEDGE_RECIPE,
    WALL_THICKNESS_RECIPE,
)


def _box(x0: float, y0: float, x1: float, y1: float) -> dict[str, object]:
    """Return the parameter values of a measurement rectangle (pixel coordinates)."""
    return {"region": REGION_BOX, "x0": x0, "y0": y0, "x1": x1, "y1": y1}


#: Largest number of pixels of one acquisition (all frames together)
MAX_ACQUISITION_PIXELS = 30_000_000


def simulation_images(
    simulation: sim.Simulation,
    scene: sim.Scene,
    geometry: sim.Geometry,
    detector: sim.Detector,
    title: str,
    film_convention: bool = False,
) -> list[ImageObj]:
    """Return the frames of a simulation as Sigima images with plugin metadata.

    Args:
        simulation: simulated frames and truth
        scene: simulated scene
        geometry: acquisition geometry
        detector: detector model
        title: title of the radiographs
        film_convention: film convention of the display profile
    """
    images = []
    frames = (
        [("raw", frame) for frame in simulation.radiographs]
        + [("dark", frame) for frame in simulation.darks]
        + [("flat", frame) for frame in simulation.flats]
    )
    counts: dict[str, int] = {}
    titles = {"raw": title, "dark": "Dark", "flat": "Flat"}
    for role, frame in frames:
        counts[role] = counts.get(role, 0) + 1
        total = {"raw": len(simulation.radiographs), "dark": len(simulation.darks)}
        total["flat"] = len(simulation.flats)
        name = titles[role]
        if total[role] > 1:
            name = f"{name} {counts[role]}"
        image = create_image(name, frame, units=("", "", "DN"))
        objects.set_detector_pitch(image, detector.pitch_mm)
        objects.set_frame_role(image, role)
        objects.set_geometry(
            image,
            {
                "sdd_mm": geometry.sdd_mm,
                "focal_spot_mm": geometry.focal_spot_mm,
                "voltage_kv": geometry.voltage_kv,
                "planes": [
                    {"name": name, "sod_mm": sod} for name, sod in scene.planes.items()
                ],
            },
        )
        image.metadata[md.SIMULATION_TRUTH_KEY] = md.dumps(simulation.truth)
        if role == "raw":
            wmag.define_from_metadata(image)
            first_plane = next(iter(scene.planes))
            wmag.set_active_magnification(image, first_plane)
        wdisplay.apply_profile(image, film_convention)
        images.append(image)
    return images


def check_acquisition_size(detector: sim.Detector, frame_count: int) -> None:
    """Refuse acquisitions creating too many pixels.

    Raises:
        ValueError: if the acquisition is too large
    """
    pixels = detector.rows * detector.cols * frame_count
    if pixels > MAX_ACQUISITION_PIXELS:
        raise ValueError(
            f"The acquisition would create {pixels / 1e6:.0f} million pixels "
            f"(at most {MAX_ACQUISITION_PIXELS / 1e6:.0f} million): reduce the "
            "detector size or the number of frames"
        )


@dataclasses.dataclass(frozen=True)
class GeneratedExample:
    """Recipe of a generated example.

    Args:
        example: example declaration
        scene: function returning the scene
        detector: detector model
        geometry: acquisition geometry
        exposure: exposure
        radiographs: number of radiographs
        darks: number of dark frames
        flats: number of flat frames
        seed: seed of the frame noise
        title: title of the radiographs
        prepare: function completing the images (e.g. adding ROIs)
        parameter_values: recipe parameter values suited to the example
    """

    example: PluginExample
    scene: Callable[[], sim.Scene]
    detector: sim.Detector = sim.Detector()
    geometry: sim.Geometry = sim.Geometry()
    exposure: sim.Exposure = sim.Exposure()
    radiographs: int = 1
    darks: int = 0
    flats: int = 0
    seed: int = 0
    title: str = "Radiograph"
    prepare: Callable[[list[ImageObj], sim.Simulation, sim.Scene], None] | None = None
    parameter_values: Mapping[str, Mapping[str, object]] = dataclasses.field(
        default_factory=dict
    )

    def materialize(self) -> PluginExampleData:
        """Generate the images of the example."""
        scene = self.scene()
        simulation = sim.simulate(
            scene,
            self.geometry,
            self.detector,
            self.exposure,
            self.radiographs,
            self.darks,
            self.flats,
            self.seed,
        )
        images = simulation_images(
            simulation, scene, self.geometry, self.detector, self.title
        )
        if self.prepare is not None:
            self.prepare(images, simulation, scene)
        return PluginExampleData(images, self.parameter_values)


def plane_to_image(
    image: ImageObj, truth: dict, plane: str, x_mm: float, y_mm: float
) -> tuple[float, float]:
    """Return the image coordinates of a point of a simulated object plane."""
    factor = [p for p in truth["geometry"]["planes"] if p["name"] == plane][0][
        "magnification"
    ]
    pitch = truth["detector"]["pitch_mm"]
    rows, cols = image.data.shape
    return (
        x_mm * factor / pitch + (cols - 1) / 2,
        y_mm * factor / pitch + (rows - 1) / 2,
    )


def _step_wedge_rois(
    images: list[ImageObj], simulation: sim.Simulation, scene: sim.Scene
) -> None:
    """Add one ROI per step (inset from the step edges) to the radiograph."""
    wedge = scene.components[0]
    radiograph = images[0]
    roi = None
    for index, (x0, x1, y0, y1) in enumerate(wedge.step_boxes(), start=1):
        inset_x, inset_y = 0.2 * (x1 - x0), 0.2 * (y1 - y0)
        xa, ya = plane_to_image(
            radiograph, simulation.truth, wedge.plane, x0 + inset_x, y0 + inset_y
        )
        xb, yb = plane_to_image(
            radiograph, simulation.truth, wedge.plane, x1 - inset_x, y1 - inset_y
        )
        single = create_image_roi(
            "rectangle", [xa, ya, xb - xa, yb - ya], title=f"Step {index}"
        )
        if roi is None:
            roi = single
        else:
            roi.add_roi(single.single_rois[0])
    radiograph.roi = roi


WELD_EXAMPLE = PluginExample(
    id="simulated-weld",
    title="Simulated weld",
    description=(
        "Butt weld with pores and a crack, wire and duplex IQIs, and a "
        "calibration ball closer to the detector (two magnifications)"
    ),
    recipe_ids=(DUPLEX_RECIPE.recipe_id, INDICATION_RECIPE.recipe_id),
)
PIPE_EXAMPLE = PluginExample(
    id="simulated-pipe",
    title="Simulated pipe (tangential view)",
    description=(
        "Steel pipe seen tangentially, to measure its wall thickness, with a "
        "calibration ball on the detector (two magnifications)"
    ),
    recipe_ids=(WALL_THICKNESS_RECIPE.recipe_id,),
)
STEP_WEDGE_EXAMPLE = PluginExample(
    id="simulated-step-wedge",
    title="Simulated step wedge",
    description=(
        "Aluminum step wedge of 2 to 20 mm with one ROI per step, and dark and "
        "flat frames, to calibrate thickness"
    ),
    recipe_ids=(STEP_WEDGE_RECIPE.recipe_id,),
)
CAMPAIGN_EXAMPLE = PluginExample(
    id="simulated-campaign",
    title="Simulated correction campaign",
    description=(
        "Radiographs of a cast plate with dark and flat frames, on a detector "
        "with non-uniform gain and defective pixels"
    ),
    recipe_ids=(OFFSET_GAIN_RECIPE.recipe_id,),
)

GENERATED_EXAMPLES: dict[str, GeneratedExample] = {
    WELD_EXAMPLE.id: GeneratedExample(
        WELD_EXAMPLE,
        sim.weld_scene,
        sim.Detector(rows=400, cols=500),
        exposure=sim.Exposure(photons_per_pixel=6000.0, frames=4),
        title="Weld",
        parameter_values={
            # Rectangle across the duplex IQI, and around the weld bead
            DUPLEX_RECIPE.recipe_id: _box(40.0, 35.0, 375.0, 90.0),
            INDICATION_RECIPE.recipe_id: _box(15.0, 150.0, 485.0, 249.0),
        },
    ),
    PIPE_EXAMPLE.id: GeneratedExample(
        PIPE_EXAMPLE,
        sim.pipe_scene,
        sim.Detector(rows=360, cols=640),
        exposure=sim.Exposure(photons_per_pixel=8000.0, frames=4),
        seed=1,
        title="Pipe",
        parameter_values={
            # Rectangle across the whole pipe, away from the calibration ball
            WALL_THICKNESS_RECIPE.recipe_id: _box(0.0, 200.0, 639.0, 340.0),
        },
    ),
    STEP_WEDGE_EXAMPLE.id: GeneratedExample(
        STEP_WEDGE_EXAMPLE,
        sim.step_wedge_scene,
        sim.Detector(rows=300, cols=520),
        exposure=sim.Exposure(photons_per_pixel=8000.0, frames=4),
        darks=4,
        flats=4,
        seed=2,
        title="Step wedge",
        prepare=_step_wedge_rois,
        parameter_values={
            STEP_WEDGE_RECIPE.recipe_id: {
                "thicknesses": ", ".join(str(t) for t in range(2, 22, 2)),
                "unit": "mm",
                "degree": 2,
            }
        },
    ),
    CAMPAIGN_EXAMPLE.id: GeneratedExample(
        CAMPAIGN_EXAMPLE,
        sim.plate_scene,
        sim.Detector(rows=320, cols=400, dead_fraction=1e-3, hot_fraction=1e-3),
        exposure=sim.Exposure(photons_per_pixel=5000.0),
        radiographs=2,
        darks=4,
        flats=4,
        seed=3,
        title="Plate",
    ),
}

#: Workspace files of the real-data examples (``examples`` package folder),
#: built from the test corpus by ``scripts/build_examples.py``
REAL_EXAMPLE_FILES = {
    "walnut": "walnut.h5",
    "circuit-board": "circuit_board.h5",
    "knee-phantom": "knee_phantom.h5",
}
EXAMPLE_PACKAGE = "datalab_radiography"


def _resource(example_id: str) -> str:
    """Return the package resource of a real-data example."""
    return f"{EXAMPLE_PACKAGE}:examples/{REAL_EXAMPLE_FILES[example_id]}"


WALNUT_EXAMPLE = PluginExample(
    id="walnut",
    title="Walnut (raw flat-panel radiograph)",
    resource=_resource("walnut"),
    description=(
        "Raw radiograph of a walnut with its dark and flat images, magnification "
        "3.015 (CWI FleX-ray Lab, CC BY 4.0)"
    ),
    recipe_ids=(OFFSET_GAIN_RECIPE.recipe_id,),
)
CIRCUIT_BOARD_EXAMPLE = PluginExample(
    id="circuit-board",
    title="Circuit board (laminography, 45° tilt)",
    resource=_resource("circuit-board"),
    description=(
        "Radiograph of a keyboard circuit board tilted by 45°, magnification "
        "6.95, free beam 1992 DN (LUT University, CC BY 4.0)"
    ),
)
KNEE_PHANTOM_EXAMPLE = PluginExample(
    id="knee-phantom",
    title="Knee phantom (cone-beam CT projection)",
    resource=_resource("knee-phantom"),
    description=(
        "Flat-field corrected projection of a knee phantom, magnification 1.707 "
        "(Zenodo 10.5281/zenodo.12722386, CC BY 4.0)"
    ),
)
REAL_EXAMPLES = (WALNUT_EXAMPLE, CIRCUIT_BOARD_EXAMPLE, KNEE_PHANTOM_EXAMPLE)

EXAMPLES: tuple[PluginExample, ...] = (
    *(generated.example for generated in GENERATED_EXAMPLES.values()),
    *REAL_EXAMPLES,
)


def materialize_generated_example(example_id: str) -> PluginExampleData | None:
    """Generate an example in memory (None for an unknown example)."""
    generated = GENERATED_EXAMPLES.get(example_id)
    return None if generated is None else generated.materialize()
