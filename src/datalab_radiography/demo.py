"""Simulated examples of the radiography plugin, shared by Desktop and Web.

Each example is generated in memory by the radiography simulator with fixed
seeds: its images carry their acquisition geometry, detector pixel pitch,
magnifications, frame role and ground truth as plugin metadata, and the
radiographic display profile.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from datalab.plugins.examples import PluginExample, PluginExampleData
from sigima.objects import ImageObj, create_image

from .core import metadata as md
from .core import simulation as sim
from .workflow import display as wdisplay
from .workflow import magnification as wmag
from .workflow import objects

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
        return PluginExampleData(
            simulation_images(
                simulation, scene, self.geometry, self.detector, self.title
            )
        )


WELD_EXAMPLE = PluginExample(
    id="simulated-weld",
    title="Simulated weld",
    description=(
        "Butt weld with pores and a crack, wire and duplex IQIs, and a "
        "calibration ball closer to the detector (two magnifications)"
    ),
)
PIPE_EXAMPLE = PluginExample(
    id="simulated-pipe",
    title="Simulated pipe (tangential view)",
    description=(
        "Steel pipe seen tangentially, to measure its wall thickness, with a "
        "calibration ball on the detector (two magnifications)"
    ),
)
STEP_WEDGE_EXAMPLE = PluginExample(
    id="simulated-step-wedge",
    title="Simulated step wedge",
    description="Aluminum step wedge of 2 to 20 mm, to calibrate thickness",
)
CAMPAIGN_EXAMPLE = PluginExample(
    id="simulated-campaign",
    title="Simulated correction campaign",
    description=(
        "Radiographs of a cast plate with dark and flat frames, on a detector "
        "with non-uniform gain and defective pixels"
    ),
)

GENERATED_EXAMPLES: dict[str, GeneratedExample] = {
    WELD_EXAMPLE.id: GeneratedExample(
        WELD_EXAMPLE,
        sim.weld_scene,
        sim.Detector(rows=400, cols=500),
        exposure=sim.Exposure(photons_per_pixel=6000.0, frames=4),
        title="Weld",
    ),
    PIPE_EXAMPLE.id: GeneratedExample(
        PIPE_EXAMPLE,
        sim.pipe_scene,
        sim.Detector(rows=360, cols=640),
        exposure=sim.Exposure(photons_per_pixel=8000.0, frames=4),
        seed=1,
        title="Pipe",
    ),
    STEP_WEDGE_EXAMPLE.id: GeneratedExample(
        STEP_WEDGE_EXAMPLE,
        sim.step_wedge_scene,
        sim.Detector(rows=300, cols=520),
        geometry=sim.Geometry(voltage_kv=80.0),
        exposure=sim.Exposure(photons_per_pixel=8000.0, frames=4),
        seed=2,
        title="Step wedge",
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

EXAMPLES: tuple[PluginExample, ...] = tuple(
    generated.example for generated in GENERATED_EXAMPLES.values()
)


def materialize_generated_example(example_id: str) -> PluginExampleData | None:
    """Generate an example in memory (None for an unknown example)."""
    generated = GENERATED_EXAMPLES.get(example_id)
    return None if generated is None else generated.materialize()
