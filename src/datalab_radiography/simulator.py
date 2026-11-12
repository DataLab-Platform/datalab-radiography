"""Radiography simulator exposed as a DataLab instrument (Desktop and Web).

The simulator behaves like a radiography bench: its settings describe the
scene, the geometry, the source and the detector; the live view shows a
radiograph, and each acquisition adds radiographs, dark and flat frames with
their geometry, magnifications and ground truth.
"""

from __future__ import annotations

import itertools

import guidata.dataset as gds
from datalab.plugins.instruments import (
    InstrumentAcquisition,
    InstrumentFrame,
    PluginInstrument,
)
from datalab.plugins.recipes import RecipeObjectType
from datalab.plugins.tools import PluginTool

from .core import display
from .core import simulation as sim
from .demo import check_acquisition_size, simulation_images

SIMULATOR_TOOL = PluginTool(
    id="xray-simulator",
    title="X-ray radiography simulator...",
    description=(
        "Acquire radiographs, dark and flat frames of simulated welds, pipes, "
        "step wedges and castings, with live view"
    ),
    icon="datalab_radiography:icons/xray_simulator.svg",
    instrument="xray_simulator",
    object_type=RecipeObjectType.IMAGE,
)

SCENE_CHOICES = (
    ("weld", "Butt weld with pores, crack, IQIs and calibration ball"),
    ("pipe", "Pipe seen tangentially, with calibration ball"),
    ("step-wedge", "Aluminum step wedge"),
    ("plate", "Cast aluminum plate with pores"),
)
SCENE_TITLES = {
    "weld": "Weld",
    "pipe": "Pipe",
    "step-wedge": "Step wedge",
    "plate": "Plate",
}

#: Live frames use these noise seeds in turn; acquisitions use the next ones
LIVE_SEEDS = 16


class SimulatorSettings(gds.DataSet):
    """X-ray radiography simulator"""

    _tabs = gds.BeginTabGroup("Settings")

    _object = gds.BeginGroup("Object")
    scene = gds.ChoiceItem("Scene", SCENE_CHOICES, default="weld")
    sod = gds.FloatItem(
        "Source-to-object distance",
        default=800.0,
        min=1.0,
        unit="mm",
        help="Distance of the main object plane (weld, pipe axis, wedge, plate)",
    )
    ball_sod = gds.FloatItem(
        "Source-to-ball distance",
        default=960.0,
        min=1.0,
        unit="mm",
        help="Distance of the calibration ball plane (weld and pipe scenes)",
    )
    _object_end = gds.EndGroup("Object")

    _source = gds.BeginGroup("Source")
    sdd = gds.FloatItem(
        "Source-to-detector distance", default=1000.0, min=1.0, unit="mm"
    )
    focal_spot = gds.FloatItem("Focal spot size", default=0.4, min=0.0, unit="mm")
    voltage = gds.FloatItem(
        "Tube voltage",
        default=160.0,
        min=10.0,
        max=1000.0,
        unit="kV",
        help="Informative: the attenuation model uses effective coefficients",
    )
    photons = gds.FloatItem(
        "Photons per pixel",
        default=6000.0,
        min=10.0,
        max=1e6,
        help="Mean detected photons per pixel in the free beam",
    )
    frames = gds.IntItem(
        "Averaged frames", default=1, min=1, max=64, help="Frames averaged per image"
    )
    _source_end = gds.EndGroup("Source")

    _detector = gds.BeginGroup("Detector")
    rows = gds.IntItem("Rows", default=400, min=32, max=4096)
    cols = gds.IntItem("Columns", default=500, min=32, max=4096)
    pitch = gds.FloatItem("Pixel pitch", default=0.1, min=0.005, max=2.0, unit="mm")
    blur = gds.FloatItem(
        "Blur", default=0.7, min=0.0, max=10.0, unit="pixel", help="Gaussian sigma"
    )
    read_noise = gds.FloatItem("Read noise", default=4.0, min=0.0, unit="DN")
    gain_nonuniformity = gds.FloatItem(
        "Gain non-uniformity", default=4.0, min=0.0, max=50.0, unit="%"
    )
    heel = gds.FloatItem("Heel effect", default=15.0, min=0.0, max=80.0, unit="%")
    defects = gds.FloatItem(
        "Defective pixels", default=0.04, min=0.0, max=5.0, unit="%"
    )
    _detector_end = gds.EndGroup("Detector")

    _acquisition = gds.BeginGroup("Acquisition")
    radiographs = gds.IntItem("Radiographs", default=1, min=0, max=100)
    darks = gds.IntItem("Dark frames", default=0, min=0, max=100)
    flats = gds.IntItem("Flat frames", default=0, min=0, max=100)
    _acquisition_end = gds.EndGroup("Acquisition")

    _tabs_end = gds.EndTabGroup("Settings")


def scene_from(settings: SimulatorSettings) -> sim.Scene:
    """Return the scene described by the settings."""
    if settings.scene == "weld":
        return sim.weld_scene(settings.sod, settings.ball_sod)
    if settings.scene == "pipe":
        return sim.pipe_scene(settings.sod, ball_sod_mm=settings.ball_sod)
    if settings.scene == "step-wedge":
        return sim.step_wedge_scene(settings.sod)
    return sim.plate_scene(settings.sod)


def detector_from(settings: SimulatorSettings) -> sim.Detector:
    """Return the detector described by the settings."""
    fraction = settings.defects / 200.0
    return sim.Detector(
        rows=settings.rows,
        cols=settings.cols,
        pitch_mm=settings.pitch,
        blur_px=settings.blur,
        read_noise_dn=settings.read_noise,
        gain_nonuniformity=settings.gain_nonuniformity / 100.0,
        heel=settings.heel / 100.0,
        dead_fraction=fraction,
        hot_fraction=fraction,
    )


def geometry_from(settings: SimulatorSettings) -> sim.Geometry:
    """Return the geometry described by the settings."""
    return sim.Geometry(settings.sdd, settings.focal_spot, settings.voltage)


def summary(scene: sim.Scene, geometry: sim.Geometry) -> str:
    """Return the magnification and unsharpness of each plane."""
    parts = []
    for plane, factor in sim.magnifications(scene, geometry).items():
        ug = geometry.focal_spot_mm * (factor - 1.0)
        parts.append(f"{plane}: M = {factor:.4g}, Ug = {ug:.3g} mm")
    return " — ".join(parts)


class RadiographySimulator(PluginInstrument):
    """Radiography bench simulator."""

    live_interval_ms = 400

    def __init__(self) -> None:
        super().__init__(SimulatorSettings())
        self._live_seeds = itertools.cycle(range(LIVE_SEEDS))
        self._acquisition_seeds = itertools.count(LIVE_SEEDS)

    def _setup(self) -> tuple[sim.Scene, sim.Geometry, sim.Detector, sim.Exposure]:
        """Return the simulation inputs, validated."""
        settings = self.settings
        scene = scene_from(settings)
        geometry = geometry_from(settings)
        sim.magnifications(scene, geometry)
        exposure = sim.Exposure(settings.photons, settings.frames)
        return scene, geometry, detector_from(settings), exposure

    def preview(self) -> InstrumentFrame:
        """Return a live radiograph."""
        scene, geometry, detector, exposure = self._setup()
        simulation = sim.simulate(
            scene, geometry, detector, exposure, seed=next(self._live_seeds)
        )
        frame = simulation.radiographs[0]
        image = simulation_images(
            simulation, scene, geometry, detector, SCENE_TITLES[scene.name]
        )[0]
        window = display.percentile_window(frame)
        return InstrumentFrame(
            (image,), summary(scene, geometry), (window.low, window.high)
        )

    def acquire(self) -> InstrumentAcquisition:
        """Acquire radiographs, dark and flat frames."""
        settings = self.settings
        scene, geometry, detector, exposure = self._setup()
        count = settings.radiographs + settings.darks + settings.flats
        if count == 0:
            raise ValueError("Choose at least one radiograph, dark or flat frame")
        check_acquisition_size(detector, count)
        simulation = sim.simulate(
            scene,
            geometry,
            detector,
            exposure,
            settings.radiographs,
            settings.darks,
            settings.flats,
            seed=next(self._acquisition_seeds),
        )
        images = simulation_images(
            simulation, scene, geometry, detector, SCENE_TITLES[scene.name]
        )
        return InstrumentAcquisition(f"Simulated {SCENE_TITLES[scene.name]}", images)
