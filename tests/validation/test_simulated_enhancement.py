"""Validation of the enhancement of indications on simulated radiographs."""

from __future__ import annotations

import numpy as np

from datalab_radiography import demo
from datalab_radiography.core import display as cdisplay
from datalab_radiography.core import enhancement as cenh
from datalab_radiography.core import metadata as md
from datalab_radiography.core import simulation as sim
from datalab_radiography.workflow import computations, objects
from datalab_radiography.workflow import enhancement as wenh

PLANE = "Casting"


def stepped_casting():
    """Return a radiograph of an aluminum step wedge with a pore in each step."""
    components = [
        sim.StepWedge(
            "Wedge",
            PLANE,
            "aluminum",
            x0=-20.0,
            step_width=8.0,
            y0=-10.0,
            y1=10.0,
            steps=(4.0, 8.0, 12.0, 16.0, 20.0),
        )
    ]
    for index, x in enumerate((-16.0, -8.0, 0.0, 8.0, 16.0)):
        components.append(
            sim.Sphere(
                f"Pore {index + 1}", PLANE, "aluminum", -1.0, x=x, y=2.0, radius=0.6
            )
        )
    scene = sim.Scene("casting", {PLANE: 900.0}, tuple(components), "Stepped casting")
    geometry = sim.Geometry(voltage_kv=80.0)
    detector = sim.Detector(rows=300, cols=520)
    simulation = sim.simulate(
        scene, geometry, detector, sim.Exposure(photons_per_pixel=20000.0, frames=4)
    )
    return demo.simulation_images(simulation, scene, geometry, detector, "Casting")[0]


def pore_regions(image) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return the core and surrounding ring of each simulated pore."""
    truth = md.loads(image.metadata[md.SIMULATION_TRUTH_KEY])
    factor = truth["geometry"]["planes"][0]["magnification"]
    pitch = objects.get_detector_pitch(image)[0]
    rows, cols = image.data.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    regions = []
    for pore in truth["components"]:
        if not pore["name"].startswith("Pore"):
            continue
        x, y = np.array(pore["center_mm"]) * factor / pitch + [
            (cols - 1) / 2,
            (rows - 1) / 2,
        ]
        radius = pore["diameter_mm"] / 2 * factor / pitch
        distance = np.hypot(xx - x, yy - y)
        ring = (distance > radius + 2) & (distance < radius + 5)
        regions.append((distance < 0.5 * radius, ring))
    return regions


def visibility(data: np.ndarray, image, region: np.ndarray) -> np.ndarray:
    """Return the pore contrasts relative to the display window of a region."""
    window = cdisplay.percentile_window(data[region], 0.5, 99.5)
    return np.array(
        [
            abs(np.mean(data[core]) - np.mean(data[ring])) / (window.high - window.low)
            for core, ring in pore_regions(image)
        ]
    )


def test_morphological_flattening_reveals_pores_in_every_step() -> None:
    """With steps kept in the background, pores of all steps share one window."""
    image = stepped_casting()
    rows, cols = image.data.shape
    wedge = np.zeros(image.data.shape, dtype=bool)
    wedge[rows // 2 - 95 : rows // 2 + 95, cols // 2 - 210 : cols // 2 + 210] = True
    before = visibility(image.data.astype(float), image, wedge)
    flattened = {}
    for method in cenh.BACKGROUND_METHODS:
        param = wenh.FlattenBackgroundParam.create(size=2.0, unit="mm", method=method)
        result = computations.flatten_background(image, param)
        flattened[method] = visibility(result.data, image, wedge)
    assert len(before) == 5
    morphological = flattened[cenh.BACKGROUND_MORPHOLOGICAL]
    assert np.all(morphological > 3.0 * before)
    assert np.all(morphological > 1.5 * flattened[cenh.BACKGROUND_GAUSSIAN])


def test_unsharp_mask_sharpens_weld_pores() -> None:
    """The unsharp mask increases the contrast of the pores of the weld."""
    weld = demo.materialize_generated_example("simulated-weld").objects[0]
    flat = computations.flatten_background(
        weld, wenh.FlattenBackgroundParam.create(size=2.0, unit="mm")
    )
    sharp = computations.unsharp_mask(
        flat, wenh.UnsharpMaskParam.create(sigma=0.3, unit="mm", amount=1.0)
    )
    regions = pore_regions(weld)
    assert len(regions) == 3
    for core, ring in regions:
        contrast = abs(np.mean(flat.data[core]) - np.mean(flat.data[ring]))
        sharpened = abs(np.mean(sharp.data[core]) - np.mean(sharp.data[ring]))
        assert sharpened > 1.2 * contrast
