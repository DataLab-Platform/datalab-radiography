"""Validation of image quality and NDT measurements against simulated truth."""

from __future__ import annotations

import numpy as np
import pytest
from datalab.plugins.recipes import RecipeExecutionContext

from datalab_radiography import demo
from datalab_radiography.core import metadata as md
from datalab_radiography.core import quality, wall
from datalab_radiography.core import simulation as sim
from datalab_radiography.workflow import ndt_recipes as recipes
from datalab_radiography.workflow import quality as wquality

WELD = demo.GENERATED_EXAMPLES["simulated-weld"]
PIPE = demo.GENERATED_EXAMPLES["simulated-pipe"]


def weld_duplex(noise: bool, focal_spot: float = 0.4) -> quality.DuplexResult:
    """Return the duplex analysis of the simulated weld."""
    geometry = sim.Geometry(focal_spot_mm=focal_spot)
    simulation = sim.simulate(
        WELD.scene(), geometry, WELD.detector, WELD.exposure, seed=0, noise=noise
    )
    profile = quality.robust_profile(simulation.radiographs[0][35:90, 40:375], 0)
    factor = WELD.geometry.sdd_mm / WELD.scene().planes[sim.WELD_PLANE]
    return quality.analyze_duplex(profile, WELD.detector.pitch_mm / factor)


def test_duplex_srb_matches_the_noise_free_radiograph() -> None:
    """The SRb of the noisy weld is within one element of the noise-free one."""
    noisy, clean = weld_duplex(True), weld_duplex(False)
    assert abs(noisy.isrb - clean.isrb) <= 1
    assert noisy.srb_mm == pytest.approx(clean.srb_mm, rel=0.1)
    worse = weld_duplex(True, focal_spot=1.5)
    assert worse.srb_mm > 1.2 * noisy.srb_mm


@pytest.mark.parametrize(
    ("outer_diameter", "wall_mm"), [(28.0, 2.0), (28.0, 3.5), (20.0, 3.5), (28.0, 5.0)]
)
def test_wall_thickness_within_two_percent(outer_diameter, wall_mm) -> None:
    """The model fit measures the wall within 2 % in the plane of the pipe axis."""
    scene = sim.pipe_scene(outer_diameter_mm=outer_diameter, wall_mm=wall_mm)
    simulation = sim.simulate(scene, PIPE.geometry, PIPE.detector, PIPE.exposure)
    profile = quality.robust_profile(simulation.radiographs[0][200:340], 0)
    factor = PIPE.geometry.sdd_mm / scene.planes[sim.PIPE_PLANE]
    step = PIPE.detector.pitch_mm / factor
    features, _darker = wall.attenuation_profile(profile)
    edges = wall.find_edges(features)
    fit = wall.fit_pipe(profile, edges)
    assert fit.wall(step) == pytest.approx(wall_mm, rel=0.02)
    assert 2 * fit.outer_radius * step == pytest.approx(outer_diameter, rel=0.005)
    for value in edges.walls(step):
        assert value == pytest.approx(wall_mm, rel=0.03)


def test_weld_indications_are_found_and_sized() -> None:
    """Pores and crack of the simulated weld are found, without false call."""
    data = demo.materialize_generated_example("simulated-weld")
    weld = data.objects[0]
    param = wquality.IndicationParam()
    for name, value in data.values_for(recipes.INDICATION_RECIPE.recipe_id).items():
        setattr(param, name, value)
    outcome = recipes.INDICATION_RECIPE.run(
        {"images": (weld,)}, param, RecipeExecutionContext()
    )
    table = outcome.results[0].value
    truth = md.loads(weld.metadata[md.SIMULATION_TRUTH_KEY])
    factor = truth["geometry"]["planes"][0]["magnification"]
    pixel = WELD.detector.pitch_mm / factor
    rows, cols = weld.data.shape

    def position(row) -> np.ndarray:
        """Return the position of an indication in mm in the weld plane."""
        return (np.array(row[1:3]) - [(cols - 1) / 2, (rows - 1) / 2]) * pixel

    found = list(table.data)
    assert len(found) == 4
    for component in truth["components"]:
        if component["name"].startswith("Pore"):
            center = np.array(component["center_mm"])
            row = min(found, key=lambda item: np.hypot(*(position(item) - center)))
            assert np.hypot(*(position(row) - center)) < 0.15
            diameter = component["diameter_mm"]
            tolerance = 0.1 if diameter >= 1.0 else 0.35
            assert row[6] == pytest.approx(diameter, rel=tolerance)
            found.remove(row)
        elif component["name"] == "Crack":
            xa, ya, xb, yb = component["segment_mm"]
            center = np.array([(xa + xb) / 2, (ya + yb) / 2])
            row = min(found, key=lambda item: np.hypot(*(position(item) - center)))
            assert np.hypot(*(position(row) - center)) < 0.3
            assert row[3] == pytest.approx(component["length_mm"], rel=0.1)
            assert row[3] > 5 * row[4]
            found.remove(row)
    assert not found
