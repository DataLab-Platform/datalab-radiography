"""Validation of the magnification methods against the simulator truth."""

from __future__ import annotations

import pytest

from datalab_radiography import demo
from datalab_radiography.core import measurement as meas
from datalab_radiography.core import metadata as md
from datalab_radiography.core import simulation as sim
from datalab_radiography.workflow import magnification as wmag
from datalab_radiography.workflow import measurements as wmeas
from datalab_radiography.workflow import objects


def _truth(image) -> dict:
    return md.loads(image.metadata[md.SIMULATION_TRUTH_KEY])


def _component(truth: dict, name: str) -> dict:
    return [c for c in truth["components"] if c["name"] == name][0]


def _factor(truth: dict, plane: str) -> float:
    return [p for p in truth["geometry"]["planes"] if p["name"] == plane][0][
        "magnification"
    ]


def _to_pixels(image, truth: dict, plane: str, x_mm: float, y_mm: float):
    """Return the image coordinates of a point of an object plane."""
    factor = _factor(truth, plane)
    pitch = truth["detector"]["pitch_mm"]
    rows, cols = image.data.shape
    return (
        x_mm * factor / pitch + (cols - 1) / 2,
        y_mm * factor / pitch + (rows - 1) / 2,
    )


@pytest.mark.parametrize("example_id", ["simulated-weld", "simulated-pipe"])
def test_ball_magnification_matches_truth(example_id) -> None:
    """The calibration ball gives the magnification of its plane within 0.5 %."""
    image = demo.materialize_generated_example(example_id).objects[0]
    truth = _truth(image)
    ball = _component(truth, sim.BALL_PLANE)
    xc, yc = _to_pixels(image, truth, sim.BALL_PLANE, *ball["center_mm"])
    half = (ball["diameter_mm"] / 2 + 3.0) * _factor(truth, sim.BALL_PLANE) / 0.1
    param = wmag.MagnificationBallParam()
    param.name, param.true_diameter = "Fitted ball", ball["diameter_mm"]
    param.unit = "mm"
    box = (xc - half, yc - half, xc + half, yc + half)
    item, _fit = wmag.define_ball(image, param, box)
    assert item.factor == pytest.approx(_factor(truth, sim.BALL_PLANE), rel=5e-3)


def test_metadata_magnifications_match_truth() -> None:
    """Magnifications from the metadata are those of the simulated planes."""
    image = demo.materialize_generated_example("simulated-weld").objects[0]
    mset = objects.get_magnifications(image)
    for plane in _truth(image)["geometry"]["planes"]:
        assert mset.get(plane["name"]).factor == pytest.approx(plane["magnification"])
    assert mset.active == sim.WELD_PLANE


def test_measured_pore_in_object_plane() -> None:
    """A segment across a pore image measures its diameter in the weld plane."""
    image = demo.materialize_generated_example("simulated-weld").objects[0]
    truth = _truth(image)
    pore = _component(truth, "Pore 3")
    xc, yc = _to_pixels(image, truth, sim.WELD_PLANE, *pore["center_mm"])
    half = pore["diameter_mm"] / 2 * _factor(truth, sim.WELD_PLANE) / 0.1
    shapes = [meas.Segment("Pore", xc - half, yc, xc + half, yc)]
    planes, rows = wmeas.measure(image, shapes, unit="mm")
    weld = [i for i, p in enumerate(planes) if p.title == sim.WELD_PLANE][0]
    assert rows[0].values[weld] == pytest.approx(pore["diameter_mm"])
