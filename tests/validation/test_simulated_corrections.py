"""Validation of corrections and thickness calibration on simulated campaigns."""

from __future__ import annotations

import numpy as np
import pytest
from datalab.plugins.recipes import RecipeExecutionContext

from datalab_radiography import demo
from datalab_radiography.core import metadata as md
from datalab_radiography.core import simulation as sim
from datalab_radiography.workflow import correction_recipes as recipes
from datalab_radiography.workflow import corrections as wcorr


def test_campaign_correction_removes_detector_patterns() -> None:
    """Gain non-uniformity, heel effect and defects are corrected."""
    images = list(demo.materialize_generated_example("simulated-campaign").objects)
    inputs = wcorr.suggest_frame_bindings(images)
    param = recipes.OffsetGainRecipeParam()
    param.output = "transmission"
    outcome = recipes.run_offset_gain(inputs, param, RecipeExecutionContext())
    transmission = outcome.objects[0].value.data
    truth = md.loads(images[0].metadata[md.SIMULATION_TRUTH_KEY])
    mu, beta = sim.MATERIALS["aluminum"]
    plate = truth["components"][0]["thickness_mm"]
    expected = np.exp(-mu * plate / (1 + beta * plate))
    # Region of the plate without pores, with known transmission
    pores = [c for c in truth["components"] if c["name"].startswith("Pore")]
    rows, cols = transmission.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    clear = np.ones(transmission.shape, dtype=bool)
    factor = truth["geometry"]["planes"][0]["magnification"]
    for pore in pores:
        x, y = (np.array(pore["center_mm"]) * factor / 0.1) + [
            (cols - 1) / 2,
            (rows - 1) / 2,
        ]
        radius = pore["diameter_mm"] / 2 * factor / 0.1 + 3
        clear &= (xx - x) ** 2 + (yy - y) ** 2 > radius**2
    values = transmission[clear]
    assert np.nanmedian(values) == pytest.approx(expected, rel=0.01)
    # Residual non-uniformity: noise only (no gain pattern or heel effect)
    raw = images[0].data.astype(float)[clear]
    assert np.nanstd(values) / np.nanmean(values) < 0.03
    assert np.std(raw[raw > 0]) / np.mean(raw[raw > 0]) > 0.04
    assert np.isfinite(transmission).all()


def test_step_wedge_thickness_map() -> None:
    """The calibrated thickness of each step is within 1 %."""
    data = demo.materialize_generated_example("simulated-step-wedge")
    images = list(data.objects)
    inputs = recipes.suggest_step_wedge_bindings(images)
    param = recipes.StepWedgeParam()
    for name, value in data.values_for(recipes.STEP_WEDGE_RECIPE.recipe_id).items():
        setattr(param, name, value)
    outcome = recipes.run_step_wedge(inputs, param, RecipeExecutionContext())
    assert not outcome.diagnostics
    wedge = images[0]
    thickness = outcome.objects[1].value.data
    truth = md.loads(wedge.metadata[md.SIMULATION_TRUTH_KEY])
    expected = truth["components"][0]["steps_mm"]
    for roi, true_thickness in zip(wedge.roi.single_rois, expected):
        inside = ~roi.to_mask(wedge)
        assert np.median(thickness[inside]) == pytest.approx(true_thickness, rel=0.01)
