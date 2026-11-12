"""Tests of the detector corrections and physical quantities."""

from __future__ import annotations

import math

import numpy as np
import pytest
from datalab.plugins.recipes import RecipeExecutionContext, RecipeValidationError
from sigima.objects import create_image, create_image_roi

from datalab_radiography.core import attenuation as catt
from datalab_radiography.core import corrections as ccorr
from datalab_radiography.core import simulation as sim
from datalab_radiography.workflow import computations
from datalab_radiography.workflow import correction_recipes as recipes
from datalab_radiography.workflow import corrections as wcorr


def test_mean_frame_and_flat_field() -> None:
    """The offset and gain correction recovers the transmission."""
    rng = np.random.default_rng(0)
    gain = rng.uniform(0.8, 1.2, (40, 50))
    dark = np.full((40, 50), 100.0)
    flat = dark + 1000.0 * gain
    transmission = np.full((40, 50), 0.5)
    raw = dark + 1000.0 * gain * transmission
    with pytest.raises(ValueError):
        ccorr.mean_frame([])
    with pytest.raises(ValueError, match="same shape"):
        ccorr.mean_frame([dark, dark[:10]])
    assert np.allclose(ccorr.mean_frame([dark, dark + 2]), dark + 1)
    result, report = ccorr.flat_field_correct(raw, dark, flat, "transmission")
    assert np.allclose(result, 0.5)
    assert report.dark_level == 100.0 and report.negative_fraction == 0.0
    corrected, _report = ccorr.flat_field_correct(raw, dark, flat)
    assert np.allclose(corrected, 0.5 * np.median(flat - dark))
    flat[0, 0] = dark[0, 0]
    result, report = ccorr.flat_field_correct(raw, dark, flat, "transmission")
    assert math.isnan(result[0, 0]) and report.invalid_gain_fraction > 0
    with pytest.raises(ValueError, match="not brighter"):
        ccorr.flat_field_correct(raw, flat, dark)
    with pytest.raises(ValueError, match="same shape"):
        ccorr.flat_field_correct(raw, dark[:5], flat)
    with pytest.raises(ValueError):
        ccorr.flat_field_correct(raw, dark, flat, "absorbance")


def test_defective_pixels() -> None:
    """Isolated and clustered defects are found and replaced."""
    rng = np.random.default_rng(1)
    image = rng.normal(1000.0, 10.0, (60, 60))
    truth = np.zeros(image.shape, dtype=bool)
    for row, col in rng.integers(5, 55, (20, 2)):
        truth[row, col] = True
    truth[30:32, 30:32] = True
    damaged = image.copy()
    damaged[truth] = rng.choice([0.0, 16383.0], truth.sum())
    mask = ccorr.defective_pixel_map(damaged)
    assert mask[truth].all()
    assert mask.sum() <= truth.sum() + 3
    corrected = ccorr.correct_defective_pixels(damaged, mask)
    assert np.abs(corrected[truth] - 1000.0).max() < 50.0
    assert np.array_equal(corrected[~mask], damaged[~mask])
    assert np.array_equal(ccorr.correct_defective_pixels(image, ~truth & False), image)
    assert ccorr.robust_sigma(np.array([np.nan])) == 0.0


def test_detector_defects_from_dark_and_flat() -> None:
    """Hot, dead and weak pixels of a simulated detector are found."""
    detector = sim.Detector(rows=120, cols=150, dead_fraction=2e-3, hot_fraction=2e-3)
    result = sim.simulate(sim.plate_scene(), detector=detector, darks=4, flats=4)
    dark = ccorr.mean_frame(result.darks)
    flat = ccorr.mean_frame(result.flats)
    found = ccorr.detector_defect_map(dark, flat)
    truth = sim.defect_mask(detector)
    recall = (found & truth).sum() / truth.sum()
    assert recall == 1.0
    assert found.sum() <= truth.sum() + 0.001 * found.size


def test_attenuation_and_thickness_calibration() -> None:
    """Attenuation, equivalent thickness and polynomial calibration."""
    assert catt.attenuation(np.array([1.0, math.exp(-2)]), 1.0) == pytest.approx(
        [0.0, 2.0]
    )
    assert math.isnan(catt.attenuation(np.array([0.0]), 1.0)[0])
    with pytest.raises(ValueError):
        catt.attenuation(np.ones(2), 0.0)
    assert catt.equivalent_thickness(np.array([0.3]), 0.15) == pytest.approx([2.0])
    with pytest.raises(ValueError):
        catt.equivalent_thickness(np.ones(1), 0.0)
    thicknesses = np.arange(2.0, 22.0, 2.0)
    mu, beta = sim.MATERIALS["aluminum"]
    values = mu * thicknesses / (1 + beta * thicknesses)
    calibration = catt.calibrate_thickness(values, thicknesses, degree=2)
    assert calibration.degree == 2 and calibration.coefficients[-1] == 0.0
    assert calibration.rms_residual < 0.05
    assert calibration.thickness(values) == pytest.approx(thicknesses, abs=0.1)
    free = catt.calibrate_thickness(values, thicknesses, 1, through_origin=False)
    assert free.degree == 1
    with pytest.raises(ValueError, match="needs at least"):
        catt.calibrate_thickness([0.1, 0.2], [1.0, 2.0], degree=2)
    with pytest.raises(ValueError):
        catt.calibrate_thickness([0.1], [1.0, 2.0])


def test_computation_functions() -> None:
    """Defect maps, attenuation and thickness as Sigima functions."""
    data = np.full((50, 60), 1000.0)
    data[:, 40:] = 2000.0  # air
    data[10, 10] = 0.0
    image = create_image("radiograph", data)
    defect = wcorr.DefectivePixelParam()
    mask = computations.defective_pixel_map(image, defect)
    assert mask.data.dtype == np.uint8 and mask.data[10, 10] == 1
    corrected = computations.correct_defective_pixels(image, defect)
    assert corrected.data[10, 10] == pytest.approx(1000.0)
    param = wcorr.AttenuationParam()
    with pytest.raises(ValueError, match="region of interest"):
        computations.attenuation(image, param)
    image.roi = create_image_roi("rectangle", [45, 5, 10, 30])
    result = computations.attenuation(image, param)
    assert result.data[20, 20] == pytest.approx(math.log(2.0))
    param.free_beam, param.i0, param.offset = wcorr.I0_VALUE, 4100.0, 100.0
    result = computations.attenuation(image, param)
    assert result.data[20, 20] == pytest.approx(math.log(4000.0 / 900.0))
    param.free_beam, param.offset = wcorr.I0_MAXIMUM, 0.0
    assert computations.attenuation(image, param).data[20, 50] == pytest.approx(0)
    flat = create_image("flat", np.full((50, 60), 2000.0))
    with_flat = computations.attenuation_with_flat(
        image, flat, wcorr.FlatAttenuationParam()
    )
    assert with_flat.data[20, 20] == pytest.approx(math.log(2.0))
    with pytest.raises(ValueError, match="same size"):
        computations.attenuation_with_flat(
            image, create_image("small", np.ones((5, 5))), wcorr.FlatAttenuationParam()
        )
    thickness_param = wcorr.ThicknessParam()
    thickness_param.mu = math.log(2.0) / 4.0
    thickness = computations.equivalent_thickness(with_flat, thickness_param)
    assert thickness.data[20, 20] == pytest.approx(4.0) and thickness.zunit == "mm"


def _campaign():
    detector = sim.Detector(rows=80, cols=100)
    result = sim.simulate(sim.plate_scene(), detector=detector, darks=2, flats=2)
    frames = []
    for role, items in (
        ("raw", result.radiographs),
        ("dark", result.darks),
        ("flat", result.flats),
    ):
        for frame in items:
            image = create_image(role, frame)
            wcorr.objects.set_frame_role(image, role)
            frames.append(image)
    return frames


def test_offset_gain_recipe() -> None:
    """The recipe corrects the radiographs and reports its indicators."""
    frames = _campaign()
    inputs = wcorr.suggest_frame_bindings(frames)
    assert {slot: len(images) for slot, images in inputs.items()} == {
        "raw": 1,
        "dark": 2,
        "flat": 2,
    }
    assert recipes.check_offset_gain_inputs(inputs, None) == []
    param = recipes.OffsetGainRecipeParam()
    param.output = ccorr.OUTPUT_TRANSMISSION
    outcome = recipes.run_offset_gain(inputs, param, RecipeExecutionContext())
    corrected = outcome.objects[0].value
    assert corrected.title == "Transmission raw" and corrected.zunit == ""
    assert np.nanmedian(corrected.data) < 1.0
    assert outcome.results[0].anchor_id == "corrected-1"
    assert outcome.results[0].value.headers[0] == "Radiograph"
    with pytest.raises(RecipeValidationError):
        recipes.run_offset_gain(inputs, None, RecipeExecutionContext())
    bad = dict(inputs, dark=(create_image("small", np.zeros((5, 5))),))
    assert recipes.check_offset_gain_inputs(bad, param)[0].code == (
        "inconsistent-image-size"
    )


def test_step_wedge_recipe_checks() -> None:
    """The step wedge recipe checks its ROIs and thicknesses."""
    wedge = create_image("wedge", np.full((40, 40), 1000.0))
    inputs = {"wedge": (wedge,)}
    param = recipes.StepWedgeParam()
    param.thicknesses = "2, 4"
    codes = [d.code for d in recipes.check_step_wedge_inputs(inputs, param)]
    assert codes == ["step-count"]
    param.thicknesses = "2, x"
    codes = [d.code for d in recipes.check_step_wedge_inputs(inputs, param)]
    assert codes == ["invalid-steps"]
    assert recipes.parse_thicknesses("1; 2 3,4") == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        recipes.parse_thicknesses("-1")
    dark_only = {"wedge": (wedge,), "dark": (wedge,)}
    param.thicknesses = ""
    codes = [d.code for d in recipes.check_step_wedge_inputs(dark_only, param)]
    assert "dark-flat-pair" in codes
    with pytest.raises(RecipeValidationError):
        recipes.run_step_wedge(inputs, param, RecipeExecutionContext())


def test_step_wedge_without_frames() -> None:
    """Without dark and flat frames, a free-beam ROI or a log fit is used."""
    mu = 0.05
    data = np.full((40, 100), 2000.0)
    thicknesses = [2.0, 6.0, 10.0, 14.0]
    for index, t in enumerate(thicknesses):
        data[10:30, 10 + 20 * index : 25 + 20 * index] = 2000.0 * math.exp(-mu * t)
    roi = create_image_roi("rectangle", [92, 5, 6, 30])
    for index in range(4):
        roi.add_roi(
            create_image_roi("rectangle", [12 + 20 * index, 12, 10, 16]).single_rois[0]
        )
    wedge = create_image("wedge", data)
    wedge.roi = roi
    param = recipes.StepWedgeParam()
    param.thicknesses, param.free_beam_roi, param.degree = "2, 6, 10, 14", 1, 1
    inputs = {"wedge": (wedge,)}
    assert recipes.check_step_wedge_inputs(inputs, param) == []
    outcome = recipes.run_step_wedge(inputs, param, RecipeExecutionContext())
    thickness = outcome.objects[1].value
    assert thickness.data[20, 15] == pytest.approx(2.0, abs=1e-6)
    assert thickness.data[20, 75] == pytest.approx(14.0, abs=1e-6)
    param.free_beam_roi = 0
    wedge.roi = create_image_roi("rectangle", [12, 12, 10, 16])
    for index in range(1, 4):
        wedge.roi.add_roi(
            create_image_roi("rectangle", [12 + 20 * index, 12, 10, 16]).single_rois[0]
        )
    outcome = recipes.run_step_wedge(inputs, param, RecipeExecutionContext())
    assert outcome.objects[1].value.data[20, 55] == pytest.approx(10.0, abs=1e-6)
    curve = outcome.objects[0].value
    assert curve.yunit == "mm" and len(curve.x) == 200
