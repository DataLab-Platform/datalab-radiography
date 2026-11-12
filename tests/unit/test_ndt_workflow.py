"""Tests of the image quality and NDT workflows (tables, regions, recipes)."""

from __future__ import annotations

import numpy as np
import pytest
from datalab.plugins.recipes import RecipeExecutionContext, RecipeValidationError
from sigima.objects import create_image, create_image_roi

from datalab_radiography import demo
from datalab_radiography.core import magnification as mag
from datalab_radiography.workflow import computations, objects
from datalab_radiography.workflow import measurements as wmeas
from datalab_radiography.workflow import ndt_recipes as recipes
from datalab_radiography.workflow import quality as wquality


def noisy_image() -> object:
    """Return an image of known mean and noise, with two ROIs."""
    rng = np.random.default_rng(0)
    data = 1000.0 + rng.normal(0.0, 10.0, (100, 120))
    data[20:40, 20:40] += 50.0
    image = create_image("radiograph", data)
    image.roi = create_image_roi("rectangle", [[22, 22, 15, 15], [60, 50, 50, 40]])
    return image


def test_snr_and_cnr_tables() -> None:
    """SNR and CNR tables give one row per region."""
    image = noisy_image()
    table = computations.snr(image, wquality.SNRParam.create(srb_um=177.2))
    assert table.headers[-1].startswith("SNRN")
    background = table.data[1]
    assert background[1] == pytest.approx(1000.0, abs=1.0)
    assert background[2] == pytest.approx(10.0, rel=0.1)
    assert background[3] == pytest.approx(100.0, rel=0.1)
    assert background[4] == pytest.approx(50.0, rel=0.1)
    assert list(table.roi_indices) == [0, 1]
    whole = create_image("plain", np.full((10, 10), 5.0) + np.eye(10))
    assert len(computations.snr(whole, wquality.SNRParam()).data) == 1
    cnr = computations.cnr(image, wquality.CNRParam())
    assert len(cnr.data) == 1 and list(cnr.roi_indices) == [0]
    assert cnr.data[0][4] == pytest.approx(50.0, abs=2.0)
    assert cnr.data[0][5] == pytest.approx(5.0, rel=0.15)
    reverse = computations.cnr(image, wquality.CNRParam.create(background_roi=1))
    assert reverse.data[0][0] != cnr.data[0][0]
    with pytest.raises(wquality.RegionError, match="no region"):
        computations.cnr(image, wquality.CNRParam.create(background_roi=3))
    with pytest.raises(wquality.RegionError, match="at least 2"):
        computations.cnr(whole, wquality.CNRParam())


def test_measurement_regions_and_profiles() -> None:
    """Rectangles come from the first rectangular ROI or from coordinates."""
    image = noisy_image()
    rows, cols = wquality.box_slices(image, wquality.RegionParam())
    assert (rows.start, rows.stop, cols.start, cols.stop) == (22, 38, 22, 38)
    param = wquality.RegionParam.create(region=wquality.REGION_BOX)
    param.x0, param.y0, param.x1, param.y1 = 10.0, 5.0, 60.0, 15.0
    rows, cols = wquality.box_slices(image, param)
    assert (rows.start, rows.stop, cols.start, cols.stop) == (5, 16, 10, 61)
    profile, coords, direction = wquality.box_profile(image, param)
    assert direction == wquality.DIRECTION_X and profile.size == coords.size == 51
    _profile, coords, direction = wquality.box_profile(image, param, "y")
    assert direction == wquality.DIRECTION_Y and coords.size == 11
    mask = wquality.region_mask(image, param)
    assert mask.sum() == 11 * 51
    assert wquality.region_mask(image, wquality.RegionParam()).sum() > 16 * 16
    param.x1 = 11.0
    with pytest.raises(wquality.RegionError, match="too small"):
        wquality.box_slices(image, param)
    plain = create_image("plain", np.zeros((20, 20)))
    with pytest.raises(wquality.RegionError, match="rectangular"):
        wquality.box_slices(plain, wquality.RegionParam())
    assert wquality.region_mask(plain, wquality.RegionParam()).all()


def test_named_planes() -> None:
    """Planes are found by name, active magnification, or detector."""
    image = create_image("radiograph", np.zeros((20, 30)))
    with pytest.raises(mag.MagnificationError, match="no magnification"):
        wmeas.plane_named(image)
    with pytest.raises(mag.MagnificationError, match="pitch"):
        wmeas.plane_named(image, "detector")
    objects.set_detector_pitch(image, 0.2)
    detector = wmeas.plane_named(image, "Detector")
    assert detector.scale == pytest.approx((0.2, 0.2)) and detector.factor == 1.0
    objects.set_magnifications(
        image,
        mag.MagnificationSet(
            (
                mag.Magnification("Weld", (0.16, 0.16)),
                mag.Magnification("Ball", (0.19, 0.19), factor=1.05),
            )
        ),
    )
    weld = wmeas.plane_named(image)
    assert weld.title == "Weld" and weld.factor == pytest.approx(1.25)
    assert wmeas.plane_named(image, "Ball").factor == 1.05
    with pytest.raises(mag.MagnificationError, match="Weld, Ball"):
        wmeas.plane_named(image, "Pipe")
    param = wquality.IndicationParam.create(background_size=1.6, plane="Weld")
    assert wquality.background_pixels(image, param) == pytest.approx((10.0, 10.0))
    param.background_unit = wquality.PIXELS
    assert wquality.background_pixels(image, param) == (1.6, 1.6)


def run_example_recipe(example_id: str, recipe, param_class, slot: str, **values):
    """Run a recipe on the first image of an example with its parameter values."""
    data = demo.materialize_generated_example(example_id)
    param = param_class()
    for name, value in {**data.values_for(recipe.recipe_id), **values}.items():
        setattr(param, name, value)
    image = data.objects[0]
    return image, recipe.run({slot: (image,)}, param, RecipeExecutionContext())


def test_duplex_recipe() -> None:
    """The duplex recipe adds an annotated profile and two tables."""
    _image, outcome = run_example_recipe(
        "simulated-weld", recipes.DUPLEX_RECIPE, wquality.DuplexParam, "image"
    )
    profile = outcome.objects[0].value
    assert profile.xunit == "mm"
    annotations = profile.get_graphical_annotations()
    assert annotations[0].title == "D1"
    summary, elements = (result.value for result in outcome.results)
    assert summary.data[0][0].startswith("D")
    assert summary.data[0][2] == pytest.approx(1.25 * summary.data[0][1])
    assert len(elements.data) == len(annotations)
    assert not outcome.diagnostics
    with pytest.raises(RecipeValidationError, match="No magnification"):
        run_example_recipe(
            "simulated-weld",
            recipes.DUPLEX_RECIPE,
            wquality.DuplexParam,
            "image",
            plane="Pipe",
        )
    with pytest.raises(RecipeValidationError, match="parameters"):
        recipes.DUPLEX_RECIPE.run({}, None, RecipeExecutionContext())


def test_wall_thickness_recipe() -> None:
    """The wall recipe adds the profile, the model and a table of both methods."""
    _image, outcome = run_example_recipe(
        "simulated-pipe",
        recipes.WALL_THICKNESS_RECIPE,
        wquality.WallThicknessParam,
        "image",
        unit="cm",
    )
    profile, model = (output.value for output in outcome.objects)
    assert profile.xunit == "cm" and model.y.shape == profile.y.shape
    assert [item.title for item in profile.get_graphical_annotations()] == [
        "Wall 1",
        "Wall 2",
    ]
    table = outcome.results[0].value
    assert table.data[0][1] == pytest.approx(0.35, rel=0.02)
    assert table.headers[1] == "Wall 1 (cm)"
    with pytest.raises(RecipeValidationError, match="whole pipe"):
        run_example_recipe(
            "simulated-pipe",
            recipes.WALL_THICKNESS_RECIPE,
            wquality.WallThicknessParam,
            "image",
            x0=0.0,
            x1=250.0,
        )


def test_indication_recipe() -> None:
    """The indication recipe annotates a copy of each radiograph."""
    image, outcome = run_example_recipe(
        "simulated-weld",
        recipes.INDICATION_RECIPE,
        wquality.IndicationParam,
        "images",
        label_image=True,
    )
    annotated, labels = (output.value for output in outcome.objects)
    table = outcome.results[0].value
    assert outcome.results[0].anchor_id == outcome.objects[0].id
    assert len(table.data) == len(annotated.get_graphical_annotations()) == 4
    assert np.array_equal(annotated.data, image.data)
    assert labels.data.max() == 4
    assert objects.get_magnifications(annotated).active == "Weld"
    _image, empty = run_example_recipe(
        "simulated-weld",
        recipes.INDICATION_RECIPE,
        wquality.IndicationParam,
        "images",
        detection=1000.0,
        sizing=500.0,
    )
    assert empty.diagnostics[0].code == "no-indication"
    with pytest.raises(RecipeValidationError, match="sizing"):
        run_example_recipe(
            "simulated-weld",
            recipes.INDICATION_RECIPE,
            wquality.IndicationParam,
            "images",
            sizing=8.0,
        )
    dark = create_image("dark", np.zeros((10, 10)))
    objects.set_frame_role(dark, "dark")
    bindings = recipes.INDICATION_RECIPE.suggest_bindings([image, dark])
    assert bindings == {"images": (image,)}
