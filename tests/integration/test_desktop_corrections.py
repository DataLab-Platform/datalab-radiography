"""Recipes and correction features in DataLab Desktop."""

from __future__ import annotations

import os

import numpy as np
import pytest
from datalab.adapters_metadata import TableAdapter
from datalab.gui.plugins.recipe_runner import RecipeRunner
from datalab.objectmodel import get_uuid
from datalab.plugins.recipes import RECIPE_RUN_RECORD_OPTION

from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import correction_recipes as recipes
from datalab_radiography.workflow import corrections as wcorr
from datalab_radiography.workflow.recipes import RECIPES

from .helpers import radiography_app, the_plugin


def test_offset_gain_recipe_in_desktop(tmp_path) -> None:
    """The correction recipe adds corrected images with their provenance."""
    filename = os.fspath(tmp_path / "corrections.h5")
    with radiography_app() as window:
        plugin = the_plugin()
        assert set(plugin.recipe_actions) == {recipe.recipe_id for recipe in RECIPES}
        plugin.open_example("simulated-campaign")
        images = window.imagepanel.objmodel.get_all_objects()
        inputs = wcorr.suggest_frame_bindings(images)
        outcome = RecipeRunner(window).run(
            recipes.OFFSET_GAIN_RECIPE, inputs, recipes.OffsetGainRecipeParam()
        )
        corrected = outcome.objects[0].value
        assert window.find_object_by_uuid(get_uuid(corrected)) is corrected
        assert corrected.metadata.get(RECIPE_RUN_RECORD_OPTION) or (
            corrected.get_metadata_option(RECIPE_RUN_RECORD_OPTION)
        )
        assert corrected.metadata[md.OUTPUT_ROLE_KEY] == "corrected"
        assert len(list(TableAdapter.iterate_from_obj(corrected))) == 1
        window.save_h5_workspace(filename)
        window.load_h5_workspace([filename], reset_all=True)
        loaded = window.find_object_by_uuid(get_uuid(corrected))
        assert loaded is not None
        assert len(list(TableAdapter.iterate_from_obj(loaded))) == 1


def test_step_wedge_recipe_and_features_in_desktop() -> None:
    """The step wedge example calibrates thickness in DataLab."""
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("simulated-step-wedge")
        images = window.imagepanel.objmodel.get_all_objects()
        inputs = recipes.suggest_step_wedge_bindings(images)
        param = recipes.StepWedgeParam()
        values = plugin.example_parameter_values(
            recipes.STEP_WEDGE_RECIPE.recipe_id, images
        )
        for name, value in values.items():
            setattr(param, name, value)
        outcome = RecipeRunner(window).run(recipes.STEP_WEDGE_RECIPE, inputs, param)
        curve = outcome.objects[0].value
        assert window.signalpanel.objmodel.get_all_objects()[-1] is curve
        thickness = outcome.objects[1].value
        assert thickness.zunit == "mm"
        assert np.nanmax(thickness.data) == pytest.approx(20.0, abs=1.5)
        wedge = inputs["wedge"][0]
        window.imagepanel.objview.select_objects([get_uuid(wedge)])
        before = {get_uuid(obj) for obj in window.imagepanel.objmodel}
        param = wcorr.AttenuationParam()
        param.free_beam = wcorr.I0_MAXIMUM
        window.imagepanel.processor.run_feature(
            "org.datalab.radiography:attenuation", param
        )
        new = [obj for obj in window.imagepanel.objmodel if get_uuid(obj) not in before]
        assert len(new) == 1 and new[0].zunit == ""
        assert np.nanmax(new[0].data) > 0.5
