"""Image quality and NDT measurements in DataLab Desktop."""

from __future__ import annotations

import os

from datalab.adapters_metadata import TableAdapter
from datalab.gui.plugins.recipe_runner import RecipeRunner
from datalab.objectmodel import get_uuid
from sigima.objects import create_image_roi

from datalab_radiography.workflow import features as wfeatures
from datalab_radiography.workflow import ndt_recipes as recipes
from datalab_radiography.workflow import quality as wquality

from .helpers import radiography_app, submenu_texts, the_plugin


def test_quality_and_ndt_menus() -> None:
    """Image quality and NDT menus hold the features and recipes."""
    with radiography_app() as window:
        quality = [
            text.removesuffix("...")
            for text in submenu_texts(window, wfeatures.GROUP_IMAGE_QUALITY)
        ]
        assert quality == [
            "Signal-to-noise ratio",
            "Contrast-to-noise ratio",
            recipes.DUPLEX_RECIPE.title,
        ]
        ndt = [
            text.removesuffix("...")
            for text in submenu_texts(window, wfeatures.GROUP_NDT)
        ]
        assert ndt == [
            recipes.WALL_THICKNESS_RECIPE.title,
            recipes.INDICATION_RECIPE.title,
        ]


def test_snr_and_cnr_features_in_desktop() -> None:
    """SNR and CNR results are attached to the radiograph."""
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("simulated-weld")
        weld = window.imagepanel.objmodel.get_all_objects()[0]
        weld.roi = create_image_roi(
            "rectangle", [[275.0, 205.0, 12.0, 12.0], [100.0, 170.0, 60.0, 60.0]]
        )
        window.imagepanel.objview.select_objects([get_uuid(weld)])
        processor = window.imagepanel.processor
        processor.run_feature(f"{plugin.plugin_id}:snr", wquality.SNRParam())
        processor.run_feature(f"{plugin.plugin_id}:cnr", wquality.CNRParam())
        tables = {table.title: table for table in TableAdapter.iterate_from_obj(weld)}
        assert {"SNR", "CNR"} <= set(tables)
        assert tables["CNR"].result.data[0][5] > 3.0


def test_ndt_recipes_in_desktop(tmp_path) -> None:
    """Recipes add profiles, annotated images and tables that survive a reload."""
    filename = os.fspath(tmp_path / "ndt.h5")
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("simulated-weld")
        images = window.imagepanel.objmodel.get_all_objects()
        runner = RecipeRunner(window)
        for recipe, param_class, slot in (
            (recipes.DUPLEX_RECIPE, wquality.DuplexParam, "image"),
            (recipes.INDICATION_RECIPE, wquality.IndicationParam, "images"),
        ):
            param = param_class()
            values = plugin.example_parameter_values(recipe.recipe_id, images)
            for name, value in values.items():
                setattr(param, name, value)
            inputs = recipe.suggest_bindings(images)
            assert inputs[slot] == (images[0],)
            runner.run(recipe, inputs, param)
        profile = window.signalpanel.objmodel.get_all_objects()[-1]
        assert len(profile.get_graphical_annotations()) >= 9
        assert len(list(TableAdapter.iterate_from_obj(profile))) == 2
        annotated = window.imagepanel.objmodel.get_all_objects()[-1]
        assert len(annotated.get_graphical_annotations()) == 4
        assert len(list(TableAdapter.iterate_from_obj(annotated))) == 1
        window.save_h5_workspace(filename)
        window.load_h5_workspace([filename], reset_all=True)
        loaded = window.find_object_by_uuid(get_uuid(profile))
        assert len(loaded.get_graphical_annotations()) == len(
            profile.get_graphical_annotations()
        )
        loaded = window.find_object_by_uuid(get_uuid(annotated))
        assert len(loaded.get_graphical_annotations()) == 4
