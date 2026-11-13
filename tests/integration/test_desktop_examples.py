"""Examples and simulator in DataLab Desktop."""

from __future__ import annotations

import numpy as np
import pytest
from datalab.gui.plugins.recipe_runner import RecipeRunner
from datalab.objectmodel import get_uuid

from datalab_radiography import demo
from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import correction_recipes as recipes
from datalab_radiography.workflow import corrections as wcorr
from datalab_radiography.workflow import display as wdisplay
from datalab_radiography.workflow import objects

from .helpers import plot_item, radiography_app, the_plugin


def test_examples_open_in_desktop() -> None:
    """Every example opens with its metadata and the radiographic display."""
    with radiography_app() as window:
        plugin = the_plugin()
        assert {tile.id for tile in plugin.get_welcome_tiles()} == {
            "application",
            "weld-example",
        }
        assert set(plugin.example_actions) == {example.id for example in demo.EXAMPLES}
        for example in plugin.get_examples():
            plugin.open_example(example.id, reset_all=True)
            images = window.imagepanel.objmodel.get_all_objects()
            assert images, example.id
            first = images[0]
            assert objects.get_magnifications(first).active is not None
            window.imagepanel.objview.select_objects([get_uuid(first)])
            assert plot_item(window, first).param.colormap == "gray"
            assert wdisplay.has_profile(first)


def test_real_examples_open_with_their_metadata() -> None:
    """Real-data examples carry their pitch, magnification and attribution."""
    expected = {
        "walnut": (3, 0.2992, 3.0152),
        "circuit-board": (1, 0.4, 6.952),
        "knee-phantom": (1, 0.556, 1.7073),
    }
    with radiography_app() as window:
        plugin = the_plugin()
        for example in demo.REAL_EXAMPLES:
            count, pitch, factor = expected[example.id]
            plugin.open_example(example.id, reset_all=True)
            images = window.imagepanel.objmodel.get_all_objects()
            assert len(images) == count, example.id
            first = images[0]
            assert objects.get_frame_role(first) == "raw"
            assert objects.get_detector_pitch(first) == pytest.approx((pitch, pitch))
            active = objects.get_magnifications(first).active_item
            assert active.factor == pytest.approx(factor, rel=1e-3), example.id
            attribution = md.loads(first.metadata[md.ATTRIBUTION_KEY])
            assert attribution["license"] == "CC BY 4.0"
            assert attribution["doi"].startswith("10.5281/zenodo.")
            window.imagepanel.objview.select_objects([get_uuid(first)])
            assert plot_item(window, first).param.colormap == "gray"


def test_walnut_example_correction() -> None:
    """The walnut example is corrected to a transmission close to 1 in the air."""
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("walnut", reset_all=True)
        images = window.imagepanel.objmodel.get_all_objects()
        inputs = wcorr.suggest_frame_bindings(images)
        assert [len(inputs[slot]) for slot in ("raw", "dark", "flat")] == [1, 1, 1]
        param = recipes.OffsetGainRecipeParam()
        param.output = "transmission"
        outcome = RecipeRunner(window).run(recipes.OFFSET_GAIN_RECIPE, inputs, param)
        transmission = outcome.objects[0].value.data
        assert np.nanpercentile(transmission, 99) == pytest.approx(1.0, abs=0.03)
        assert np.nanpercentile(transmission, 5) < 0.5


def test_simulator_tool_is_declared() -> None:
    """The simulator is a tool of the image panel, with an instrument."""
    with radiography_app():
        plugin = the_plugin()
        tool = plugin.get_tool("xray-simulator")
        instrument = plugin.get_instrument(tool.id)
        instrument.settings.rows, instrument.settings.cols = 64, 64
        assert instrument.preview().objects[0].data.shape == (64, 64)
