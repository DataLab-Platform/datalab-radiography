"""Examples and simulator in DataLab Desktop."""

from __future__ import annotations

from datalab.objectmodel import get_uuid

from datalab_radiography import demo
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
        assert set(plugin.example_actions) == set(demo.GENERATED_EXAMPLES)
        for example in plugin.get_examples():
            plugin.open_example(example.id, reset_all=True)
            images = window.imagepanel.objmodel.get_all_objects()
            assert images, example.id
            first = images[0]
            assert objects.get_magnifications(first).active is not None
            window.imagepanel.objview.select_objects([get_uuid(first)])
            assert plot_item(window, first).param.colormap == "gray"
            assert wdisplay.has_profile(first)


def test_simulator_tool_is_declared() -> None:
    """The simulator is a tool of the image panel, with an instrument."""
    with radiography_app():
        plugin = the_plugin()
        tool = plugin.get_tool("xray-simulator")
        instrument = plugin.get_instrument(tool.id)
        instrument.settings.rows, instrument.settings.cols = 64, 64
        assert instrument.preview().objects[0].data.shape == (64, 64)
