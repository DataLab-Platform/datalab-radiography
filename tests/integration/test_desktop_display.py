"""Radiographic display of new images in DataLab Desktop."""

from __future__ import annotations

import os

import numpy as np
import pytest
import sigima.proc.image as sipi
from datalab.config import Conf
from datalab.gui.actionhandler import ActionCategory
from datalab.objectmodel import get_uuid
from sigima.objects import create_image

from datalab_radiography import PLUGIN_NAME
from datalab_radiography.core import display
from datalab_radiography.workflow import display as wdisplay

from .helpers import plot_item, radiography_app, radiography_plugins, the_plugin


def _radiograph(title: str = "radiograph") -> object:
    """Return a 12-bit radiograph with a clipped free-beam region."""
    rng = np.random.default_rng(3)
    data = rng.normal(1500.0, 40.0, (120, 160))
    data[:, 140:] = 4095.0
    data[40:80, 30:70] -= 700.0
    return create_image(title, np.clip(data, 1, 4095).astype(np.uint16))


def _lut(item) -> tuple[float, float]:
    """Return the LUT range of a plot item."""
    low, high = item.get_lut_range()
    return float(low), float(high)


def test_new_image_gets_radiographic_display() -> None:
    """A new image is displayed in gray with an automatic window."""
    default_colormap = Conf.get_sigima_defaults("ima")["colormap"]
    with radiography_app() as window:
        image = _radiograph()
        window.imagepanel.add_object(image)
        item = plot_item(window, image)
        assert item.param.colormap == "gray"
        assert item.param.invert_colormap is False
        expected = display.percentile_window(image.data)
        assert _lut(item) == pytest.approx((expected.low, expected.high))
        assert wdisplay.has_profile(image)
    # The preferences of the user are unchanged
    assert Conf.get_sigima_defaults("ima")["colormap"] == default_colormap


def test_film_convention_and_processing_results() -> None:
    """Results inherit the profile and get their own automatic window."""
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.settings_store.update(film_convention=True)
        image = _radiograph()
        window.imagepanel.add_object(image)
        assert plot_item(window, image).param.invert_colormap is True
        window.imagepanel.processor.compute_1_to_1(sipi.normalize, title="Normalize")
        result = window.imagepanel.objmodel.get_all_objects()[-1]
        assert result is not image
        item = plot_item(window, result)
        assert item.param.colormap == "gray"
        assert item.param.invert_colormap is True
        low, high = _lut(item)
        assert 0.0 <= low < high <= 1.0
        assert result.zscalemin == pytest.approx(low)


def test_explicit_settings_and_workspaces_are_kept(tmp_path) -> None:
    """Explicit colormaps and saved display settings are not overridden."""
    filename = os.fspath(tmp_path / "workspace.h5")
    with radiography_app() as window:
        jet = _radiograph("jet")
        jet.set_metadata_option("colormap", "jet")
        window.imagepanel.add_object(jet)
        assert plot_item(window, jet).param.colormap == "jet"
        gray = _radiograph("gray")
        window.imagepanel.add_object(gray)
        gray_window = (gray.zscalemin, gray.zscalemax)
        window.save_h5_workspace(filename)
    with radiography_app(enabled=False) as window:
        viridis = _radiograph("viridis")
        window.imagepanel.add_object(viridis)
        assert plot_item(window, viridis).param.colormap == "viridis"
        window.save_h5_workspace(filename.replace(".h5", "-viridis.h5"))
    with radiography_app() as window:
        window.load_h5_workspace([filename], reset_all=True)
        loaded = {obj.title: obj for obj in window.imagepanel.objmodel}
        assert loaded["jet"].get_metadata_option("colormap") == "jet"
        assert (loaded["gray"].zscalemin, loaded["gray"].zscalemax) == pytest.approx(
            gray_window
        )
        window.load_h5_workspace(
            [filename.replace(".h5", "-viridis.h5")], reset_all=True
        )
        viridis = window.imagepanel.objmodel.get_all_objects()[0]
        window.imagepanel.objview.select_objects([get_uuid(viridis)])
        assert plot_item(window, viridis).param.colormap == "viridis"


def test_disabled_plugin_and_automatic_display_preference() -> None:
    """Without the plugin, or with the preference off, DataLab is unchanged."""
    with radiography_app(enabled=False) as window:
        assert not radiography_plugins()
        image = _radiograph()
        window.imagepanel.add_object(image)
        assert plot_item(window, image).param.colormap != "gray"
    with radiography_app() as window:
        the_plugin().settings_store.update(auto_display=False)
        image = _radiograph()
        window.imagepanel.add_object(image)
        assert plot_item(window, image).param.colormap != "gray"
        assert not wdisplay.has_profile(image)


def test_reload_keeps_one_plugin_menu_and_watcher() -> None:
    """Hot reload replaces the plugin and its image watcher."""
    with radiography_app() as window:
        before = the_plugin()
        assert before.display._connected
        sources = before.__class__.__plugin_discovery_sources__
        assert any("entry point 'datalab_radiography'" in s for s in sources)
        window.reload_plugins()
        after = the_plugin()
        assert after is not before
        assert not before.display._connected and after.display._connected
        menus = [
            action
            for action in window.imagepanel.get_category_actions(ActionCategory.PLUGINS)
            if hasattr(action, "title")
            and action.title() == PLUGIN_NAME.replace("&", "&&")
        ]
        assert len(menus) == 1
        image = _radiograph()
        window.imagepanel.add_object(image)
        assert plot_item(window, image).param.colormap == "gray"


def test_display_actions_on_selected_images() -> None:
    """Menu actions change the display of the selected images."""
    with radiography_app() as window:
        plugin = the_plugin()
        jet = _radiograph()
        jet.set_metadata_option("colormap", "jet")
        window.imagepanel.add_object(jet)
        window.imagepanel.objview.select_objects([get_uuid(jet)])
        assert plugin.display.apply_profile_to_selection() == [jet]
        assert plot_item(window, jet).param.colormap == "gray"
        plugin.film_convention_action.setChecked(True)
        assert plugin.settings_store.settings.film_convention is True
        assert plot_item(window, jet).param.invert_colormap is True
        jet.zscalemin, jet.zscalemax = 0.0, 1.0
        plugin.display.auto_window()
        assert jet.zscalemax > 1.0
        assert plugin.roi_window_action is not None
        assert plugin.settings_action.isEnabled()
