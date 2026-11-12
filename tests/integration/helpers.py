"""Helpers to run DataLab Desktop with the radiography plugin."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from datalab.config import Conf
from datalab.env import execenv
from datalab.gui.main import DLMainWindow
from datalab.objectmodel import get_uuid
from datalab.plugins import PluginRegistry
from datalab.tests import datalab_test_app_context
from qtpy import QtWidgets as QW

from datalab_radiography import PLUGIN_ID
from datalab_radiography.adapters.desktop import XRayRadiographyNDTPlugin


@contextmanager
def radiography_app(
    enabled: bool = True,
) -> Generator[DLMainWindow, None, None]:
    """Run DataLab with the plugin loaded from its entry point.

    Args:
        enabled: if False, the plugin is disabled
    """
    enabled_list = None if enabled else []
    with (
        Conf.plugins_enabled.context(True),
        Conf.plugins_enabled_list.context(enabled_list),
        execenv.context(unattended=True),
        datalab_test_app_context(console=False, exec_loop=False) as window,
    ):
        window.reload_plugins()
        yield window


def radiography_plugins() -> list[XRayRadiographyNDTPlugin]:
    """Return the registered instances of the plugin."""
    return [
        plugin
        for plugin in PluginRegistry.get_plugins()
        if plugin.plugin_id == PLUGIN_ID
    ]


def the_plugin() -> XRayRadiographyNDTPlugin:
    """Return the only registered instance of the plugin."""
    plugins = radiography_plugins()
    assert len(plugins) == 1
    return plugins[0]


def plot_item(window: DLMainWindow, obj):
    """Return the plot item of an image, after processing pending events."""
    QW.QApplication.processEvents()
    return window.imagepanel.plothandler.get(get_uuid(obj))
