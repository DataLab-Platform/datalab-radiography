"""Contract tests for the DataLab Desktop plugin class."""

from __future__ import annotations

from pathlib import Path

from datalab.plugins import PluginCapability

import datalab_radiography
from datalab_radiography import PLUGIN_ID, __version__
from datalab_radiography.adapters import desktop

ICONS = Path(datalab_radiography.__file__).parent / "icons"


def test_plugin_descriptor() -> None:
    """The entry point exposes stable plugin metadata."""
    plugin_class = desktop.XRayRadiographyNDTPlugin
    assert plugin_class.get_plugin_id() == PLUGIN_ID == "org.datalab.radiography"
    assert plugin_class.PLUGIN_INFO.version == __version__
    assert plugin_class.PLUGIN_INFO.capabilities == frozenset(
        {
            PluginCapability.APPLICATION,
            PluginCapability.PROCESSING,
            PluginCapability.VISUALIZATION,
        }
    )
    assert plugin_class.PLUGIN_INFO.documentation_url.startswith("https://")


def test_plugin_icons_exist() -> None:
    """Every icon referenced as a package resource is packaged."""
    package, relative = desktop.PLUGIN_ICON.split(":", 1)
    assert package == "datalab_radiography"
    assert (ICONS.parent / relative).is_file()
    for icon in ICONS.glob("*.svg"):
        assert icon.read_text(encoding="utf-8").lstrip().startswith("<svg")
