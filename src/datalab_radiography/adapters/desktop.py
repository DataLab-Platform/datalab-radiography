"""DataLab Desktop plugin adapter."""

from __future__ import annotations

from datalab.plugins import PluginBase, PluginCapability, PluginInfo

from .. import PLUGIN_DESCRIPTION, PLUGIN_ID, PLUGIN_NAME, __version__
from ..workflow import RECIPES as WORKFLOW_RECIPES

PLUGIN_ICON = "datalab_radiography:icons/radiography.svg"
DOCUMENTATION_URL = "https://github.com/DataLab-Platform/datalab-radiography"


class XRayRadiographyNDTPlugin(PluginBase):
    """Expose X-ray radiography and NDT features to DataLab Desktop."""

    PLUGIN_INFO = PluginInfo(
        id=PLUGIN_ID,
        name=PLUGIN_NAME,
        version=__version__,
        description=PLUGIN_DESCRIPTION,
        icon=PLUGIN_ICON,
        capabilities=(
            PluginCapability.APPLICATION,
            PluginCapability.PROCESSING,
            PluginCapability.VISUALIZATION,
        ),
        documentation_url=DOCUMENTATION_URL,
    )
    RECIPES = WORKFLOW_RECIPES

    def create_actions(self) -> None:
        """Create the plugin menu of the image panel."""
