"""DataLab Desktop plugin adapter."""

from __future__ import annotations

import os
import os.path as osp

from datalab.config import get_config_path
from datalab.plugins import PluginBase, PluginCapability, PluginInfo

from .. import PLUGIN_DESCRIPTION, PLUGIN_ID, PLUGIN_NAME, __version__
from ..workflow import RECIPES as WORKFLOW_RECIPES
from ..workflow.parameters import SettingsParam
from ..workflow.settings import SettingsStore
from .desktop_display import DisplayController

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
    #: Environment variable overriding the preference file (used by the tests)
    SETTINGS_ENV = "DATALAB_RADIOGRAPHY_SETTINGS"

    def __init__(self) -> None:
        super().__init__()
        path = os.environ.get(self.SETTINGS_ENV)
        if not path:
            path = osp.join(get_config_path("plugin_settings"), f"{PLUGIN_ID}.json")
        self.settings_store = SettingsStore(path)
        self.display = DisplayController(self)

    # -- Registration -----------------------------------------------------

    def register_computations(self) -> None:
        """Watch new images once the panels exist."""
        self.display.connect()

    def unregister_hooks(self) -> None:
        """Stop watching the images."""
        self.display.disconnect()

    # -- Actions ----------------------------------------------------------

    def edit_settings(self) -> bool:
        """Edit the plugin preferences."""
        param = SettingsParam.from_settings(self.settings_store.settings)
        param.set_title(f"{PLUGIN_NAME} preferences")
        if not param.edit(parent=self.main):
            return False
        self.settings_store.update(**param.to_values())
        self.film_convention_action.setChecked(
            self.settings_store.settings.film_convention
        )
        return True

    def create_actions(self) -> None:
        """Create the plugin menu of the image panel."""
        handler = self.imagepanel.acthandler
        display = self.display
        with handler.new_menu(PLUGIN_NAME.replace("&", "&&")):
            with handler.new_menu("Display"):
                self.apply_display_action = handler.new_action(
                    "Apply radiographic display",
                    triggered=display.apply_profile_to_selection,
                    tip="Gray scale, inversion and automatic window",
                    select_condition="at_least_one",
                )
                self.film_convention_action = handler.new_action(
                    "Film convention (dense matter bright)",
                    toggled=display.set_film_convention,
                    tip=(
                        "Invert the gray scale of new and selected images "
                        "(DICOM images follow their attributes)"
                    ),
                    select_condition="always",
                )
                self.film_convention_action.blockSignals(True)
                self.film_convention_action.setChecked(
                    self.settings_store.settings.film_convention
                )
                self.film_convention_action.blockSignals(False)
                self.auto_window_action = handler.new_action(
                    "Automatic window",
                    triggered=display.auto_window,
                    tip="Window between two percentiles of the image values",
                    select_condition="at_least_one",
                    separator=True,
                )
                self.roi_window_action = handler.new_action(
                    "Window from regions of interest",
                    triggered=lambda: display.auto_window(use_roi=True),
                    tip="Window between two percentiles of the values in the ROIs",
                    select_condition="with_roi",
                )
                self.edit_window_action = handler.new_action(
                    "Window...",
                    triggered=display.edit_window,
                    tip="Percentiles, DICOM attributes, or level and width",
                    select_condition="at_least_one",
                )
            self.settings_action = handler.new_action(
                "Preferences...",
                triggered=self.edit_settings,
                select_condition="always",
                separator=True,
            )
