"""DataLab Desktop plugin adapter."""

from __future__ import annotations

import logging
import os
import os.path as osp

from datalab.config import get_config_path
from datalab.env import execenv
from datalab.plugins import PluginBase, PluginCapability, PluginInfo
from qtpy import QtWidgets as QW

from .. import PLUGIN_DESCRIPTION, PLUGIN_ID, PLUGIN_NAME, __version__
from ..workflow import RECIPES as WORKFLOW_RECIPES
from ..workflow import features as wfeatures
from ..workflow import magnification as wmag
from ..workflow.parameters import SettingsParam
from ..workflow.settings import SettingsStore
from .desktop_display import DisplayController
from .desktop_magnification import (
    MagnificationManagerDialog,
    show_geometry_calculator,
)
from .desktop_measure import MeasurementTool

PLUGIN_ICON = "datalab_radiography:icons/radiography.svg"
DOCUMENTATION_URL = "https://github.com/DataLab-Platform/datalab-radiography"

_LOGGER = logging.getLogger(__name__)


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

    # -- Messages ---------------------------------------------------------

    def inform(self, message: str, title: str | None = None) -> None:
        """Show an information message (logged only in unattended mode)."""
        if execenv.unattended:
            _LOGGER.info("%s", message)
            return
        QW.QMessageBox.information(self.main, title or PLUGIN_NAME, message)

    def warn(self, message: str, title: str | None = None) -> None:
        """Show a warning message (logged only in unattended mode)."""
        if execenv.unattended:
            _LOGGER.warning("%s", message)
            return
        QW.QMessageBox.warning(self.main, title or PLUGIN_NAME, message)

    # -- Registration -----------------------------------------------------

    def register_computations(self) -> None:
        """Register the processing features and watch new images."""
        processor = self.imagepanel.processor
        for feature in wfeatures.features_for(wfeatures.HOST_DESKTOP):
            options = {
                "feature_id": feature.feature_id,
                "owner_plugin_id": self.plugin_id,
            }
            if feature.pattern == wfeatures.PATTERN_1_TO_1:
                processor.register_1_to_1(
                    feature.function,
                    feature.title,
                    paramclass=feature.paramclass,
                    comment=feature.comment,
                    preview_enabled=feature.preview,
                    **options,
                )
            elif feature.pattern == wfeatures.PATTERN_1_TO_0:
                processor.register_1_to_0(
                    feature.function,
                    feature.title,
                    paramclass=feature.paramclass,
                    comment=feature.comment,
                    **options,
                )
            elif feature.pattern == wfeatures.PATTERN_2_TO_1:
                processor.register_2_to_1(
                    feature.function,
                    feature.title,
                    paramclass=feature.paramclass,
                    comment=feature.comment,
                    obj2_name=feature.obj2_name,
                    **options,
                )
            else:
                processor.register_n_to_1(
                    feature.function,
                    feature.title,
                    paramclass=feature.paramclass,
                    comment=feature.comment,
                    **options,
                )
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
        self.film_convention_action.blockSignals(True)
        self.film_convention_action.setChecked(
            self.settings_store.settings.film_convention
        )
        self.film_convention_action.blockSignals(False)
        return True

    def magnification_manager(self) -> MagnificationManagerDialog | None:
        """Open the magnification manager of the current image."""
        obj = self.imagepanel.objview.get_current_object()
        if obj is None:
            return None
        dialog = MagnificationManagerDialog(self, obj)
        if not execenv.unattended:
            dialog.exec()
            self._after_metadata_change(dialog.changed)
        return dialog

    def _after_metadata_change(self, changed: bool) -> None:
        """Refresh the panel after the plugin changed image metadata."""
        if changed:
            self.imagepanel.refresh_plot("selected", True, False)
            self.imagepanel.SIG_OBJECT_MODIFIED.emit()

    def set_detector_pitch(
        self, param: wmag.DetectorPitchParam | None = None
    ) -> tuple[float, float] | None:
        """Set the detector pixel pitch of the selected images."""
        images = self.display.selected_images()
        if not images:
            return None
        if param is None:
            param = wmag.detector_pitch_param(
                images[0], self.settings_store.settings.detector_pitch_mm
            )
            if not param.edit(parent=self.main):
                return None
        pitch = None
        for obj in images:
            pitch = wmag.set_detector_pitch(obj, param)
        self.settings_store.update(detector_pitch_mm=pitch[0])
        self._after_metadata_change(True)
        return pitch

    def geometry_calculator(
        self, param: wmag.GeometryCalculatorParam | None = None
    ) -> list[tuple[str, str]] | None:
        """Open the radiographic geometry calculator."""
        return show_geometry_calculator(self, param)

    def measurement_tool(self) -> MeasurementTool | None:
        """Open the measurement tool on the current image."""
        obj = self.imagepanel.objview.get_current_object()
        if obj is None:
            return None
        try:
            tool = MeasurementTool(self, obj)
            tool.exec()
        except ValueError as exc:
            self.warn(str(exc))
            return None
        return tool

    def create_actions(self) -> None:
        """Create the plugin menu of the image panel."""
        handler = self.imagepanel.acthandler
        display = self.display
        with handler.new_menu(PLUGIN_NAME.replace("&", "&&")):
            with handler.new_menu(wfeatures.GROUP_DISPLAY):
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
            with handler.new_menu(wfeatures.GROUP_MAGNIFICATION):
                self.magnification_action = handler.new_action(
                    "Magnifications...",
                    triggered=self.magnification_manager,
                    tip="Define, choose and copy the magnifications of the image",
                    select_condition="exactly_one",
                )
                self.pitch_action = handler.new_action(
                    "Detector pixel pitch...",
                    triggered=self.set_detector_pitch,
                    select_condition="at_least_one",
                )
                self._feature_actions(wfeatures.GROUP_MAGNIFICATION)
                self.calculator_action = handler.new_action(
                    "Geometry calculator...",
                    triggered=self.geometry_calculator,
                    tip="Magnification, unsharpness and optimal magnification",
                    select_condition="always",
                    separator=True,
                )
            with handler.new_menu(wfeatures.GROUP_MEASUREMENTS):
                self.measure_action = handler.new_action(
                    "Measurement tool...",
                    triggered=self.measurement_tool,
                    tip="Measure distances, angles, diameters and areas in every plane",
                    select_condition="exactly_one",
                )
                self._feature_actions(wfeatures.GROUP_MEASUREMENTS)
            self.settings_action = handler.new_action(
                "Preferences...",
                triggered=self.edit_settings,
                select_condition="always",
                separator=True,
            )

    def _feature_actions(self, group: str) -> list[QW.QAction]:
        """Create the actions of the registered features of a menu group."""
        handler = self.imagepanel.acthandler
        actions = []
        for feature in wfeatures.features_for(wfeatures.HOST_DESKTOP):
            if feature.group == group:
                actions.append(handler.action_for(feature.feature_id))
        return actions
