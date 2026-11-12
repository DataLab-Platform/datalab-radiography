"""Radiographic display in DataLab Desktop.

DataLab Desktop writes its default display options into an image when it
first displays it, before plugins are notified of new images. The controller
therefore applies the radiographic profile to new images whose colormap is
still the default one (or that are not displayed yet and carry no colormap),
then refreshes them. Images read from a workspace keep their settings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from datalab.config import Conf
from datalab.objectmodel import get_uuid
from sigima.objects import ImageObj

from ..core import display as cdisplay
from ..workflow import display as wdisplay
from ..workflow.parameters import WindowParam, apply_window_param

if TYPE_CHECKING:
    from datalab.gui.panel.image import ImagePanel

    from .desktop import XRayRadiographyNDTPlugin

_LOGGER = logging.getLogger(__name__)

#: Function called for each new image, returning True if it changed the image
NewImageHook = Callable[[ImageObj, bool], bool]


class DisplayController:
    """Apply the radiographic display to the images of DataLab Desktop.

    Args:
        plugin: plugin owning the controller
    """

    def __init__(self, plugin: XRayRadiographyNDTPlugin) -> None:
        self.plugin = plugin
        self.new_image_hooks: list[NewImageHook] = []
        self._known: set[str] = set()
        self._connected = False

    @property
    def panel(self) -> ImagePanel:
        """Image panel."""
        return self.plugin.imagepanel

    @property
    def settings(self):
        """Plugin preferences."""
        return self.plugin.settings_store.settings

    # -- Connection to the image panel ----------------------------------

    def connect(self) -> None:
        """Watch the images added to the image panel."""
        if self._connected:
            return
        self._known = {get_uuid(obj) for obj in self.panel.objmodel}
        self.panel.SIG_OBJECT_ADDED.connect(self._on_objects_added)
        self.panel.SIG_OBJECT_REMOVED.connect(self._on_objects_removed)
        self._connected = True

    def disconnect(self) -> None:
        """Stop watching the image panel."""
        if not self._connected:
            return
        for signal, slot in (
            (self.panel.SIG_OBJECT_ADDED, self._on_objects_added),
            (self.panel.SIG_OBJECT_REMOVED, self._on_objects_removed),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        self._connected = False

    def _on_objects_removed(self) -> None:
        """Forget the removed images."""
        self._known &= {get_uuid(obj) for obj in self.panel.objmodel}

    def _on_objects_added(self) -> None:
        """Prepare the images added to the panel."""
        panel = self.panel
        new_images = [obj for obj in panel.objmodel if get_uuid(obj) not in self._known]
        default_colormap = Conf.get_sigima_defaults("ima").get("colormap")
        for obj in new_images:
            uuid = get_uuid(obj)
            self._known.add(uuid)
            displayed = panel.plothandler.get(uuid) is not None
            try:
                changed = False
                for hook in self.new_image_hooks:
                    changed |= bool(hook(obj, displayed))
                if self.settings.auto_display:
                    changed |= self.prepare_new_image(obj, displayed, default_colormap)
            except Exception:  # pylint: disable=broad-except
                # A new image must never be refused because of its display
                _LOGGER.warning("Radiographic display of %s failed", uuid, exc_info=1)
                continue
            if changed and displayed:
                panel.refresh_plot(uuid, True, False)

    def prepare_new_image(
        self, obj: ImageObj, displayed: bool, default_colormap: str | None
    ) -> bool:
        """Apply the radiographic profile to a new image, if needed.

        Returns:
            True if the display settings of the image changed
        """
        settings = self.settings
        if not wdisplay.is_explicit_display(obj, displayed, default_colormap):
            wdisplay.apply_profile(
                obj,
                settings.film_convention,
                settings.low_percentile,
                settings.high_percentile,
            )
            return True
        if (
            wdisplay.has_profile(obj)
            and obj.zscalemin is None
            and obj.zscalemax is None
        ):
            # Processing results inherit the profile, not the window of the source
            window = wdisplay.auto_window(
                obj, settings.low_percentile, settings.high_percentile
            )
            wdisplay.update_window(obj, window, wdisplay.WINDOW_AUTO)
            return True
        return False

    # -- Actions on the selected images ---------------------------------

    def selected_images(self) -> list[ImageObj]:
        """Return the selected images (groups included)."""
        return list(self.panel.objview.get_sel_objects(include_groups=True))

    def _refresh(self, images: list[ImageObj]) -> None:
        """Refresh the plot of changed images and mark the workspace modified."""
        if images:
            self.panel.refresh_plot("selected", True, False)
            self.panel.SIG_OBJECT_MODIFIED.emit()

    def _apply(self, function: Callable[[ImageObj], None]) -> list[ImageObj]:
        """Apply a function to the selected images, reporting failures."""
        done, errors = [], []
        for obj in self.selected_images():
            try:
                function(obj)
            except ValueError as exc:
                errors.append(f"{obj.title}: {exc}")
            else:
                done.append(obj)
        self._refresh(done)
        if errors:
            self.plugin.warn("\n".join(errors))
        return done

    def apply_profile_to_selection(self) -> list[ImageObj]:
        """Apply the radiographic display to the selected images."""
        settings = self.settings
        return self._apply(
            lambda obj: wdisplay.apply_profile(
                obj,
                settings.film_convention,
                settings.low_percentile,
                settings.high_percentile,
            )
        )

    def set_film_convention(self, enabled: bool) -> list[ImageObj]:
        """Change the film convention preference and apply it to the selection."""
        self.plugin.settings_store.update(film_convention=bool(enabled))

        def invert(obj: ImageObj) -> None:
            obj.set_metadata_option(
                "invert_colormap", wdisplay.inversion_for(obj, bool(enabled))
            )

        return self._apply(invert)

    def auto_window(self, use_roi: bool = False) -> list[ImageObj]:
        """Set an automatic window on the selected images."""
        settings = self.settings
        source = wdisplay.WINDOW_ROI if use_roi else wdisplay.WINDOW_AUTO

        def window(obj: ImageObj) -> None:
            wdisplay.update_window(
                obj,
                wdisplay.auto_window(
                    obj, settings.low_percentile, settings.high_percentile, use_roi
                ),
                source,
            )

        return self._apply(window)

    def edit_window(self) -> list[ImageObj]:
        """Edit the window of the selected images."""
        images = self.selected_images()
        if not images:
            return []
        param = WindowParam()
        settings = self.settings
        param.low_percentile = settings.low_percentile
        param.high_percentile = settings.high_percentile
        current = images[0]
        if current.zscalemin is not None and current.zscalemax is not None:
            if current.zscalemax > current.zscalemin:
                window = cdisplay.Window(current.zscalemin, current.zscalemax)
                param.level, param.width = window.center, window.width
        if not param.edit(parent=self.plugin.main):
            return []
        return self._apply(lambda obj: apply_window_param(obj, param))
