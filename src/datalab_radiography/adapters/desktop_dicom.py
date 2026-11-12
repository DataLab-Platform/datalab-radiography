"""DICOM attributes of the images opened in DataLab Desktop.

DataLab reads the pixels of DICOM files but no attribute: the plugin reads the
header of the source file with pydicom when a DICOM image is added (if the
preference is enabled), or on demand for the selected images.
"""

from __future__ import annotations

import logging
import os.path as osp
from typing import TYPE_CHECKING, Any

from sigima.objects import ImageObj

from ..core import dicom as cdicom
from ..core import metadata as md
from ..workflow import dicom as wdicom

if TYPE_CHECKING:
    from .desktop import XRayRadiographyNDTPlugin

_LOGGER = logging.getLogger(__name__)
DICOM_EXTENSIONS = (".dcm", ".dicom")


def dicom_path(obj: ImageObj) -> str | None:
    """Return the DICOM file an image was read from, if any."""
    source = obj.metadata.get("source")
    if not isinstance(source, str) or not osp.isfile(source):
        return None
    if osp.splitext(source)[1].lower() in DICOM_EXTENSIONS:
        return source
    try:
        with open(source, "rb") as file:
            prefix = file.read(132)
    except OSError:
        return None
    return source if cdicom.is_dicom_header(prefix) else None


def read_header(path: str) -> dict[str, Any]:
    """Read the technical attributes of a DICOM file.

    Raises:
        ImportError: if pydicom is not installed
        ValueError: if the file cannot be read
    """
    import pydicom  # pylint: disable=import-outside-toplevel

    try:
        dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as exc:  # pylint: disable=broad-except
        raise ValueError(f"Cannot read the DICOM header of {path}: {exc}") from exc
    return cdicom.attributes({key: dataset.get(key) for key in cdicom.KEYWORDS})


class DicomController:
    """Read the DICOM attributes of images.

    Args:
        plugin: plugin owning the controller
    """

    def __init__(self, plugin: XRayRadiographyNDTPlugin) -> None:
        self.plugin = plugin
        self._missing_pydicom_reported = False

    def _read(self, path: str) -> dict[str, Any] | None:
        """Read a header, reporting a missing pydicom once."""
        try:
            return read_header(path)
        except ImportError:
            if not self._missing_pydicom_reported:
                self._missing_pydicom_reported = True
                self.plugin.warn(
                    "pydicom is not installed: the DICOM attributes (pixel "
                    "spacing, magnification, window) cannot be read. Install it "
                    "in the Python environment of DataLab."
                )
            return None

    def on_new_image(self, obj: ImageObj, _displayed: bool) -> bool:
        """Apply the attributes of a new DICOM image (display controller hook)."""
        settings = self.plugin.settings_store.settings
        if not settings.apply_dicom_attributes or md.DICOM_KEY in obj.metadata:
            return False
        path = dicom_path(obj)
        if path is None:
            return False
        try:
            attrs = self._read(path)
        except ValueError as exc:
            _LOGGER.warning("%s", exc)
            return False
        if not attrs or not wdicom.matches(obj, attrs):
            return False
        wdicom.apply_attributes(obj, attrs)
        return True

    def apply_to_selection(self) -> list[ImageObj]:
        """Read the attributes of the selected DICOM images and apply them."""
        settings = self.plugin.settings_store.settings

        def apply(obj: ImageObj) -> None:
            path = dicom_path(obj)
            if path is None:
                raise ValueError("the image was not read from a DICOM file")
            attrs = self._read(path)
            if attrs is None:
                raise ValueError("pydicom is not installed")
            if not wdicom.matches(obj, attrs):
                raise ValueError("the image size differs from the DICOM file")
            wdicom.apply_attributes(obj, attrs, settings, force=True)

        return self.plugin.display.apply_to_selection(apply)
