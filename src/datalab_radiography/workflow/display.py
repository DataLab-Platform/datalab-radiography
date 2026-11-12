"""Radiographic display profile of Sigima images.

The profile sets the gray-scale colormap, its inversion and a window on the
display options of an image (DataLab metadata options and LUT range). It never
changes the image data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sigima.objects import ImageObj

from ..core import display
from ..core import metadata as md

COLORMAP = "gray"
WINDOW_AUTO = "auto"
WINDOW_ROI = "roi"
WINDOW_DICOM = "dicom"
WINDOW_MANUAL = "manual"
WINDOW_SOURCES = (WINDOW_AUTO, WINDOW_ROI, WINDOW_DICOM, WINDOW_MANUAL)


@dataclasses.dataclass(frozen=True)
class DisplayProfile:
    """Radiographic display applied to an image.

    Args:
        invert: True if the gray scale is inverted
        window: displayed range of values
        window_source: origin of the window (one of :data:`WINDOW_SOURCES`)
        film_convention: film convention preference when the profile was applied
    """

    invert: bool
    window: display.Window
    window_source: str
    film_convention: bool

    def to_json(self) -> str:
        """Serialize the profile as versioned JSON."""
        return md.dumps(
            {
                "colormap": COLORMAP,
                "invert": self.invert,
                "window": [self.window.low, self.window.high],
                "window_source": self.window_source,
                "film_convention": self.film_convention,
            }
        )


def get_dicom_attributes(obj: ImageObj) -> dict:
    """Return the DICOM attributes stored by the plugin (empty if none)."""
    return md.loads(obj.metadata.get(md.DICOM_KEY)) or {}


def has_profile(obj: ImageObj) -> bool:
    """Return True if the radiographic profile was applied to the image."""
    return md.DISPLAY_PROFILE_KEY in obj.metadata


def roi_inside_mask(obj: ImageObj) -> np.ndarray | None:
    """Return a boolean mask, True inside the ROIs of the image (None if none)."""
    if obj.roi is None or obj.roi.is_empty():
        return None
    return ~obj.roi.to_mask(obj)


def auto_window(
    obj: ImageObj, low: float = 0.5, high: float = 99.5, use_roi: bool = False
) -> display.Window:
    """Return a percentile window of an image, in its ROIs if requested.

    Raises:
        ValueError: if ``use_roi`` is True and the image has no ROI
    """
    mask = None
    if use_roi:
        mask = roi_inside_mask(obj)
        if mask is None or not mask.any():
            raise ValueError("Define a region of interest on the image first")
    return display.percentile_window(obj.data, low, high, mask=mask)


def dicom_window(obj: ImageObj) -> display.Window | None:
    """Return the window given by the DICOM attributes of an image, if any."""
    attrs = get_dicom_attributes(obj)
    center, width = attrs.get("WindowCenter"), attrs.get("WindowWidth")
    if center is None or width is None:
        return None
    try:
        return display.window_from_dicom(
            float(center),
            float(width),
            float(attrs.get("RescaleSlope", 1.0)),
            float(attrs.get("RescaleIntercept", 0.0)),
        )
    except ValueError:
        return None


def inversion_for(obj: ImageObj, film_convention: bool) -> bool:
    """Return the gray-scale inversion of an image for a film convention."""
    attrs = get_dicom_attributes(obj)
    return display.display_inversion(
        film_convention,
        attrs.get("PhotometricInterpretation"),
        attrs.get("PresentationLUTShape"),
    )


def set_window(obj: ImageObj, window: display.Window) -> None:
    """Set the displayed range of values of an image."""
    obj.zscalemin, obj.zscalemax = window.low, window.high


def apply_profile(
    obj: ImageObj,
    film_convention: bool = False,
    low: float = 0.5,
    high: float = 99.5,
    window: display.Window | None = None,
    window_source: str = WINDOW_MANUAL,
) -> DisplayProfile:
    """Apply the radiographic display profile to an image.

    Args:
        obj: image
        film_convention: display dense matter bright (ignored for DICOM images,
         displayed as their attributes require)
        low: lower percentile of the automatic window
        high: upper percentile of the automatic window
        window: window to apply; by default, the DICOM window of the image if
         any, otherwise an automatic window
        window_source: origin of ``window`` when it is given

    Returns:
        Applied profile
    """
    if window is None:
        window = dicom_window(obj)
        window_source = WINDOW_DICOM
        if window is None:
            window = auto_window(obj, low, high)
            window_source = WINDOW_AUTO
    if window_source not in WINDOW_SOURCES:
        raise ValueError(f"Unknown window source {window_source!r}")
    profile = DisplayProfile(
        inversion_for(obj, film_convention), window, window_source, film_convention
    )
    obj.set_metadata_option("colormap", COLORMAP)
    obj.set_metadata_option("invert_colormap", profile.invert)
    set_window(obj, window)
    obj.metadata[md.DISPLAY_PROFILE_KEY] = profile.to_json()
    return profile


def update_window(
    obj: ImageObj, window: display.Window, window_source: str
) -> display.Window:
    """Change the window of an image, keeping the rest of its profile."""
    if window_source not in WINDOW_SOURCES:
        raise ValueError(f"Unknown window source {window_source!r}")
    set_window(obj, window)
    data = md.loads(obj.metadata.get(md.DISPLAY_PROFILE_KEY)) or {}
    if data:
        data.update(window=[window.low, window.high], window_source=window_source)
        obj.metadata[md.DISPLAY_PROFILE_KEY] = md.dumps(data)
    return window


def is_explicit_display(
    obj: ImageObj, displayed: bool, default_colormap: str | None
) -> bool:
    """Return True if an image carries display settings that must be kept.

    DataLab Desktop writes its default display options into an image when it
    first displays it. An image already displayed whose colormap is still the
    default one has no explicit settings; an image not displayed yet has
    explicit settings if it carries a colormap (read from a file or a
    workspace, or inherited from a processed image).

    Args:
        obj: image
        displayed: True if the image was already displayed
        default_colormap: default colormap of the host, if any
    """
    colormap = obj.metadata.get("__colormap")
    if colormap is None:
        return False
    return not displayed or colormap != default_colormap
