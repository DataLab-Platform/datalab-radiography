"""Parameter sets (guidata) of the radiography plugin."""

from __future__ import annotations

import guidata.dataset as gds
from sigima.objects import ImageObj

from ..core import units
from ..core.display import Window
from . import display as wdisplay
from .settings import RadiographySettings

UNIT_CHOICES = tuple((unit, unit) for unit in units.LENGTH_UNITS_MM)


class SettingsParam(gds.DataSet):
    """X-ray Radiography & NDT preferences"""

    _display = gds.BeginGroup("Radiographic display")
    auto_display = gds.BoolItem(
        "Apply to new images",
        default=True,
        help="Display new images in gray scale with an automatic window",
    )
    film_convention = gds.BoolItem(
        "Film convention",
        default=False,
        help=(
            "Display dense matter bright, as on a film (inverted gray scale). "
            "DICOM images are displayed as their attributes require."
        ),
    )
    low_percentile = gds.FloatItem(
        "Lower percentile", default=0.5, min=0.0, max=50.0, unit="%"
    )
    high_percentile = gds.FloatItem(
        "Upper percentile", default=99.5, min=50.0, max=100.0, unit="%"
    )
    _display_end = gds.EndGroup("Radiographic display")
    _lengths = gds.BeginGroup("Lengths")
    unit = gds.ChoiceItem("Display unit", UNIT_CHOICES, default=units.DEFAULT_UNIT)
    _lengths_end = gds.EndGroup("Lengths")
    _dicom = gds.BeginGroup("DICOM")
    apply_dicom_attributes = gds.BoolItem(
        "Read the attributes of new DICOM images",
        default=True,
        help="Detector pixel spacing, magnification, inversion and window",
    )
    _dicom_end = gds.EndGroup("DICOM")

    @classmethod
    def from_settings(cls, settings: RadiographySettings) -> SettingsParam:
        """Return a parameter set filled with preferences."""
        param = cls()
        for name in (
            "auto_display",
            "film_convention",
            "low_percentile",
            "high_percentile",
            "unit",
            "apply_dicom_attributes",
        ):
            setattr(param, name, getattr(settings, name))
        return param

    def to_values(self) -> dict:
        """Return the edited preferences."""
        return {
            "auto_display": bool(self.auto_display),
            "film_convention": bool(self.film_convention),
            "low_percentile": float(self.low_percentile),
            "high_percentile": float(self.high_percentile),
            "unit": self.unit,
            "apply_dicom_attributes": bool(self.apply_dicom_attributes),
        }


_WINDOW_METHOD = gds.ValueProp(wdisplay.WINDOW_AUTO)


def _method_is(*methods: str) -> gds.FuncProp:
    """Return a property active for some window methods."""
    return gds.FuncProp(_WINDOW_METHOD, lambda method: method in methods)


class WindowParam(gds.DataSet):
    """Radiographic window"""

    method = gds.ChoiceItem(
        "Window",
        (
            (wdisplay.WINDOW_AUTO, "Percentiles of the image"),
            (wdisplay.WINDOW_ROI, "Percentiles in the regions of interest"),
            (wdisplay.WINDOW_DICOM, "DICOM window attributes"),
            (wdisplay.WINDOW_MANUAL, "Level and width"),
        ),
        default=wdisplay.WINDOW_AUTO,
    ).set_prop("display", store=_WINDOW_METHOD)
    low_percentile = gds.FloatItem(
        "Lower percentile", default=0.5, min=0.0, max=50.0, unit="%"
    ).set_prop("display", active=_method_is(wdisplay.WINDOW_AUTO, wdisplay.WINDOW_ROI))
    high_percentile = gds.FloatItem(
        "Upper percentile", default=99.5, min=50.0, max=100.0, unit="%"
    ).set_prop("display", active=_method_is(wdisplay.WINDOW_AUTO, wdisplay.WINDOW_ROI))
    level = gds.FloatItem("Level (center)", default=0.0).set_prop(
        "display", active=_method_is(wdisplay.WINDOW_MANUAL)
    )
    width = gds.FloatItem("Width", default=1.0, nonzero=True, min=0.0).set_prop(
        "display", active=_method_is(wdisplay.WINDOW_MANUAL)
    )


def apply_window_param(obj: ImageObj, param: WindowParam) -> Window:
    """Apply a window parameter set to an image.

    Raises:
        ValueError: if the window cannot be computed for the image
    """
    method = param.method
    if method in (wdisplay.WINDOW_AUTO, wdisplay.WINDOW_ROI):
        window = wdisplay.auto_window(
            obj,
            param.low_percentile,
            param.high_percentile,
            use_roi=method == wdisplay.WINDOW_ROI,
        )
    elif method == wdisplay.WINDOW_DICOM:
        window = wdisplay.dicom_window(obj)
        if window is None:
            raise ValueError("The image has no DICOM window attributes")
    else:
        window = Window.from_center_width(param.level, param.width)
    return wdisplay.update_window(obj, window, method)


class InversionParam(gds.DataSet):
    """Gray-scale inversion"""

    film_convention = gds.BoolItem(
        "Film convention (dense matter bright)",
        default=False,
        help=(
            "Changes the preference for new images and applies it to the "
            "selected images. DICOM images are displayed as their attributes "
            "require."
        ),
    )
