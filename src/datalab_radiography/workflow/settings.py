"""User preferences of the radiography plugin.

DataLab Desktop stores them in a JSON file of its configuration folder;
DataLab-Web keeps them for the session. Unknown or invalid values fall back to
their defaults, so that a preference file written by another plugin version
never prevents the plugin from loading.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

from ..core import units

_LOGGER = logging.getLogger(__name__)


@dataclasses.dataclass
class RadiographySettings:
    """Preferences of the radiography plugin.

    Attributes:
        auto_display: apply the radiographic display profile to new images
        film_convention: display dense matter bright (inverted gray scale),
         except for DICOM images, displayed as their attributes require
        low_percentile: lower percentile of the automatic window
        high_percentile: upper percentile of the automatic window
        unit: display unit of lengths
        apply_dicom_attributes: read pixel spacing, magnification and window
         from the attributes of new DICOM images
        detector_pitch_mm: last detector pixel pitch entered, in mm
        measurement_window: geometry of the measurement tool window (Qt state
         encoded in base64), or None for the default size
    """

    auto_display: bool = True
    film_convention: bool = False
    low_percentile: float = 0.5
    high_percentile: float = 99.5
    unit: str = units.DEFAULT_UNIT
    apply_dicom_attributes: bool = True
    detector_pitch_mm: float | None = None
    measurement_window: str | None = None

    def __post_init__(self) -> None:
        """Replace invalid values by their defaults."""
        defaults = {field.name: field.default for field in dataclasses.fields(self)}
        for name in ("auto_display", "film_convention", "apply_dicom_attributes"):
            if not isinstance(getattr(self, name), bool):
                setattr(self, name, defaults[name])
        low, high = self.low_percentile, self.high_percentile
        if not (
            isinstance(low, (int, float))
            and isinstance(high, (int, float))
            and 0.0 <= low < high <= 100.0
        ):
            self.low_percentile = defaults["low_percentile"]
            self.high_percentile = defaults["high_percentile"]
        canonical = units.normalize_unit(self.unit)
        self.unit = canonical if canonical is not None else defaults["unit"]
        pitch = self.detector_pitch_mm
        if pitch is not None and not (
            isinstance(pitch, (int, float)) and math.isfinite(pitch) and pitch > 0
        ):
            self.detector_pitch_mm = None
        if not (
            self.measurement_window is None or isinstance(self.measurement_window, str)
        ):
            self.measurement_window = None

    def to_dict(self) -> dict[str, Any]:
        """Return the preferences as a JSON-compatible mapping."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RadiographySettings:
        """Build preferences from a mapping, ignoring unknown keys."""
        names = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in names})


class SettingsStore:
    """Load and save the preferences, in a JSON file or in memory.

    Args:
        path: JSON file, or None to keep the preferences in memory
    """

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = None if path is None else Path(path)
        self.settings = self.load()

    def load(self) -> RadiographySettings:
        """Read the preferences (defaults if the file is missing or invalid)."""
        if self.path is None or not self.path.is_file():
            return RadiographySettings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            return RadiographySettings.from_dict(data)
        except (OSError, ValueError) as exc:
            _LOGGER.warning("Ignoring radiography preferences %s: %s", self.path, exc)
            return RadiographySettings()

    def save(self) -> None:
        """Write the preferences (does nothing for in-memory preferences)."""
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.settings.to_dict(), indent=2, sort_keys=True)
        self.path.write_text(text + "\n", encoding="utf-8")

    def update(self, **values: Any) -> RadiographySettings:
        """Change some preferences, validate them and save them."""
        data = self.settings.to_dict()
        data.update(values)
        self.settings = RadiographySettings.from_dict(data)
        self.save()
        return self.settings
