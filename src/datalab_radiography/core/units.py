"""Length units used to display and enter physical lengths.

Lengths are stored in millimeters; they are converted only for display and
input. The micrometer is written with the micro sign (U+00B5).
"""

from __future__ import annotations

import math

#: Length of one unit, in millimeters
LENGTH_UNITS_MM: dict[str, float] = {
    "µm": 1e-3,
    "mm": 1.0,
    "cm": 10.0,
    "m": 1000.0,
    "in": 25.4,
}
#: Default display unit of the plugin
DEFAULT_UNIT = "cm"

_ALIASES = {
    "um": "µm",
    "μm": "µm",  # Greek small letter mu
    "micron": "µm",
    "microns": "µm",
    "micrometer": "µm",
    "micrometre": "µm",
    "millimeter": "mm",
    "millimetre": "mm",
    "centimeter": "cm",
    "centimetre": "cm",
    "meter": "m",
    "metre": "m",
    "inch": "in",
    "inches": "in",
    '"': "in",
}


def normalize_unit(unit: str | None) -> str | None:
    """Return the canonical name of a length unit, or None if it is not one.

    Args:
        unit: unit name, as found for instance in image axis units

    Returns:
        Canonical unit name (a key of :data:`LENGTH_UNITS_MM`), or None
    """
    if unit is None:
        return None
    text = str(unit).strip()
    if text in LENGTH_UNITS_MM:
        return text
    return _ALIASES.get(text.lower(), _ALIASES.get(text))


def unit_mm(unit: str) -> float:
    """Return the length of one unit, in millimeters.

    Raises:
        ValueError: if the unit is not a length unit
    """
    canonical = normalize_unit(unit)
    if canonical is None:
        raise ValueError(f"Unknown length unit: {unit!r}")
    return LENGTH_UNITS_MM[canonical]


def to_mm(value: float, unit: str) -> float:
    """Convert a length to millimeters."""
    return float(value) * unit_mm(unit)


def from_mm(value_mm: float, unit: str) -> float:
    """Convert a length in millimeters to another unit."""
    return float(value_mm) / unit_mm(unit)


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a length from one unit to another."""
    return float(value) * unit_mm(from_unit) / unit_mm(to_unit)


def convert_area(value: float, from_unit: str, to_unit: str) -> float:
    """Convert an area from one squared unit to another."""
    return float(value) * (unit_mm(from_unit) / unit_mm(to_unit)) ** 2


def format_length(value_mm: float, unit: str, digits: int = 4) -> str:
    """Format a length given in millimeters in a unit, with significant digits."""
    if value_mm is None or not math.isfinite(value_mm):
        return "-"
    value = from_mm(value_mm, unit)
    return f"{value:.{digits}g} {normalize_unit(unit)}"
