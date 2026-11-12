"""Geometric magnification of radiographs and coordinate frames.

In a radiograph taken with a point-like source, an object plane parallel to
the detector is magnified by M = SDD / SOD (source-to-detector distance over
source-to-object distance). A radiograph may show several planes of interest,
hence several named magnifications.

Lengths are expressed relative to the image coordinates: a magnification
stores the length, in millimeters in its object plane, of one unit of image
coordinates along X and Y (its *scale*). Processing that keeps the meaning of
image coordinates (binning updating the pixel size, ROI extraction, rotation)
keeps the magnifications valid.

The *coordinate frame* tells what one unit of image coordinates is:

- ``pixel``: one detector pixel (the detector pitch may be known or not);
- ``detector``: a length unit in the detector plane;
- ``object``: a length unit in an object plane of known magnification (for
  instance after :func:`express_factors`, or DICOM pixel spacing already
  corrected for magnification).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

from . import metadata, units

FRAME_PIXEL = "pixel"
FRAME_DETECTOR = "detector"
FRAME_OBJECT = "object"
FRAMES = (FRAME_PIXEL, FRAME_DETECTOR, FRAME_OBJECT)

METHOD_VALUE = "value"
METHOD_GEOMETRY = "geometry"
METHOD_LENGTH = "length"
METHOD_DIAMETER = "diameter"
METHOD_METADATA = "metadata"
METHODS = (
    METHOD_VALUE,
    METHOD_GEOMETRY,
    METHOD_LENGTH,
    METHOD_DIAMETER,
    METHOD_METADATA,
)
METHOD_TITLES = {
    METHOD_VALUE: "Magnification value",
    METHOD_GEOMETRY: "Source, object and detector distances",
    METHOD_LENGTH: "Known length",
    METHOD_DIAMETER: "Calibration ball or disk",
    METHOD_METADATA: "Acquisition metadata",
}

Pair = tuple[float, float]


class MagnificationError(ValueError):
    """Raised when a magnification cannot be defined or applied."""


def _positive(value: float, name: str) -> float:
    """Return a finite, strictly positive float."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MagnificationError(f"{name} must be a number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise MagnificationError(f"{name} must be strictly positive")
    return number


def _pair(value: float | Sequence[float], name: str) -> Pair:
    """Return a pair of strictly positive floats (a scalar applies to both)."""
    if isinstance(value, (int, float)):
        number = _positive(value, name)
        return (number, number)
    values = tuple(value)
    if len(values) != 2:
        raise MagnificationError(f"{name} must have two values (X and Y)")
    return (_positive(values[0], name), _positive(values[1], name))


def _mean_scale(pair: Pair) -> float:
    """Return the geometric mean of a pair of scales."""
    return math.sqrt(pair[0] * pair[1])


def positive_pair(value: float | Sequence[float], name: str) -> Pair:
    """Return a pair of strictly positive floats (a scalar applies to both).

    Raises:
        MagnificationError: if a value is not a strictly positive number
    """
    return _pair(value, name)


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CoordinateFrame:
    """What one unit of image coordinates represents.

    Args:
        frame: ``pixel``, ``detector`` or ``object``
        unit: length unit of the coordinates (``detector`` and ``object``)
        plane: name of the object plane (``object``)
        magnification: magnification of the object plane, if known (``object``)
    """

    frame: str = FRAME_PIXEL
    unit: str | None = None
    plane: str | None = None
    magnification: float | None = None

    def __post_init__(self) -> None:
        """Validate the frame."""
        if self.frame not in FRAMES:
            raise MagnificationError(f"Unknown coordinate frame {self.frame!r}")
        if self.frame == FRAME_PIXEL:
            if self.unit is not None or self.plane is not None:
                raise MagnificationError("Pixel coordinates have no unit and plane")
        else:
            unit = units.normalize_unit(self.unit)
            if unit is None:
                raise MagnificationError(
                    f"{self.frame.capitalize()} coordinates need a length unit"
                )
            object.__setattr__(self, "unit", unit)
        if self.magnification is not None:
            if self.frame != FRAME_OBJECT:
                raise MagnificationError("Only object frames have a magnification")
            object.__setattr__(
                self,
                "magnification",
                _positive(self.magnification, "Magnification"),
            )

    def detector_mm_per_unit(self, pitch_mm: Pair | None) -> Pair | None:
        """Return the detector-plane length of one coordinate unit, in mm.

        Args:
            pitch_mm: detector pixel pitch along X and Y, in mm, if known

        Returns:
            Length along X and Y, or None if it cannot be known
        """
        if self.frame == FRAME_PIXEL:
            return None if pitch_mm is None else _pair(pitch_mm, "Detector pitch")
        length = units.unit_mm(self.unit)
        if self.frame == FRAME_DETECTOR:
            return (length, length)
        if self.magnification is None:
            return None
        return (length * self.magnification, length * self.magnification)

    def describe(self) -> str:
        """Return a short description of the frame."""
        if self.frame == FRAME_PIXEL:
            return "pixels"
        if self.frame == FRAME_DETECTOR:
            return f"{self.unit} in the detector plane"
        plane = f" '{self.plane}'" if self.plane else ""
        mag = "" if self.magnification is None else f" (M = {self.magnification:g})"
        return f"{self.unit} in the object plane{plane}{mag}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "frame": self.frame,
            "unit": self.unit,
            "plane": self.plane,
            "magnification": self.magnification,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoordinateFrame:
        """Build a frame from :meth:`to_dict` output."""
        return cls(
            frame=data.get("frame", FRAME_PIXEL),
            unit=data.get("unit"),
            plane=data.get("plane"),
            magnification=data.get("magnification"),
        )

    def to_json(self) -> str:
        """Serialize the frame as versioned JSON."""
        return metadata.dumps(self.to_dict())

    @classmethod
    def from_json(cls, text: str | dict) -> CoordinateFrame:
        """Read a frame serialized by :meth:`to_json`."""
        return cls.from_dict(metadata.loads(text))


def default_frame(xunit: str | None, yunit: str | None) -> CoordinateFrame:
    """Guess the frame of an image without plugin metadata.

    Coordinates with a length unit on both axes are assumed to be in the
    detector plane; others are pixels.

    Args:
        xunit: unit of the X axis
        yunit: unit of the Y axis

    Returns:
        Coordinate frame
    """
    xu, yu = units.normalize_unit(xunit), units.normalize_unit(yunit)
    if xu is not None and xu == yu:
        return CoordinateFrame(FRAME_DETECTOR, xu)
    return CoordinateFrame(FRAME_PIXEL)


# ---------------------------------------------------------------------------
# Magnifications
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Magnification:
    """Named magnification of an object plane.

    Args:
        name: name of the object plane (e.g. "Pipe axis")
        scale_mm: length, in mm in the object plane, of one unit of image
         coordinates along X and Y
        method: definition method (one of :data:`METHODS`)
        factor: magnification M, if known
        details: parameters of the definition (distances, measured length...)
        remarks: free text
    """

    name: str
    scale_mm: Pair
    method: str = METHOD_VALUE
    factor: float | None = None
    details: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    remarks: str = ""

    def __post_init__(self) -> None:
        """Validate and normalize the magnification."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise MagnificationError("A magnification needs a name")
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "scale_mm", _pair(self.scale_mm, "Scale"))
        if self.method not in METHODS:
            raise MagnificationError(f"Unknown magnification method {self.method!r}")
        if self.factor is not None:
            object.__setattr__(self, "factor", _positive(self.factor, "Magnification"))
        object.__setattr__(self, "details", dict(self.details))
        object.__setattr__(self, "remarks", str(self.remarks or ""))

    @property
    def object_pixel_mm(self) -> float:
        """Mean object-plane length of one unit of image coordinates, in mm."""
        return _mean_scale(self.scale_mm)

    def length_mm(self, dx: float, dy: float) -> float:
        """Return the object-plane length of a vector given in image coordinates."""
        return math.hypot(dx * self.scale_mm[0], dy * self.scale_mm[1])

    def area_mm2(self, area: float) -> float:
        """Return the object-plane area of an area given in image coordinates."""
        return float(area) * self.scale_mm[0] * self.scale_mm[1]

    def rescaled(self, fx: float, fy: float) -> Magnification:
        """Return the magnification after image coordinates are multiplied.

        Args:
            fx: factor applied to X coordinates
            fy: factor applied to Y coordinates
        """
        return dataclasses.replace(
            self,
            scale_mm=(
                self.scale_mm[0] / _positive(fx, "Factor"),
                self.scale_mm[1] / _positive(fy, "Factor"),
            ),
        )

    def with_factor_from(self, detector_mm: Pair | None) -> Magnification:
        """Return the magnification with M computed from the detector scale."""
        if detector_mm is None:
            return self
        factor = _mean_scale(_pair(detector_mm, "Detector scale")) / (
            self.object_pixel_mm
        )
        return dataclasses.replace(self, factor=factor)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible mapping."""
        return {
            "name": self.name,
            "scale_mm": list(self.scale_mm),
            "method": self.method,
            "factor": self.factor,
            "details": dict(self.details),
            "remarks": self.remarks,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Magnification:
        """Build a magnification from :meth:`to_dict` output."""
        return cls(
            name=data["name"],
            scale_mm=tuple(data["scale_mm"]),
            method=data.get("method", METHOD_VALUE),
            factor=data.get("factor"),
            details=data.get("details") or {},
            remarks=data.get("remarks") or "",
        )


@dataclasses.dataclass(frozen=True)
class MagnificationSet:
    """Magnifications of an image, with the active one.

    Args:
        items: magnifications, with unique names
        active: name of the active magnification, or None
        coordinate_unit: unit of the image X axis when the set was last written,
         used to detect coordinates changed outside the plugin
    """

    items: tuple[Magnification, ...] = ()
    active: str | None = None
    coordinate_unit: str = ""

    def __post_init__(self) -> None:
        """Validate the set."""
        items = tuple(self.items)
        names = [item.name for item in items]
        if len(set(names)) != len(names):
            raise MagnificationError("Magnification names must be unique")
        object.__setattr__(self, "items", items)
        active = self.active
        if active is not None and active not in names:
            raise MagnificationError(f"Unknown active magnification {active!r}")
        if active is None and items:
            object.__setattr__(self, "active", items[0].name)
        object.__setattr__(self, "coordinate_unit", str(self.coordinate_unit or ""))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    @property
    def names(self) -> tuple[str, ...]:
        """Names of the magnifications."""
        return tuple(item.name for item in self.items)

    def get(self, name: str) -> Magnification:
        """Return a magnification by name.

        Raises:
            KeyError: if there is no such magnification
        """
        for item in self.items:
            if item.name == name:
                return item
        raise KeyError(f"No magnification named {name!r}")

    @property
    def active_item(self) -> Magnification | None:
        """Active magnification, if any."""
        return None if self.active is None else self.get(self.active)

    def with_item(
        self, item: Magnification, make_active: bool = False, replace: str | None = None
    ) -> MagnificationSet:
        """Return the set with a magnification added or replaced.

        Args:
            item: magnification to add
            make_active: if True, the new magnification becomes active
            replace: name of the magnification replaced by ``item`` (it keeps
             its position); by default, the magnification named like ``item``
        """
        target = item.name if replace is None else replace
        items = list(self.items)
        names = [existing.name for existing in items]
        if target in names:
            items[names.index(target)] = item
        else:
            items.append(item)
        active = self.active
        if make_active or active == target or active is None:
            active = item.name
        return dataclasses.replace(self, items=tuple(items), active=active)

    def without(self, name: str) -> MagnificationSet:
        """Return the set without a magnification."""
        self.get(name)
        items = tuple(item for item in self.items if item.name != name)
        active = self.active if self.active != name else None
        if active is None and items:
            active = items[0].name
        return dataclasses.replace(self, items=items, active=active)

    def with_active(self, name: str | None) -> MagnificationSet:
        """Return the set with another active magnification."""
        if name is not None:
            self.get(name)
        return dataclasses.replace(self, active=name)

    def rescaled(self, fx: float, fy: float) -> MagnificationSet:
        """Return the set after image coordinates are multiplied."""
        return dataclasses.replace(
            self, items=tuple(item.rescaled(fx, fy) for item in self.items)
        )

    def to_json(self) -> str:
        """Serialize the set as versioned JSON."""
        return metadata.dumps(
            {
                "items": [item.to_dict() for item in self.items],
                "active": self.active,
                "coordinate_unit": self.coordinate_unit,
            }
        )

    @classmethod
    def from_json(cls, text: str | dict | None) -> MagnificationSet:
        """Read a set serialized by :meth:`to_json` (empty set for None)."""
        data = metadata.loads(text)
        if data is None:
            return cls()
        return cls(
            items=tuple(Magnification.from_dict(item) for item in data["items"]),
            active=data.get("active"),
            coordinate_unit=data.get("coordinate_unit") or "",
        )


# ---------------------------------------------------------------------------
# Definition methods
# ---------------------------------------------------------------------------


def magnification_from_distances(
    sdd_mm: float, sod_mm: float | None = None, odd_mm: float | None = None
) -> float:
    """Return M = SDD / SOD from the source, object and detector distances.

    Args:
        sdd_mm: source-to-detector distance
        sod_mm: source-to-object distance (or give ``odd_mm``)
        odd_mm: object-to-detector distance (SOD = SDD - ODD)

    Raises:
        MagnificationError: if the distances are inconsistent
    """
    sdd = _positive(sdd_mm, "Source-to-detector distance")
    if (sod_mm is None) == (odd_mm is None):
        raise MagnificationError(
            "Give either the source-to-object or the object-to-detector distance"
        )
    if sod_mm is None:
        odd = float(odd_mm)
        if not math.isfinite(odd) or odd < 0.0:
            raise MagnificationError("Object-to-detector distance must be positive")
        sod = sdd - odd
    else:
        sod = float(sod_mm)
    if not math.isfinite(sod) or sod <= 0.0 or sod > sdd:
        raise MagnificationError(
            "The object must lie between the source and the detector (0 < SOD <= SDD)"
        )
    return sdd / sod


def geometric_unsharpness(focal_spot_mm: float, factor: float) -> float:
    """Return the geometric unsharpness Ug = f (M - 1) in the detector plane."""
    return _positive(focal_spot_mm, "Focal spot size") * (
        _positive(factor, "Magnification") - 1.0
    )


def image_unsharpness(
    focal_spot_mm: float, factor: float, detector_srb_mm: float
) -> float:
    """Return the total image unsharpness in the object plane, in mm.

    Combines the geometric unsharpness and the detector unsharpness (twice its
    basic spatial resolution SRb) with a cubic law, then divides by the
    magnification, as for the magnification technique of ISO 17636-2.
    """
    ug = geometric_unsharpness(focal_spot_mm, factor)
    ud = 2.0 * _positive(detector_srb_mm, "Detector basic spatial resolution")
    return (ug**3 + ud**3) ** (1.0 / 3.0) / factor


def optimal_magnification(focal_spot_mm: float, detector_srb_mm: float) -> float:
    """Return the magnification minimizing :func:`image_unsharpness`.

    v_opt = 1 + (2 SRb / f) ** 1.5
    """
    ratio = 2.0 * _positive(detector_srb_mm, "Detector basic spatial resolution")
    return 1.0 + (ratio / _positive(focal_spot_mm, "Focal spot size")) ** 1.5


def from_value(
    name: str, factor: float, detector_mm: Pair | None, remarks: str = ""
) -> Magnification:
    """Define a magnification from its value.

    Args:
        name: plane name
        factor: magnification M
        detector_mm: detector-plane length of one coordinate unit (X, Y)
        remarks: free text

    Raises:
        MagnificationError: if the detector scale is unknown
    """
    factor = _positive(factor, "Magnification")
    if detector_mm is None:
        raise MagnificationError(
            "The detector pixel pitch is unknown: enter it, or define the "
            "magnification from a known length"
        )
    kx, ky = _pair(detector_mm, "Detector scale")
    return Magnification(
        name, (kx / factor, ky / factor), METHOD_VALUE, factor, {}, remarks
    )


def from_geometry(
    name: str,
    sdd_mm: float,
    detector_mm: Pair | None,
    sod_mm: float | None = None,
    odd_mm: float | None = None,
    focal_spot_mm: float | None = None,
    remarks: str = "",
) -> Magnification:
    """Define a magnification from the acquisition distances.

    Args:
        name: plane name
        sdd_mm: source-to-detector distance
        detector_mm: detector-plane length of one coordinate unit (X, Y)
        sod_mm: source-to-object distance (or give ``odd_mm``)
        odd_mm: object-to-detector distance
        focal_spot_mm: optical focal spot size, to compute Ug
        remarks: free text
    """
    factor = magnification_from_distances(sdd_mm, sod_mm, odd_mm)
    item = from_value(name, factor, detector_mm, remarks)
    details: dict[str, Any] = {"sdd_mm": float(sdd_mm)}
    if sod_mm is not None:
        details["sod_mm"] = float(sod_mm)
    if odd_mm is not None:
        details["odd_mm"] = float(odd_mm)
    if focal_spot_mm:
        details["focal_spot_mm"] = float(focal_spot_mm)
        details["geometric_unsharpness_mm"] = geometric_unsharpness(
            focal_spot_mm, factor
        )
    return dataclasses.replace(item, method=METHOD_GEOMETRY, details=details)


def from_length(
    name: str,
    delta: Pair,
    true_length_mm: float,
    detector_mm: Pair | None = None,
    remarks: str = "",
) -> Magnification:
    """Define a magnification from a segment of known length.

    Args:
        name: plane name
        delta: segment vector (X, Y) in image coordinates
        true_length_mm: true length of the segment in its object plane
        detector_mm: detector-plane length of one coordinate unit (X, Y), if
         known (then M is computed)
        remarks: free text
    """
    dx, dy = (float(value) for value in delta)
    true_length = _positive(true_length_mm, "True length")
    details = {"delta": [dx, dy], "true_length_mm": true_length}
    if detector_mm is None:
        length = _positive(math.hypot(dx, dy), "Measured length")
        scale = true_length / length
        return Magnification(
            name, (scale, scale), METHOD_LENGTH, None, details, remarks
        )
    kx, ky = _pair(detector_mm, "Detector scale")
    detector_length = _positive(math.hypot(dx * kx, dy * ky), "Measured length")
    factor = detector_length / true_length
    return Magnification(
        name, (kx / factor, ky / factor), METHOD_LENGTH, factor, details, remarks
    )


def from_diameter(
    name: str,
    diameter: float,
    true_diameter_mm: float,
    detector_mm: Pair | None = None,
    remarks: str = "",
    fit_details: Mapping[str, Any] | None = None,
) -> Magnification:
    """Define a magnification from the fitted diameter of a ball or disk.

    Args:
        name: plane name
        diameter: diameter measured in image coordinates
        true_diameter_mm: true diameter of the ball or disk
        detector_mm: detector-plane length of one coordinate unit (X, Y), if
         known (then M is computed)
        remarks: free text
        fit_details: details of the circle fit (center, residual...)
    """
    measured = _positive(diameter, "Measured diameter")
    true_diameter = _positive(true_diameter_mm, "True diameter")
    details = {
        "diameter": measured,
        "true_diameter_mm": true_diameter,
        **dict(fit_details or {}),
    }
    if detector_mm is None:
        scale = true_diameter / measured
        return Magnification(
            name, (scale, scale), METHOD_DIAMETER, None, details, remarks
        )
    kx, ky = _pair(detector_mm, "Detector scale")
    factor = measured * _mean_scale((kx, ky)) / true_diameter
    return Magnification(
        name, (kx / factor, ky / factor), METHOD_DIAMETER, factor, details, remarks
    )


# ---------------------------------------------------------------------------
# Changing the coordinate frame
# ---------------------------------------------------------------------------


def express_factors(
    frame: CoordinateFrame,
    pitch_mm: Pair | None,
    target: str,
    unit: str | None = None,
    magnification: Magnification | None = None,
) -> tuple[Pair, CoordinateFrame]:
    """Return the factors converting image coordinates to another frame.

    New coordinates are the current ones multiplied by the factors.

    Args:
        frame: current coordinate frame
        pitch_mm: detector pixel pitch (X, Y) in mm, if known
        target: target frame (``pixel``, ``detector`` or ``object``)
        unit: length unit of the target frame (``detector`` and ``object``)
        magnification: object plane of the target frame (``object``)

    Returns:
        Factors along X and Y, and the target coordinate frame

    Raises:
        MagnificationError: if the conversion needs an unknown quantity
    """
    detector = frame.detector_mm_per_unit(pitch_mm)
    if target == FRAME_OBJECT:
        if magnification is None:
            raise MagnificationError("Choose the magnification of the object plane")
        length = units.unit_mm(unit)
        factors = (
            magnification.scale_mm[0] / length,
            magnification.scale_mm[1] / length,
        )
        factor = magnification.factor
        if factor is None and detector is not None:
            factor = magnification.with_factor_from(detector).factor
        new_frame = CoordinateFrame(FRAME_OBJECT, unit, magnification.name, factor)
        return factors, new_frame
    if detector is None:
        raise MagnificationError(
            "The detector pixel pitch is unknown: enter it to express "
            "coordinates in the detector plane or in pixels"
        )
    if target == FRAME_DETECTOR:
        length = units.unit_mm(unit)
        return (detector[0] / length, detector[1] / length), CoordinateFrame(
            FRAME_DETECTOR, unit
        )
    if target == FRAME_PIXEL:
        if pitch_mm is None:
            raise MagnificationError("The detector pixel pitch is unknown")
        px, py = _pair(pitch_mm, "Detector pitch")
        return (detector[0] / px, detector[1] / py), CoordinateFrame(FRAME_PIXEL)
    raise MagnificationError(f"Unknown coordinate frame {target!r}")
