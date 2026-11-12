"""Measurements of shapes drawn on a radiograph, in several planes.

Shapes are given in image coordinates. A *scale* gives the length, in mm, of
one unit of image coordinates along X and Y in a plane (detector plane or
object plane of a magnification). Each shape is measured in the scaled space,
so that anisotropic scales are handled exactly.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Sequence

import numpy as np

Pair = tuple[float, float]

LENGTH = "length"
AREA = "area"
ANGLE = "angle"
COORDINATE = "coordinate"
VALUE = "value"
DIMENSIONS = (LENGTH, AREA, ANGLE, COORDINATE, VALUE)


@dataclasses.dataclass(frozen=True)
class Shape:
    """Base class of measured shapes.

    Args:
        title: shape title, shown in measurement tables
    """

    title: str = ""

    KIND = ""

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        """Return (quantity, dimension, value) triplets in a scaled space.

        Lengths are in mm (or in coordinate units for a unit scale), areas in
        squared mm, angles in degrees.
        """
        raise NotImplementedError


def _scaled(points: np.ndarray, scale: Pair) -> np.ndarray:
    """Return points multiplied by a scale."""
    return np.asarray(points, dtype=float) * np.asarray(scale, dtype=float)


def _path_length(points: np.ndarray) -> float:
    """Return the length of a polygonal path."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.hypot(*np.diff(points, axis=0).T)))


def _angle_deg(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Return the angle (degrees, 0 to 180) between two rays from a vertex."""
    u, v = a - vertex, b - vertex
    norms = np.linalg.norm(u) * np.linalg.norm(v)
    if norms == 0:
        return float("nan")
    cosine = float(np.clip(np.dot(u, v) / norms, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


@dataclasses.dataclass(frozen=True)
class Point(Shape):
    """Point, with the image value at its position."""

    x: float = 0.0
    y: float = 0.0
    value: float = float("nan")

    KIND = "point"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        x, y = _scaled([self.x, self.y], scale)
        return [
            ("X", COORDINATE, x),
            ("Y", COORDINATE, y),
            ("Value", VALUE, self.value),
        ]


@dataclasses.dataclass(frozen=True)
class Segment(Shape):
    """Segment (distance between two points)."""

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0

    KIND = "segment"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        points = _scaled([[self.x0, self.y0], [self.x1, self.y1]], scale)
        return [("Length", LENGTH, _path_length(points))]


@dataclasses.dataclass(frozen=True)
class Polyline(Shape):
    """Open polygonal line (e.g. a crack); three points also give an angle."""

    points: tuple[Pair, ...] = ()

    KIND = "polyline"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        points = _scaled(self.points, scale)
        result = [("Length", LENGTH, _path_length(points))]
        if len(points) == 3:
            result.append(("Angle", ANGLE, _angle_deg(*points)))
        return result


@dataclasses.dataclass(frozen=True)
class Polygon(Shape):
    """Closed polygon."""

    points: tuple[Pair, ...] = ()

    KIND = "polygon"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        points = _scaled(self.points, scale)
        closed = np.vstack([points, points[:1]]) if len(points) else points
        x, y = points[:, 0], points[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
        result = [("Perimeter", LENGTH, _path_length(closed)), ("Area", AREA, area)]
        if len(points) == 3:
            result.append(("Angle", ANGLE, _angle_deg(*points)))
        return result


@dataclasses.dataclass(frozen=True)
class Circle(Shape):
    """Circle."""

    cx: float = 0.0
    cy: float = 0.0
    radius: float = 0.0

    KIND = "circle"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        sx, sy = scale
        if math.isclose(sx, sy):
            diameter = 2.0 * self.radius * sx
            return [
                ("Diameter", LENGTH, diameter),
                ("Area", AREA, math.pi * self.radius**2 * sx * sy),
            ]
        return Ellipse(self.title, self.cx, self.cy, self.radius, self.radius).measure(
            scale
        )


@dataclasses.dataclass(frozen=True)
class Ellipse(Shape):
    """Ellipse (semi-axes along its own rotated axes)."""

    cx: float = 0.0
    cy: float = 0.0
    radius_x: float = 0.0
    radius_y: float = 0.0
    angle: float = 0.0

    KIND = "ellipse"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        transform = (
            np.diag(scale)
            @ np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            @ np.diag([self.radius_x, self.radius_y])
        )
        radii = np.linalg.svd(transform, compute_uv=False)
        area = math.pi * self.radius_x * self.radius_y * scale[0] * scale[1]
        return [
            ("Major axis", LENGTH, 2.0 * float(radii[0])),
            ("Minor axis", LENGTH, 2.0 * float(radii[1])),
            ("Area", AREA, area),
        ]


@dataclasses.dataclass(frozen=True)
class Rectangle(Shape):
    """Rectangle centered on (x, y), possibly rotated by ``angle``."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0
    angle: float = 0.0

    KIND = "rectangle"

    def measure(self, scale: Pair) -> list[tuple[str, str, float]]:
        cos_a, sin_a = math.cos(self.angle), math.sin(self.angle)
        sx, sy = scale
        width = math.hypot(self.width * cos_a * sx, self.width * sin_a * sy)
        height = math.hypot(self.height * sin_a * sx, self.height * cos_a * sy)
        area = abs(self.width * self.height) * sx * sy
        return [
            ("Width", LENGTH, width),
            ("Height", LENGTH, height),
            ("Area", AREA, area),
        ]


@dataclasses.dataclass(frozen=True)
class Plane:
    """Plane in which shapes are measured (a column of a measurement table).

    Args:
        title: column title
        scale: length in mm of one unit of image coordinates (X, Y); None for
         image coordinates themselves
        factor: magnification of the plane, if known
    """

    title: str
    scale: Pair | None = None
    factor: float | None = None


@dataclasses.dataclass(frozen=True)
class MeasurementRow:
    """One quantity of one shape, in every plane.

    Args:
        shape: shape title
        kind: shape kind
        quantity: measured quantity
        dimension: dimension of the quantity (one of :data:`DIMENSIONS`)
        values: value in each plane (in the plane order of the table)
    """

    shape: str
    kind: str
    quantity: str
    dimension: str
    values: tuple[float, ...]


def measure_shapes(
    shapes: Sequence[Shape], planes: Sequence[Plane], unit_mm: float = 1.0
) -> list[MeasurementRow]:
    """Measure shapes in several planes.

    Args:
        shapes: shapes in image coordinates
        planes: planes of the table columns
        unit_mm: length of the display unit in mm, for physical planes

    Returns:
        One row per quantity of each shape. Physical lengths are in the display
        unit, areas in its square; image-coordinate values are unchanged.
    """
    rows: list[MeasurementRow] = []
    for shape in shapes:
        per_plane = []
        for plane in planes:
            scale = plane.scale if plane.scale is not None else (1.0, 1.0)
            measured = shape.measure(scale)
            if plane.scale is not None:
                measured = [
                    (name, dim, _to_unit(value, dim, unit_mm))
                    for name, dim, value in measured
                ]
            per_plane.append(measured)
        for index, (quantity, dimension, _value) in enumerate(per_plane[0]):
            values = tuple(measured[index][2] for measured in per_plane)
            if dimension == VALUE:
                values = (values[0],) + (float("nan"),) * (len(values) - 1)
            rows.append(
                MeasurementRow(shape.title, shape.KIND, quantity, dimension, values)
            )
    return rows


def _to_unit(value: float, dimension: str, unit_mm: float) -> float:
    """Convert a value in mm (or squared mm) to the display unit."""
    if dimension in (LENGTH, COORDINATE):
        return value / unit_mm
    if dimension == AREA:
        return value / unit_mm**2
    return value
