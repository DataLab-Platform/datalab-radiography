"""Deterministic simulation of X-ray radiographs with known ground truth.

A scene is made of components (plate, weld bead, pores, crack, wires, image
quality indicators, step wedge, ball, pipe) lying in object planes parallel to
the detector. Each plane has its own magnification M = SDD / SOD, so that a
single radiograph may hold several magnifications.

The model is a monochromatic Beer-Lambert law with an optional beam hardening
term per material, A(t) = mu t / (1 + beta t), where t is the thickness
crossed. The transmission of each plane is blurred by the focal spot (geometric
unsharpness Ug = f (M - 1), in the detector plane), then the detector adds its
blur, a heel effect, a gain and offset non-uniformity, Poisson and read noise,
defective pixels and saturation.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy import ndimage

#: Effective linear attenuation coefficients (1/mm) and beam hardening
#: coefficients (1/mm), illustrative values for a 150 to 200 kV spectrum
MATERIALS: dict[str, tuple[float, float]] = {
    "steel": (0.15, 0.010),
    "aluminum": (0.05, 0.004),
    "copper": (0.20, 0.012),
    "platinum": (3.5, 0.0),
    "air": (0.0, 0.0),
}

#: Wire diameters (mm) of the duplex elements D1 to D13 (ISO 19232-5)
DUPLEX_DIAMETERS = (
    0.80,
    0.63,
    0.50,
    0.40,
    0.32,
    0.25,
    0.20,
    0.16,
    0.13,
    0.10,
    0.080,
    0.063,
    0.050,
)
#: Wire diameters (mm) of the wire IQI W6 to W12 (ISO 19232-1, W6 FE set)
WIRE_DIAMETERS = (1.00, 0.80, 0.63, 0.50, 0.40, 0.32, 0.25)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Component:
    """Part of a scene, made of one material and lying in one object plane.

    Coordinates are in mm in the object plane, centered on the central ray.

    Args:
        name: component name
        plane: name of the object plane
        material: material name (key of :data:`MATERIALS`)
        sign: +1 for matter, -1 for a void removing matter of the same material
        supersampling: sub-samples per pixel and axis when rendering
    """

    name: str
    plane: str
    material: str = "steel"
    sign: float = 1.0
    supersampling: int = 3

    def bbox(self) -> tuple[float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax) of the component, in mm."""
        raise NotImplementedError

    def thickness(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return the thickness crossed at plane coordinates (mm)."""
        raise NotImplementedError

    def truth(self) -> dict[str, Any]:
        """Return the ground truth of the component."""
        return {"name": self.name, "plane": self.plane, "material": self.material}


@dataclasses.dataclass(frozen=True)
class Box(Component):
    """Slab of constant thickness over a rectangle."""

    x0: float = -1e3
    x1: float = 1e3
    y0: float = -1e3
    y1: float = 1e3
    depth: float = 10.0

    def bbox(self):
        return self.x0, self.x1, self.y0, self.y1

    def thickness(self, x, y):
        inside = (x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1)
        return np.where(inside, self.depth, 0.0)

    def truth(self):
        return {**super().truth(), "thickness_mm": self.depth}


@dataclasses.dataclass(frozen=True)
class WeldBead(Component):
    """Weld reinforcement along X: parabolic cap of a given width and height."""

    y: float = 0.0
    width: float = 10.0
    height: float = 2.0
    x0: float = -1e3
    x1: float = 1e3

    def bbox(self):
        return self.x0, self.x1, self.y - self.width / 2, self.y + self.width / 2

    def thickness(self, x, y):
        u = (y - self.y) / (self.width / 2)
        cap = self.height * np.clip(1.0 - u**2, 0.0, None)
        return np.where((x >= self.x0) & (x < self.x1), cap, 0.0)


@dataclasses.dataclass(frozen=True)
class Sphere(Component):
    """Sphere (pore with ``sign=-1``, or calibration ball)."""

    x: float = 0.0
    y: float = 0.0
    radius: float = 1.0
    supersampling: int = 5

    def bbox(self):
        r = self.radius
        return self.x - r, self.x + r, self.y - r, self.y + r

    def thickness(self, x, y):
        squared = self.radius**2 - (x - self.x) ** 2 - (y - self.y) ** 2
        return 2.0 * np.sqrt(np.clip(squared, 0.0, None))

    def truth(self):
        return {
            **super().truth(),
            "center_mm": [self.x, self.y],
            "diameter_mm": 2 * self.radius,
        }


@dataclasses.dataclass(frozen=True)
class Crack(Component):
    """Thin planar crack along a segment, removing matter over its depth."""

    xa: float = 0.0
    ya: float = 0.0
    xb: float = 10.0
    yb: float = 0.0
    width: float = 0.1
    depth: float = 3.0
    sign: float = -1.0
    supersampling: int = 6

    def bbox(self):
        margin = self.width
        return (
            min(self.xa, self.xb) - margin,
            max(self.xa, self.xb) + margin,
            min(self.ya, self.yb) - margin,
            max(self.ya, self.yb) + margin,
        )

    def thickness(self, x, y):
        vx, vy = self.xb - self.xa, self.yb - self.ya
        length2 = vx * vx + vy * vy
        t = np.clip(((x - self.xa) * vx + (y - self.ya) * vy) / length2, 0.0, 1.0)
        distance = np.hypot(x - (self.xa + t * vx), y - (self.ya + t * vy))
        return np.where(distance <= self.width / 2, self.depth, 0.0)

    def truth(self):
        length = math.hypot(self.xb - self.xa, self.yb - self.ya)
        return {
            **super().truth(),
            "segment_mm": [self.xa, self.ya, self.xb, self.yb],
            "length_mm": length,
            "width_mm": self.width,
            "depth_mm": self.depth,
        }


def _wire_integral(u: np.ndarray, radius: float) -> np.ndarray:
    """Antiderivative of the chord 2 sqrt(r² - u²) of a wire, clipped at ±r."""
    u = np.clip(u, -radius, radius)
    return u * np.sqrt(radius**2 - u**2) + radius**2 * np.arcsin(u / radius)


@dataclasses.dataclass(frozen=True)
class Wires(Component):
    """Parallel wires along Y, given by their X centers and diameters.

    Rendering integrates the circular cross-section exactly over each pixel
    along X, so that wires thinner than a pixel are rendered accurately.
    """

    centers: tuple[float, ...] = ()
    diameters: tuple[float, ...] = ()
    y0: float = -5.0
    y1: float = 5.0
    supersampling: int = 1

    def bbox(self):
        radius = max(self.diameters) / 2
        return (
            min(self.centers) - radius,
            max(self.centers) + radius,
            self.y0,
            self.y1,
        )

    def thickness(self, x, y):
        total = np.zeros(np.broadcast(x, y).shape)
        for center, diameter in zip(self.centers, self.diameters):
            r = diameter / 2
            total += 2.0 * np.sqrt(np.clip(r**2 - (x - center) ** 2, 0.0, None))
        return np.where((y >= self.y0) & (y < self.y1), total, 0.0)

    def average_thickness(self, x, y, half_width):
        """Return the thickness averaged over pixels of half-width ``half_width``."""
        total = np.zeros(np.broadcast(x, y).shape)
        for center, diameter in zip(self.centers, self.diameters):
            r = diameter / 2
            left = _wire_integral(x - half_width - center, r)
            right = _wire_integral(x + half_width - center, r)
            total += (right - left) / (2.0 * half_width)
        return np.where((y >= self.y0) & (y < self.y1), total, 0.0)

    def truth(self):
        return {
            **super().truth(),
            "centers_mm": list(self.centers),
            "diameters_mm": list(self.diameters),
        }


def duplex_iqi(
    name: str,
    plane: str,
    x0: float,
    y0: float,
    length: float = 15.0,
    elements: Sequence[int] = range(1, 14),
) -> Wires:
    """Return a duplex wire IQI (ISO 19232-5) starting at ``x0``.

    Each element Dn is a pair of wires of diameter d with a gap d between them;
    elements are spaced so that their profiles do not overlap.
    """
    centers, diameters = [], []
    x = x0
    for number in elements:
        d = DUPLEX_DIAMETERS[number - 1]
        centers += [x + d / 2, x + 2.5 * d]
        diameters += [d, d]
        x += 3 * d + max(1.0, 2 * d)
    return Wires(
        name,
        plane,
        "platinum",
        centers=tuple(centers),
        diameters=tuple(diameters),
        y0=y0,
        y1=y0 + length,
    )


def wire_iqi(
    name: str, plane: str, x0: float, y0: float, length: float = 25.0
) -> Wires:
    """Return a wire IQI W6 to W12 (ISO 19232-1) with 5 mm spacing."""
    centers = tuple(x0 + 5.0 * index for index in range(len(WIRE_DIAMETERS)))
    return Wires(
        name,
        plane,
        "copper",
        centers=centers,
        diameters=WIRE_DIAMETERS,
        y0=y0,
        y1=y0 + length,
    )


@dataclasses.dataclass(frozen=True)
class StepWedge(Component):
    """Step wedge: steps of increasing thickness along X."""

    x0: float = -20.0
    step_width: float = 5.0
    y0: float = -5.0
    y1: float = 5.0
    steps: tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 10.0)
    supersampling: int = 2

    def bbox(self):
        return self.x0, self.x0 + self.step_width * len(self.steps), self.y0, self.y1

    def thickness(self, x, y):
        index = np.floor((x - self.x0) / self.step_width).astype(int)
        valid = (index >= 0) & (index < len(self.steps)) & (y >= self.y0)
        valid &= y < self.y1
        steps = np.asarray(self.steps)
        return np.where(valid, steps[np.clip(index, 0, len(steps) - 1)], 0.0)

    def step_boxes(self) -> list[tuple[float, float, float, float]]:
        """Return (xmin, xmax, ymin, ymax) of each step, in mm."""
        return [
            (
                self.x0 + i * self.step_width,
                self.x0 + (i + 1) * self.step_width,
                self.y0,
                self.y1,
            )
            for i in range(len(self.steps))
        ]

    def truth(self):
        return {
            **super().truth(),
            "steps_mm": list(self.steps),
            "step_boxes_mm": [list(box) for box in self.step_boxes()],
        }


@dataclasses.dataclass(frozen=True)
class Pipe(Component):
    """Pipe along Y seen tangentially: annulus of outer and inner radii."""

    x: float = 0.0
    outer_radius: float = 20.0
    inner_radius: float = 15.0
    y0: float = -1e3
    y1: float = 1e3
    supersampling: int = 3

    def bbox(self):
        r = self.outer_radius
        return self.x - r, self.x + r, self.y0, self.y1

    def thickness(self, x, y):
        u2 = (x - self.x) ** 2
        outer = 2.0 * np.sqrt(np.clip(self.outer_radius**2 - u2, 0.0, None))
        inner = 2.0 * np.sqrt(np.clip(self.inner_radius**2 - u2, 0.0, None))
        return np.where((y >= self.y0) & (y < self.y1), outer - inner, 0.0)

    def truth(self):
        return {
            **super().truth(),
            "axis_x_mm": self.x,
            "outer_diameter_mm": 2 * self.outer_radius,
            "inner_diameter_mm": 2 * self.inner_radius,
            "wall_thickness_mm": self.outer_radius - self.inner_radius,
        }


# ---------------------------------------------------------------------------
# Scene, detector and acquisition
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Scene:
    """Components and the source-to-object distance of their planes.

    Args:
        name: scene name
        planes: plane name to source-to-object distance (mm)
        components: components of the scene
        description: short description
    """

    name: str
    planes: Mapping[str, float]
    components: tuple[Component, ...]
    description: str = ""

    def __post_init__(self) -> None:
        """Check that every component lies in a declared plane."""
        for component in self.components:
            if component.plane not in self.planes:
                raise ValueError(f"Unknown plane {component.plane!r}")
            if component.material not in MATERIALS:
                raise ValueError(f"Unknown material {component.material!r}")


@dataclasses.dataclass(frozen=True)
class Geometry:
    """Acquisition geometry.

    Args:
        sdd_mm: source-to-detector distance
        focal_spot_mm: optical focal spot size
        voltage_kv: tube voltage (informative)
    """

    sdd_mm: float = 1000.0
    focal_spot_mm: float = 0.4
    voltage_kv: float = 160.0


@dataclasses.dataclass(frozen=True)
class Detector:
    """Flat-panel detector model.

    Args:
        rows: number of rows
        cols: number of columns
        pitch_mm: pixel pitch
        bit_depth: number of bits of the digitizer
        offset_dn: mean offset (dark level)
        offset_nonuniformity_dn: standard deviation of the offset map
        gain_nonuniformity: relative standard deviation of the gain map
        dn_per_photon: digital numbers per detected photon
        read_noise_dn: standard deviation of the read noise
        blur_px: standard deviation of the detector blur, in pixels
        heel: relative flux variation across the field along X (heel effect)
        dead_fraction: fraction of dead pixels (stuck at zero)
        hot_fraction: fraction of hot pixels (saturated)
        seed: seed of the fixed pattern (gain, offset, defects)
    """

    rows: int = 400
    cols: int = 500
    pitch_mm: float = 0.1
    bit_depth: int = 14
    offset_dn: float = 400.0
    offset_nonuniformity_dn: float = 6.0
    gain_nonuniformity: float = 0.04
    dn_per_photon: float = 0.5
    read_noise_dn: float = 4.0
    blur_px: float = 0.7
    heel: float = 0.15
    dead_fraction: float = 2e-4
    hot_fraction: float = 2e-4
    seed: int = 1

    @property
    def max_dn(self) -> int:
        """Largest digital number."""
        return 2**self.bit_depth - 1

    def fixed_pattern(self) -> dict[str, np.ndarray]:
        """Return the gain map, offset map and defect masks of the detector."""
        rng = np.random.default_rng(self.seed)
        shape = (self.rows, self.cols)
        smooth = ndimage.gaussian_filter(rng.normal(0, 1, shape), 25)
        smooth /= max(float(smooth.std()), 1e-12)
        pixel = rng.normal(0, 1, shape)
        gain = 1.0 + self.gain_nonuniformity * (0.8 * smooth + 0.6 * pixel)
        offset = self.offset_dn + self.offset_nonuniformity_dn * rng.normal(0, 1, shape)
        dead = rng.random(shape) < self.dead_fraction
        hot = (rng.random(shape) < self.hot_fraction) & ~dead
        return {"gain": gain, "offset": offset, "dead": dead, "hot": hot}


def magnifications(scene: Scene, geometry: Geometry) -> dict[str, float]:
    """Return the magnification of each plane of a scene."""
    result = {}
    for plane, sod in scene.planes.items():
        if not 0 < sod <= geometry.sdd_mm:
            raise ValueError(f"Plane {plane!r} must lie between source and detector")
        result[plane] = geometry.sdd_mm / sod
    return result


def _pixel_grid(detector: Detector) -> tuple[np.ndarray, np.ndarray]:
    """Return detector-plane X (columns) and Y (rows) of pixel centers, in mm."""
    u = (np.arange(detector.cols) - (detector.cols - 1) / 2) * detector.pitch_mm
    v = (np.arange(detector.rows) - (detector.rows - 1) / 2) * detector.pitch_mm
    return u, v


def render_thickness(
    component: Component, detector: Detector, factor: float
) -> tuple[np.ndarray, tuple[slice, slice]] | None:
    """Render the pixel-averaged thickness of a component on the detector.

    Args:
        component: component
        detector: detector
        factor: magnification of the plane of the component

    Returns:
        Thickness of the pixels covered by the component and their slices, or
        None if the component is outside the field
    """
    u, v = _pixel_grid(detector)
    pixel = detector.pitch_mm / factor
    xmin, xmax, ymin, ymax = component.bbox()
    cols = np.flatnonzero((u / factor >= xmin - pixel) & (u / factor <= xmax + pixel))
    rows = np.flatnonzero((v / factor >= ymin - pixel) & (v / factor <= ymax + pixel))
    if not cols.size or not rows.size:
        return None
    slices = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    x = u[slices[1]] / factor
    y = v[slices[0]] / factor
    if isinstance(component, Wires):
        yy, xx = np.meshgrid(y, x, indexing="ij")
        return component.average_thickness(xx, yy, pixel / 2), slices
    n = max(int(component.supersampling), 1)
    offsets = ((np.arange(n) + 0.5) / n - 0.5) * pixel
    xs = (x[:, None] + offsets[None, :]).ravel()
    ys = (y[:, None] + offsets[None, :]).ravel()
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    thickness = component.thickness(xx, yy)
    thickness = thickness.reshape(len(y), n, len(x), n).mean(axis=(1, 3))
    return thickness, slices


def attenuation(thickness_by_material: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return the attenuation (line integral) of thicknesses per material."""
    total = None
    for material, thickness in thickness_by_material.items():
        mu, beta = MATERIALS[material]
        value = mu * thickness / (1.0 + beta * thickness)
        total = value if total is None else total + value
    return total


def plane_transmissions(
    scene: Scene, geometry: Geometry, detector: Detector, focal_blur: bool = True
) -> dict[str, np.ndarray]:
    """Return the blurred transmission of each plane of a scene on the detector."""
    factors = magnifications(scene, geometry)
    shape = (detector.rows, detector.cols)
    transmissions = {}
    for plane, factor in factors.items():
        by_material: dict[str, np.ndarray] = {}
        for component in scene.components:
            if component.plane != plane:
                continue
            rendered = render_thickness(component, detector, factor)
            if rendered is None:
                continue
            thickness, slices = rendered
            target = by_material.setdefault(component.material, np.zeros(shape))
            target[slices] += component.sign * thickness
        if not by_material:
            continue
        for thickness in by_material.values():
            np.clip(thickness, 0.0, None, out=thickness)
        transmission = np.exp(-attenuation(by_material))
        ug = geometry.focal_spot_mm * (factor - 1.0) / detector.pitch_mm
        if focal_blur and ug > 0:
            # Gaussian focal spot: its FWHM is the geometric unsharpness
            transmission = ndimage.gaussian_filter(
                transmission, ug / 2.3548, mode="nearest"
            )
        transmissions[plane] = transmission
    return transmissions


@dataclasses.dataclass(frozen=True)
class Exposure:
    """Exposure of an acquisition.

    Args:
        photons_per_pixel: mean detected photons per pixel in the free beam, at
         the center of the field
        frames: number of frames averaged by the detector (reduces noise)
    """

    photons_per_pixel: float = 4000.0
    frames: int = 1


def acquire(
    transmission: np.ndarray | None,
    detector: Detector,
    exposure: Exposure,
    rng: np.random.Generator,
    pattern: Mapping[str, np.ndarray] | None = None,
    beam_on: bool = True,
    noise: bool = True,
) -> np.ndarray:
    """Return a detector frame (digital numbers, unsigned 16-bit integers).

    Args:
        transmission: transmission of the scene (None for a flat frame)
        detector: detector
        exposure: exposure
        rng: random generator of the frame noise
        pattern: fixed pattern of the detector (computed if None)
        beam_on: False for a dark frame
        noise: False for an ideal, noise-free frame
    """
    pattern = detector.fixed_pattern() if pattern is None else pattern
    shape = (detector.rows, detector.cols)
    signal = np.zeros(shape)
    if beam_on:
        u, _v = _pixel_grid(detector)
        half_width = max(abs(u[0]), 1e-12)
        heel = 1.0 + 0.5 * detector.heel * (u / half_width)
        flux = exposure.photons_per_pixel * np.broadcast_to(heel, shape)
        if transmission is not None:
            flux = flux * transmission
        frames = max(int(exposure.frames), 1)
        photons = rng.poisson(flux * frames) / frames if noise else flux
        signal = photons * detector.dn_per_photon * pattern["gain"]
        signal = ndimage.gaussian_filter(signal, detector.blur_px, mode="nearest")
    frame = signal + pattern["offset"]
    if noise:
        frame = frame + rng.normal(0.0, detector.read_noise_dn, shape) / math.sqrt(
            max(int(exposure.frames), 1)
        )
    frame = np.clip(np.round(frame), 0, detector.max_dn)
    frame[pattern["dead"]] = 0
    frame[pattern["hot"]] = detector.max_dn
    return frame.astype(np.uint16)


@dataclasses.dataclass
class Simulation:
    """Frames and ground truth of a simulated acquisition.

    Attributes:
        radiographs: raw radiographs
        darks: dark frames (beam off)
        flats: flat frames (beam on, no object)
        truth: ground truth (geometry, planes, components, detector)
    """

    radiographs: list[np.ndarray]
    darks: list[np.ndarray]
    flats: list[np.ndarray]
    truth: dict[str, Any]


def simulate(
    scene: Scene,
    geometry: Geometry | None = None,
    detector: Detector | None = None,
    exposure: Exposure | None = None,
    radiographs: int = 1,
    darks: int = 0,
    flats: int = 0,
    seed: int = 0,
    noise: bool = True,
) -> Simulation:
    """Simulate an acquisition campaign of a scene.

    Args:
        scene: scene
        geometry: acquisition geometry
        detector: detector model
        exposure: exposure
        radiographs: number of radiographs
        darks: number of dark frames
        flats: number of flat frames
        seed: seed of the frame noise
        noise: False for noise-free frames

    Returns:
        Frames and ground truth
    """
    geometry = geometry or Geometry()
    detector = detector or Detector()
    exposure = exposure or Exposure()
    pattern = detector.fixed_pattern()
    transmissions = plane_transmissions(scene, geometry, detector)
    transmission = np.ones((detector.rows, detector.cols))
    for value in transmissions.values():
        transmission = transmission * value
    rng = np.random.default_rng(seed)

    def frames(count: int, **kwargs) -> list[np.ndarray]:
        return [
            acquire(
                kwargs.get("transmission"),
                detector,
                exposure,
                rng,
                pattern,
                kwargs.get("beam_on", True),
                noise,
            )
            for _index in range(count)
        ]

    factors = magnifications(scene, geometry)
    truth = {
        "scene": scene.name,
        "geometry": {
            "sdd_mm": geometry.sdd_mm,
            "focal_spot_mm": geometry.focal_spot_mm,
            "voltage_kv": geometry.voltage_kv,
            "planes": [
                {"name": name, "sod_mm": sod, "magnification": factors[name]}
                for name, sod in scene.planes.items()
            ],
        },
        "detector": {
            "pitch_mm": detector.pitch_mm,
            "bit_depth": detector.bit_depth,
            "blur_px": detector.blur_px,
            "dead_pixels": int(pattern["dead"].sum()),
            "hot_pixels": int(pattern["hot"].sum()),
        },
        "exposure": {
            "photons_per_pixel": exposure.photons_per_pixel,
            "frames": exposure.frames,
        },
        "components": [component.truth() for component in scene.components],
        "seed": seed,
    }
    return Simulation(
        frames(radiographs, transmission=transmission),
        frames(darks, beam_on=False),
        frames(flats),
        truth,
    )


def defect_mask(detector: Detector) -> np.ndarray:
    """Return the mask of the defective pixels of a detector."""
    pattern = detector.fixed_pattern()
    return pattern["dead"] | pattern["hot"]


# ---------------------------------------------------------------------------
# Scenes
# ---------------------------------------------------------------------------

WELD_PLANE = "Weld"
BALL_PLANE = "Calibration ball"
PIPE_PLANE = "Pipe axis"


def weld_scene(
    sod_mm: float = 800.0, ball_sod_mm: float = 960.0, thickness_mm: float = 8.0
) -> Scene:
    """Return a butt weld with pores, a crack, two IQIs and a calibration ball.

    The plate, its weld and the IQIs lie in the weld plane; the calibration
    ball lies closer to the detector, hence a second magnification.
    """
    components = (
        Box("Plate", WELD_PLANE, "steel", depth=thickness_mm, supersampling=1),
        WeldBead("Weld bead", WELD_PLANE, "steel", y=0.0, width=9.0, height=2.0),
        Sphere("Pore 1", WELD_PLANE, "steel", -1.0, x=-8.0, y=0.5, radius=0.6),
        Sphere("Pore 2", WELD_PLANE, "steel", -1.0, x=-4.0, y=-1.0, radius=0.35),
        Sphere("Pore 3", WELD_PLANE, "steel", -1.0, x=2.5, y=1.2, radius=0.9),
        Crack(
            "Crack",
            WELD_PLANE,
            "steel",
            xa=6.0,
            ya=-1.5,
            xb=12.0,
            yb=0.8,
            width=0.08,
            depth=4.0,
        ),
        wire_iqi("Wire IQI W6FE", WELD_PLANE, -16.0, 6.0, length=8.0),
        duplex_iqi("Duplex IQI", WELD_PLANE, -16.0, -14.0, length=6.0),
        Sphere(BALL_PLANE, BALL_PLANE, "steel", x=17.0, y=-11.0, radius=1.5),
    )
    return Scene(
        "weld",
        {WELD_PLANE: sod_mm, BALL_PLANE: ball_sod_mm},
        components,
        "Butt weld with pores and a crack, wire and duplex IQIs, calibration ball",
    )


def pipe_scene(
    axis_sod_mm: float = 700.0,
    outer_diameter_mm: float = 28.0,
    wall_mm: float = 3.5,
    ball_sod_mm: float = 980.0,
) -> Scene:
    """Return a pipe seen tangentially, with a calibration ball on the detector.

    With the default geometry, the pipe spans 40 mm on the detector: use a
    detector at least 55 mm wide to see the ball beside it.
    """
    outer = outer_diameter_mm / 2
    components = (
        Pipe(
            "Pipe",
            PIPE_PLANE,
            "steel",
            x=0.0,
            outer_radius=outer,
            inner_radius=outer - wall_mm,
        ),
        Sphere(BALL_PLANE, BALL_PLANE, "steel", x=24.0, y=-10.0, radius=1.5),
    )
    return Scene(
        "pipe",
        {PIPE_PLANE: axis_sod_mm, BALL_PLANE: ball_sod_mm},
        components,
        "Pipe seen tangentially, to measure its wall thickness",
    )


def step_wedge_scene(sod_mm: float = 900.0) -> Scene:
    """Return an aluminum step wedge of 2 to 20 mm."""
    steps = tuple(float(value) for value in range(2, 22, 2))
    return Scene(
        "step-wedge",
        {"Step wedge": sod_mm},
        (
            StepWedge(
                "Step wedge",
                "Step wedge",
                "aluminum",
                x0=-22.0,
                step_width=4.4,
                y0=-8.0,
                y1=8.0,
                steps=steps,
            ),
        ),
        "Aluminum step wedge, to calibrate thickness",
    )


def plate_scene(sod_mm: float = 900.0) -> Scene:
    """Return a cast aluminum plate with pores, for detector corrections."""
    components = [Box("Plate", "Plate", "aluminum", depth=20.0, supersampling=1)]
    rng = np.random.default_rng(7)
    for index in range(12):
        components.append(
            Sphere(
                f"Pore {index + 1}",
                "Plate",
                "aluminum",
                -1.0,
                x=float(rng.uniform(-18, 18)),
                y=float(rng.uniform(-14, 14)),
                radius=float(rng.uniform(0.3, 1.5)),
            )
        )
    return Scene(
        "plate",
        {"Plate": sod_mm},
        tuple(components),
        "Cast aluminum plate with pores",
    )


SCENES = {
    "weld": weld_scene,
    "pipe": pipe_scene,
    "step-wedge": step_wedge_scene,
    "plate": plate_scene,
}
