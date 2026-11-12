"""Wall thickness of pipes in tangential radiography.

The profile crosses a pipe perpendicularly to its axis, from the free beam on
one side to the free beam on the other side. Two methods are given, inspired by
ISO 20769-1 without claiming conformity:

- edges: outer edges at the steepest rise of the attenuation, inner edges at
  its steepest fall after the maxima reached at the inner wall tangents;
- model: fit of the projected annulus (Beer-Lambert law with beam hardening,
  smooth background, Gaussian blur), which corrects the blur biases.

Lengths are given in the unit of the profile sample spacing (mm in the plane
of the pipe axis).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import ndimage, optimize

from .quality import QualityError


@dataclasses.dataclass(frozen=True)
class WallEdges:
    """Edges of the two walls along a profile (sample indices).

    Attributes:
        outer_left: outer edge of the first wall
        inner_left: inner edge of the first wall
        inner_right: inner edge of the second wall
        outer_right: outer edge of the second wall
    """

    outer_left: float
    inner_left: float
    inner_right: float
    outer_right: float

    def walls(self, step: float) -> tuple[float, float]:
        """Return the thickness of the first and second walls."""
        return (
            (self.inner_left - self.outer_left) * step,
            (self.outer_right - self.inner_right) * step,
        )

    def outer_diameter(self, step: float) -> float:
        """Return the outer diameter."""
        return (self.outer_right - self.outer_left) * step


@dataclasses.dataclass(frozen=True)
class PipeFit:
    """Projected annulus fitted to a profile.

    Attributes:
        center: axis position (sample index)
        outer_radius: outer radius (samples)
        inner_radius: inner radius (samples)
        blur: Gaussian blur, standard deviation (samples)
        rms: RMS residual of the fit
        fitted: fitted profile
    """

    center: float
    outer_radius: float
    inner_radius: float
    blur: float
    rms: float
    fitted: np.ndarray | None = dataclasses.field(
        repr=False, compare=False, default=None
    )

    def wall(self, step: float) -> float:
        """Return the wall thickness."""
        return (self.outer_radius - self.inner_radius) * step


def _subsample_extremum(values: np.ndarray, index: int) -> float:
    """Refine the position of an extremum by a parabola through 3 samples."""
    if 0 < index < values.size - 1:
        a, b, c = values[index - 1 : index + 2]
        denominator = a - 2.0 * b + c
        if denominator != 0:
            return index + float(np.clip(0.5 * (a - c) / denominator, -0.5, 0.5))
    return float(index)


def attenuation_profile(profile: np.ndarray) -> tuple[np.ndarray, bool]:
    """Return a profile positive in the pipe, and True if the pipe is darker.

    The free beam level is a straight baseline fitted to both ends.

    Raises:
        QualityError: if both ends of the profile are not in the free beam
    """
    profile = np.asarray(profile, dtype=float)
    ends = max(profile.size // 20, 3)
    left = float(np.median(profile[:ends]))
    right = float(np.median(profile[-ends:]))
    if abs(left - right) > 0.3 * float(np.ptp(profile)):
        raise QualityError(
            "The profile must cross the whole pipe, with free beam on both sides"
        )
    x = np.arange(profile.size, dtype=float)
    keep = np.zeros(profile.size, dtype=bool)
    keep[:ends] = keep[-ends:] = True
    coef = np.polyfit(x[keep], profile[keep], 1)
    residual = profile - np.polyval(coef, x)
    darker = -np.min(residual) > np.max(residual)
    return (-residual if darker else residual), darker


def find_edges(features: np.ndarray, smoothing: float = 1.0) -> WallEdges:
    """Return the wall edges of a pipe profile, positive in the pipe.

    Raises:
        QualityError: if the profile does not cross a whole pipe
    """
    features = np.asarray(features, dtype=float)
    peak = float(np.max(features))
    if not peak > 0:
        raise QualityError("No pipe found in the profile")
    inside = np.nonzero(features > 0.1 * peak)[0]
    if inside.size < 8 or inside[0] < 2 or inside[-1] > features.size - 3:
        raise QualityError(
            "The profile must cross the whole pipe, with free beam on both sides"
        )
    start, stop = int(inside[0]), int(inside[-1])
    middle = (start + stop) // 2
    gradient = np.gradient(ndimage.gaussian_filter1d(features, smoothing))
    peak_left = start + int(np.argmax(features[start:middle]))
    peak_right = middle + int(np.argmax(features[middle : stop + 1]))
    outer_left = int(np.argmax(gradient[: peak_left + 1]))
    inner_left = peak_left + int(np.argmin(gradient[peak_left : middle + 1]))
    inner_right = middle + int(np.argmax(gradient[middle : peak_right + 1]))
    outer_right = peak_right + int(np.argmin(gradient[peak_right:]))
    edges = WallEdges(
        _subsample_extremum(gradient, outer_left),
        _subsample_extremum(-gradient, inner_left),
        _subsample_extremum(gradient, inner_right),
        _subsample_extremum(-gradient, outer_right),
    )
    if not edges.outer_left < edges.inner_left < edges.inner_right < edges.outer_right:
        raise QualityError("The walls of the pipe could not be told apart")
    return edges


def _chord(u: np.ndarray, radius: float) -> np.ndarray:
    """Return the chord of a circle of a given radius at distance u."""
    return 2.0 * np.sqrt(np.clip(radius * radius - u * u, 0.0, None))


def fit_pipe(profile: np.ndarray, edges: WallEdges | None = None) -> PipeFit:
    """Fit a projected annulus to a profile across a pipe.

    The model attenuates a cubic background along the profile (Beer-Lambert
    law with beam hardening, plus an offset) when the pipe is darker than the
    free beam, or adds a contrast proportional to the attenuation otherwise,
    then blurs it with a Gaussian.

    Args:
        profile: profile across the pipe (not smoothed)
        edges: first estimate of the edges (computed if None)

    Raises:
        QualityError: if the fit fails
    """
    profile = np.asarray(profile, dtype=float)
    features, darker = attenuation_profile(profile)
    if edges is None:
        edges = find_edges(features)
    size = profile.size
    factor = 4
    positions = (np.arange(size * factor) + 0.5) / factor - 0.5
    centered = (positions - 0.5 * (size - 1)) / size
    ends = max(size // 20, 3)
    level = float(np.median(np.r_[profile[:ends], profile[-ends:]]))
    center = 0.5 * (edges.outer_left + edges.outer_right)
    outer = 0.5 * (edges.outer_right - edges.outer_left)
    inner = 0.5 * (edges.inner_right - edges.inner_left)
    chord_max = _chord(np.array([inner]), outer)[0]
    contrast = float(np.max(features))
    if darker:
        transmission = max(1.0 - contrast / max(level, 1e-12), 1e-3)
        mu = -math.log(transmission) / max(chord_max, 1e-12)
    else:
        mu = contrast / max(chord_max, 1e-12)
    start = np.array([center, outer, inner, mu, 0.0, 1.0, level, 0, 0, 0, 0.0])

    def model(params: np.ndarray) -> np.ndarray:
        c, ro, ri, mu_, beta, blur, a0, a1, a2, a3, offset = params
        u = positions - c
        thickness = _chord(u, ro) - _chord(u, min(abs(ri), abs(ro)))
        attenuation = mu_ * thickness / (1.0 + abs(beta) * thickness)
        background = a0 + a1 * centered + a2 * centered**2 + a3 * centered**3
        if darker:
            image = background * np.exp(-attenuation) + offset
        else:
            image = background + attenuation
        image = image.reshape(size, factor).mean(axis=1)
        return ndimage.gaussian_filter1d(image, abs(blur), mode="nearest")

    lower = [-np.inf, 2.0, 1.0, 0.0, 0.0, 0.05, -np.inf, -np.inf, -np.inf, -np.inf]
    upper = [np.inf, np.inf, np.inf, np.inf, 10.0, 50.0, np.inf, np.inf, np.inf, np.inf]
    lower.append(-np.inf if darker else -1e-12)
    upper.append(np.inf if darker else 1e-12)
    try:
        result = optimize.least_squares(
            lambda params: model(params) - profile,
            start,
            bounds=(lower, upper),
            x_scale="jac",
            diff_step=1e-4,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise QualityError(f"The pipe model fit failed: {exc}") from exc
    if not result.success:
        raise QualityError(f"The pipe model fit failed: {result.message}")
    c, ro, ri = result.x[:3]
    ri = min(abs(ri), abs(ro))
    rms = float(np.sqrt(np.mean(result.fun**2)))
    return PipeFit(
        float(c),
        float(abs(ro)),
        float(ri),
        float(abs(result.x[5])),
        rms,
        model(result.x),
    )
