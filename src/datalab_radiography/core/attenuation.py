"""Physical quantities: attenuation, equivalent thickness, step wedge calibration.

The attenuation of a ray is A = -ln(I / I0), where I0 is the intensity of the
free beam. With a monochromatic approximation, A = mu t, so that a material of
known linear attenuation coefficient mu gives the equivalent thickness t. With
a polychromatic spectrum, the beam hardens through the matter and A grows
slower than t: a step wedge of known thicknesses calibrates t as a polynomial
function of A.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np


def attenuation(intensity: np.ndarray, free_beam: np.ndarray | float) -> np.ndarray:
    """Return the attenuation -ln(I / I0).

    Non-positive intensities give NaN.

    Args:
        intensity: intensity I (offset corrected)
        free_beam: free-beam intensity I0 (value or image)

    Raises:
        ValueError: if the free-beam intensity is not positive
    """
    intensity = np.asarray(intensity, dtype=float)
    free_beam = np.asarray(free_beam, dtype=float)
    if np.any(free_beam <= 0) or not np.all(np.isfinite(free_beam)):
        raise ValueError("The free-beam intensity must be strictly positive")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(intensity > 0, -np.log(intensity / free_beam), np.nan)


def equivalent_thickness(values: np.ndarray, mu_per_mm: float) -> np.ndarray:
    """Return the thickness (mm) of a material giving an attenuation.

    Args:
        values: attenuation image
        mu_per_mm: linear attenuation coefficient of the material (1/mm)

    Raises:
        ValueError: if the coefficient is not positive
    """
    if not mu_per_mm > 0:
        raise ValueError("The attenuation coefficient must be strictly positive")
    return np.asarray(values, dtype=float) / mu_per_mm


#: Smallest attenuation difference between steps of different thicknesses,
#: relative to the largest attenuation of the steps
MIN_STEP_SEPARATION = 1e-3
#: Margin of the calibrated range, relative to its span, within which
#: attenuations are not considered extrapolated (noise of the free beam)
EXTRAPOLATION_MARGIN = 0.05
#: Number of attenuations at which the slope of the calibration is checked
MONOTONICITY_SAMPLES = 1001


@dataclasses.dataclass(frozen=True)
class ThicknessCalibration:
    """Polynomial thickness calibration t = P(A).

    Attributes:
        coefficients: polynomial coefficients, highest degree first (mm)
        attenuations: mean attenuation of each calibration step
        thicknesses: true thickness of each step (mm)
        residuals: fitted minus true thickness of each step (mm)
        through_origin: whether a zero attenuation means zero thickness
    """

    coefficients: tuple[float, ...]
    attenuations: tuple[float, ...]
    thicknesses: tuple[float, ...]
    residuals: tuple[float, ...]
    through_origin: bool = True

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.coefficients) - 1

    @property
    def rms_residual(self) -> float:
        """RMS of the thickness residuals (mm)."""
        return float(np.sqrt(np.mean(np.square(self.residuals))))

    @property
    def domain(self) -> tuple[float, float]:
        """Calibrated attenuation range (from zero when through the origin)."""
        low = min(self.attenuations)
        if self.through_origin:
            low = min(low, 0.0)
        return low, max(self.attenuations)

    def thickness(self, values: np.ndarray) -> np.ndarray:
        """Return the thickness (mm) of attenuation values."""
        return np.polyval(self.coefficients, np.asarray(values, dtype=float))

    def extrapolated_fraction(self, values: np.ndarray) -> float:
        """Return the fraction of finite attenuations outside the calibrated range.

        Attenuations within a margin of the range (5 % of its span) are not
        counted, so that the noise of the free beam is not reported.
        """
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if not values.size:
            return 0.0
        low, high = self.domain
        margin = EXTRAPOLATION_MARGIN * (high - low)
        return float(np.mean((values < low - margin) | (values > high + margin)))


def _check_steps(a: np.ndarray, t: np.ndarray, through_origin: bool) -> None:
    """Check that the attenuation increases with the thickness of the steps.

    Steps of equal thickness are averaged. With ``through_origin``, a zero
    thickness has a zero attenuation.

    Raises:
        ValueError: if the attenuation does not increase with the thickness
    """
    levels = np.unique(t)
    means = np.array([a[t == level].mean() for level in levels])
    if through_origin and levels[0] > 0:
        levels = np.insert(levels, 0, 0.0)
        means = np.insert(means, 0, 0.0)
    tolerance = MIN_STEP_SEPARATION * max(float(np.max(np.abs(a))), 1e-12)
    for index in np.flatnonzero(np.diff(means) <= tolerance):
        t1, t2 = levels[index], levels[index + 1]
        a1, a2 = means[index], means[index + 1]
        raise ValueError(
            f"The attenuation does not increase with the thickness between "
            f"{t1:g} mm ({a1:.4g}) and {t2:g} mm ({a2:.4g}): check the order of "
            "the step regions and the thicknesses (steps beyond the dynamic "
            "range of the detector give the same attenuation)"
        )


def _check_increasing(coefficients: np.ndarray, low: float, high: float) -> None:
    """Check that the calibration curve increases over the calibrated range.

    Raises:
        ValueError: if the slope of the curve is not positive somewhere
    """
    slope = np.polyder(coefficients)
    values = np.linspace(low, high, MONOTONICITY_SAMPLES)
    bad = np.flatnonzero(np.polyval(slope, values) <= 0)
    if bad.size:
        raise ValueError(
            "The calibration curve does not increase with the attenuation "
            f"(near {values[bad[0]]:.4g}, calibrated range {low:.4g} to "
            f"{high:.4g}): lower the polynomial degree or add steps"
        )


def calibrate_thickness(
    attenuations: Sequence[float],
    thicknesses: Sequence[float],
    degree: int = 2,
    through_origin: bool = True,
) -> ThicknessCalibration:
    """Fit the thickness as a polynomial of the attenuation.

    The calibration is refused when it would be degenerate or ambiguous: the
    attenuation must increase with the thickness of the steps, and the fitted
    curve must increase over the calibrated range.

    Args:
        attenuations: mean attenuation of each step
        thicknesses: true thickness of each step (mm)
        degree: degree of the polynomial
        through_origin: force a zero thickness for a zero attenuation

    Raises:
        ValueError: if there are too few steps of distinct thicknesses for the
         degree, if the attenuation does not increase with the thickness, or if
         the calibration curve does not increase over the calibrated range
    """
    a = np.asarray(attenuations, dtype=float)
    t = np.asarray(thicknesses, dtype=float)
    if a.shape != t.shape or a.ndim != 1 or not a.size:
        raise ValueError("Give one attenuation per thickness")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(t)):
        raise ValueError("Attenuations and thicknesses must be finite")
    unknowns = degree if through_origin else degree + 1
    if degree < 1 or np.unique(t).size < unknowns + 1:
        raise ValueError(
            f"A polynomial of degree {degree} needs at least {unknowns + 1} steps "
            "of different thicknesses"
        )
    _check_steps(a, t, through_origin)
    if through_origin:
        powers = np.column_stack([a**power for power in range(degree, 0, -1)])
        solution, *_ = np.linalg.lstsq(powers, t, rcond=None)
        coefficients = np.append(solution, 0.0)
    else:
        coefficients = np.polyfit(a, t, degree)
    calibration = ThicknessCalibration(
        tuple(float(value) for value in coefficients),
        tuple(float(value) for value in a),
        tuple(float(value) for value in t),
        tuple(float(value) for value in np.polyval(coefficients, a) - t),
        through_origin,
    )
    _check_increasing(coefficients, *calibration.domain)
    return calibration
