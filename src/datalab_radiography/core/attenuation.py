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


@dataclasses.dataclass(frozen=True)
class ThicknessCalibration:
    """Polynomial thickness calibration t = P(A).

    Attributes:
        coefficients: polynomial coefficients, highest degree first (mm)
        attenuations: mean attenuation of each calibration step
        thicknesses: true thickness of each step (mm)
        residuals: fitted minus true thickness of each step (mm)
    """

    coefficients: tuple[float, ...]
    attenuations: tuple[float, ...]
    thicknesses: tuple[float, ...]
    residuals: tuple[float, ...]

    @property
    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.coefficients) - 1

    @property
    def rms_residual(self) -> float:
        """RMS of the thickness residuals (mm)."""
        return float(np.sqrt(np.mean(np.square(self.residuals))))

    def thickness(self, values: np.ndarray) -> np.ndarray:
        """Return the thickness (mm) of attenuation values."""
        return np.polyval(self.coefficients, np.asarray(values, dtype=float))


def calibrate_thickness(
    attenuations: Sequence[float],
    thicknesses: Sequence[float],
    degree: int = 2,
    through_origin: bool = True,
) -> ThicknessCalibration:
    """Fit the thickness as a polynomial of the attenuation.

    Args:
        attenuations: mean attenuation of each step
        thicknesses: true thickness of each step (mm)
        degree: degree of the polynomial
        through_origin: force a zero thickness for a zero attenuation

    Raises:
        ValueError: if there are too few steps for the degree
    """
    a = np.asarray(attenuations, dtype=float)
    t = np.asarray(thicknesses, dtype=float)
    if a.shape != t.shape or a.ndim != 1:
        raise ValueError("Give one attenuation per thickness")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(t)):
        raise ValueError("Attenuations and thicknesses must be finite")
    unknowns = degree if through_origin else degree + 1
    if degree < 1 or a.size < unknowns + 1:
        raise ValueError(
            f"A polynomial of degree {degree} needs at least {unknowns + 1} steps"
        )
    if through_origin:
        powers = np.column_stack([a**power for power in range(degree, 0, -1)])
        solution, *_ = np.linalg.lstsq(powers, t, rcond=None)
        coefficients = np.append(solution, 0.0)
    else:
        coefficients = np.polyfit(a, t, degree)
    residuals = np.polyval(coefficients, a) - t
    return ThicknessCalibration(
        tuple(float(value) for value in coefficients),
        tuple(float(value) for value in a),
        tuple(float(value) for value in t),
        tuple(float(value) for value in residuals),
    )
