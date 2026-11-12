"""Image quality of radiographs: SNR, CNR and duplex wire IQI.

The methods are inspired by ISO 17636-2 (normalized SNR) and ISO 19232-5
(duplex wire IQI, 20 % dip criterion), without claiming conformity.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
from scipy import signal, stats

#: Reference basic spatial resolution of the normalized SNR (ISO 17636-2), µm
SNR_REFERENCE_UM = 88.6
#: Dip below which a duplex element is unresolved (ISO 19232-5), percent
DUPLEX_DIP_THRESHOLD = 20.0
#: Smallest height of the wires of the first duplex element, in noise units
MIN_FIRST_ELEMENT_SNR = 8.0
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

POLARITY_AUTO = "auto"
POLARITY_DARKER = "darker"
POLARITY_BRIGHTER = "brighter"
POLARITIES = (POLARITY_AUTO, POLARITY_DARKER, POLARITY_BRIGHTER)


class QualityError(ValueError):
    """Raised when an image quality measurement cannot be made."""


# ---------------------------------------------------------------------------
# SNR and CNR
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RegionStatistics:
    """Mean and noise of a homogeneous region.

    Attributes:
        mean: mean value
        std: standard deviation (of the residual of a plane fit, if detrended)
        count: number of pixels
    """

    mean: float
    std: float
    count: int

    @property
    def snr(self) -> float:
        """Signal-to-noise ratio, mean / std."""
        return self.mean / self.std if self.std > 0 else math.inf


def region_statistics(
    values: np.ndarray,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
    detrend: bool = False,
) -> RegionStatistics:
    """Return the mean and noise of the values of a region.

    Args:
        values: pixel values
        rows: row index of each value (required to detrend)
        cols: column index of each value (required to detrend)
        detrend: remove a plane fitted to the values before computing the
         noise, so that slow variations (heel effect, thickness) are not
         counted as noise

    Raises:
        QualityError: if the region has too few finite values
    """
    values = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(values)
    if finite.sum() < 3:
        raise QualityError("The region has fewer than 3 valid pixels")
    mean = float(np.mean(values[finite]))
    residual = values[finite] - mean
    if detrend:
        if rows is None or cols is None:
            raise ValueError("Detrending needs the pixel positions")
        r = np.asarray(rows, dtype=float).ravel()[finite]
        c = np.asarray(cols, dtype=float).ravel()[finite]
        design = np.column_stack([np.ones_like(r), r - r.mean(), c - c.mean()])
        coef, *_ = np.linalg.lstsq(design, values[finite], rcond=None)
        residual = values[finite] - design @ coef
    std = float(np.std(residual, ddof=1))
    return RegionStatistics(mean, std, int(finite.sum()))


def normalized_snr(snr: float, srb_um: float) -> float:
    """Return the SNR normalized to a basic spatial resolution of 88.6 µm."""
    if not srb_um > 0:
        raise QualityError("The basic spatial resolution must be positive")
    return snr * SNR_REFERENCE_UM / srb_um


def contrast_to_noise(
    indication: RegionStatistics, background: RegionStatistics
) -> tuple[float, float]:
    """Return the contrast and the contrast-to-noise ratio of an indication.

    CNR = |mean(indication) - mean(background)| / std(background).
    """
    contrast = abs(indication.mean - background.mean)
    cnr = contrast / background.std if background.std > 0 else math.inf
    return contrast, cnr


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


def robust_profile(region: np.ndarray, axis: int) -> np.ndarray:
    """Average a region across ``axis`` with a 10 % trimmed mean.

    The trimmed mean ignores defective pixels and direct photon hits.
    """
    return stats.trim_mean(np.asarray(region, dtype=float), 0.1, axis=axis)


def noise_level(profile: np.ndarray) -> float:
    """Return a robust estimate of the noise of a profile (from differences)."""
    diff = np.diff(np.asarray(profile, dtype=float))
    if diff.size == 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(diff - np.median(diff))) / math.sqrt(2))


def baseline(profile: np.ndarray, iterations: int = 20) -> np.ndarray:
    """Return a straight baseline fitted to the background of a profile.

    Points deviating from the line by more than 2.5 robust standard
    deviations (features) are iteratively excluded from the fit.
    """
    profile = np.asarray(profile, dtype=float)
    x = np.arange(profile.size, dtype=float)
    keep = np.isfinite(profile)
    coef = np.polyfit(x[keep], profile[keep], 1)
    for _index in range(iterations):
        residual = profile - np.polyval(coef, x)
        spread = 1.4826 * np.median(np.abs(residual[keep] - np.median(residual[keep])))
        spread = max(spread, noise_level(profile), 1e-12)
        new_keep = np.isfinite(profile) & (np.abs(residual) < 2.5 * spread)
        if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        coef = np.polyfit(x[keep], profile[keep], 1)
    return np.polyval(coef, x)


def feature_profile(
    profile: np.ndarray, polarity: str = POLARITY_AUTO
) -> tuple[np.ndarray, bool]:
    """Return the profile of features above its baseline, positive.

    Args:
        profile: profile
        polarity: ``darker`` (wires in a radiograph), ``brighter`` (wires in
         an attenuation image) or ``auto``

    Returns:
        Positive feature profile, and True if the features are darker
    """
    if polarity not in POLARITIES:
        raise ValueError(f"Unknown polarity {polarity!r}")
    residual = np.asarray(profile, dtype=float) - baseline(profile)
    darker = polarity == POLARITY_DARKER
    if polarity == POLARITY_AUTO:
        darker = -np.nanmin(residual) > np.nanmax(residual)
    return (-residual if darker else residual), darker


# ---------------------------------------------------------------------------
# Duplex wire IQI
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DuplexElement:
    """Measurement of one duplex element.

    Attributes:
        number: element number (1 for D1)
        diameter_mm: nominal wire diameter
        dip: dip between the two wires, percent (0 if they are merged)
        first: profile index of the first wire (or of the merged peak)
        second: profile index of the second wire (None if merged)
    """

    number: int
    diameter_mm: float
    dip: float
    first: float
    second: float | None

    @property
    def resolved(self) -> bool:
        """True if the dip reaches the 20 % criterion."""
        return self.dip >= DUPLEX_DIP_THRESHOLD


@dataclasses.dataclass(frozen=True)
class DuplexResult:
    """Basic spatial resolution measured with a duplex wire IQI.

    Attributes:
        elements: measured elements, up to the first unresolved one
        reversed: True if the profile was read from its end (D1 last)
        isrb: number of the first unresolved element, or None
        srb_mm: interpolated wire diameter at a 20 % dip (basic spatial
         resolution, half the image unsharpness) in the IQI plane, or None
        noise: noise level of the feature profile
    """

    elements: tuple[DuplexElement, ...]
    reversed: bool
    isrb: int | None
    srb_mm: float | None
    noise: float


def element_dip(features: np.ndarray, first: int, second: int) -> float:
    """Return the dip between two wire peaks, percent: (A + B - 2C) / (A + B)."""
    a, b = features[first], features[second]
    c = float(np.min(features[min(first, second) : max(first, second) + 1]))
    total = a + b
    if total <= 0:
        return 0.0
    return float(np.clip(100.0 * (a + b - 2.0 * c) / total, 0.0, 100.0))


def _read_elements(
    features: np.ndarray, peaks: np.ndarray, step_mm: float, first_element: int
) -> list[DuplexElement]:
    """Assign the peaks of a feature profile to the duplex elements, in order."""
    positions = peaks * step_mm
    elements = []
    cursor = -math.inf
    for number in range(first_element, len(DUPLEX_DIAMETERS) + 1):
        diameter = DUPLEX_DIAMETERS[number - 1]
        spacing = 2.0 * diameter
        candidates = np.nonzero(positions > cursor)[0]
        if candidates.size == 0:
            break
        first = int(candidates[0])
        second = None
        for index in candidates[1:]:
            separation = positions[index] - positions[first]
            if separation > 1.5 * spacing:
                break
            if abs(separation - spacing) <= 0.45 * spacing:
                second = int(index)
                break
        if second is None:
            elements.append(
                DuplexElement(number, diameter, 0.0, float(peaks[first]), None)
            )
            break
        dip = element_dip(features, int(peaks[first]), int(peaks[second]))
        elements.append(
            DuplexElement(
                number, diameter, dip, float(peaks[first]), float(peaks[second])
            )
        )
        if dip < DUPLEX_DIP_THRESHOLD:
            break
        cursor = positions[second] + 0.5 * diameter
    return elements


def interpolate_srb(elements: list[DuplexElement]) -> tuple[int | None, float | None]:
    """Return the first unresolved element and the wire diameter at a 20 % dip.

    The diameter is interpolated geometrically between the last resolved and
    the first unresolved elements, linearly in dip.
    """
    for previous, element in zip(elements, elements[1:]):
        if previous.resolved and not element.resolved:
            fraction = (previous.dip - DUPLEX_DIP_THRESHOLD) / (
                previous.dip - element.dip
            )
            ratio = element.diameter_mm / previous.diameter_mm
            return element.number, previous.diameter_mm * ratio**fraction
    if elements and not elements[0].resolved:
        return elements[0].number, None
    return None, None


def analyze_duplex(
    profile: np.ndarray,
    step_mm: float,
    first_element: int = 1,
    polarity: str = POLARITY_AUTO,
    prominence: float = 4.0,
) -> DuplexResult:
    """Measure the basic spatial resolution from a profile across a duplex IQI.

    The profile crosses the wires of the IQI; its sample spacing is given in
    the plane of the IQI, where the nominal wire diameters apply. Peaks are
    assigned to the elements in order, using the nominal spacing of the two
    wires of each element (twice their diameter). The profile may be read in
    either direction: the direction matching more elements is kept.

    Args:
        profile: profile across the IQI wires
        step_mm: sample spacing of the profile, in mm in the IQI plane
        first_element: number of the first element crossed by the profile
        polarity: polarity of the wires (see :func:`feature_profile`)
        prominence: minimum peak prominence, in noise standard deviations

    Raises:
        QualityError: if no element is found
    """
    if not step_mm > 0:
        raise ValueError("The profile step must be positive")
    if not 1 <= first_element <= len(DUPLEX_DIAMETERS):
        raise ValueError("The first element must be between 1 and 13")
    profile = np.asarray(profile, dtype=float)
    if profile.size < 8 or not np.isfinite(profile).all():
        raise QualityError("The IQI profile is too short or has invalid values")
    features, _darker = feature_profile(profile, polarity)
    noise = noise_level(features)
    threshold = max(prominence * noise, 1e-9 * float(np.max(np.abs(features))))
    best = None
    for reverse in (False, True):
        values = features[::-1] if reverse else features
        peaks, _props = signal.find_peaks(values, prominence=threshold)
        elements = _read_elements(values, peaks, step_mm, first_element)
        if reverse:
            last = profile.size - 1
            elements = [
                dataclasses.replace(
                    item,
                    first=last - item.first,
                    second=None if item.second is None else last - item.second,
                )
                for item in elements
            ]
        score = sum(item.second is not None for item in elements)
        if elements and elements[0].second is not None:
            first = elements[0]
            heights = features[[int(first.first), int(first.second)]]
            # The wires of the first element stand far above the noise
            if np.min(heights) < MIN_FIRST_ELEMENT_SNR * noise:
                score = 0
        if best is None or score > best[0]:
            best = (score, reverse, elements)
    score, reverse, elements = best
    if score == 0:
        raise QualityError(
            "No duplex IQI element found: the profile must cross the wire pairs, "
            "and the magnification must be the one of the IQI plane"
        )
    isrb, srb = interpolate_srb(elements)
    return DuplexResult(tuple(elements), reverse, isrb, srb, noise)
