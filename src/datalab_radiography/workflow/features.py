"""Host-neutral catalog of the processing features of the plugin.

Each adapter registers these features with its host: DataLab Desktop in
``register_computations()`` (with ``feature_id`` and ``owner_plugin_id``),
DataLab-Web in ``create_actions()`` (with a ``menu_path``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import guidata.dataset as gds

from .. import PLUGIN_ID, PLUGIN_NAME

HOST_DESKTOP = "desktop"
HOST_WEB = "web"
HOSTS = frozenset({HOST_DESKTOP, HOST_WEB})

PATTERN_1_TO_1 = "1_to_1"
PATTERN_1_TO_0 = "1_to_0"
PATTERN_2_TO_1 = "2_to_1"
PATTERN_N_TO_1 = "n_to_1"
PATTERNS = (PATTERN_1_TO_1, PATTERN_1_TO_0, PATTERN_2_TO_1, PATTERN_N_TO_1)

GROUP_DISPLAY = "Display"
GROUP_MAGNIFICATION = "Magnification"
GROUP_MEASUREMENTS = "Measurements"
GROUP_CORRECTIONS = "Detector corrections"
GROUP_PHYSICS = "Physical quantities"
GROUP_ENHANCEMENT = "Enhancement"
GROUP_IMAGE_QUALITY = "Image quality"
GROUP_NDT = "NDT measurements"
GROUPS = (
    GROUP_DISPLAY,
    GROUP_MAGNIFICATION,
    GROUP_MEASUREMENTS,
    GROUP_CORRECTIONS,
    GROUP_PHYSICS,
    GROUP_ENHANCEMENT,
    GROUP_IMAGE_QUALITY,
    GROUP_NDT,
)


@dataclasses.dataclass(frozen=True)
class FeatureSpec:
    """Processing feature registered by the plugin in each host.

    Args:
        local_id: identifier local to the plugin
        title: menu title
        group: menu group (one of :data:`GROUPS`)
        pattern: processing pattern (one of :data:`PATTERNS`)
        function: Sigima computation function
        paramclass: parameter class, if any
        hosts: hosts where the feature is available
        preview: allow the live preview of the parameters (Desktop)
        obj2_name: title of the second operand (2-to-1 features)
        comment: tooltip
    """

    local_id: str
    title: str
    group: str
    pattern: str
    function: Callable
    paramclass: type[gds.DataSet] | None = None
    hosts: frozenset[str] = HOSTS
    preview: bool = True
    obj2_name: str | None = None
    comment: str | None = None

    def __post_init__(self) -> None:
        """Validate the feature."""
        if self.group not in GROUPS:
            raise ValueError(f"Unknown feature group {self.group!r}")
        if self.pattern not in PATTERNS:
            raise ValueError(f"Unknown feature pattern {self.pattern!r}")
        hosts = frozenset(self.hosts)
        if not hosts or not hosts <= HOSTS:
            raise ValueError(f"Invalid feature hosts {sorted(hosts)!r}")
        object.__setattr__(self, "hosts", hosts)

    @property
    def feature_id(self) -> str:
        """Stable, namespaced feature identifier."""
        return f"{PLUGIN_ID}:{self.local_id}"

    @property
    def menu_path(self) -> str:
        """Menu path of the feature in DataLab-Web."""
        return f"Plugins/{PLUGIN_NAME}/{self.group}/{self.title}"


#: Features of the plugin, in menu order (filled below)
FEATURES: tuple[FeatureSpec, ...] = ()


def features_for(host: str) -> tuple[FeatureSpec, ...]:
    """Return the features available in a host, in menu order."""
    if host not in HOSTS:
        raise ValueError(f"Unknown host {host!r}")
    return tuple(feature for feature in FEATURES if host in feature.hosts)


def check_catalog(features: tuple[FeatureSpec, ...] = ()) -> None:
    """Check that feature identifiers are unique.

    Raises:
        ValueError: if two features share an identifier
    """
    seen: set[str] = set()
    for feature in features:
        if feature.local_id in seen:
            raise ValueError(f"Duplicate feature {feature.local_id!r}")
        seen.add(feature.local_id)


def _catalog() -> tuple[FeatureSpec, ...]:
    """Return the features of the plugin, in menu order."""
    # pylint: disable=import-outside-toplevel
    from . import computations as comp
    from . import corrections as wcorr
    from . import magnification as wmag
    from . import measurements as wmeas

    return (
        FeatureSpec(
            "express-coordinates",
            "Express coordinates in a plane",
            GROUP_MAGNIFICATION,
            PATTERN_1_TO_1,
            comp.express_coordinates,
            wmag.ExpressCoordinatesParam,
            preview=False,
            comment="Coordinates in an object plane, the detector plane or pixels",
        ),
        FeatureSpec(
            "measurement-report",
            "Measurement report",
            GROUP_MEASUREMENTS,
            PATTERN_1_TO_0,
            comp.measurement_report,
            wmeas.MeasurementReportParam,
            comment="Measure annotations and ROIs in every plane",
        ),
        FeatureSpec(
            "defective-pixel-map",
            "Defective pixel map",
            GROUP_CORRECTIONS,
            PATTERN_1_TO_1,
            comp.defective_pixel_map,
            wcorr.DefectivePixelParam,
            comment="Pixels deviating from their local median",
        ),
        FeatureSpec(
            "correct-defective-pixels",
            "Correct defective pixels",
            GROUP_CORRECTIONS,
            PATTERN_1_TO_1,
            comp.correct_defective_pixels,
            wcorr.DefectivePixelParam,
            comment="Replace defective pixels by the median of their neighbors",
        ),
        FeatureSpec(
            "attenuation",
            "Attenuation",
            GROUP_PHYSICS,
            PATTERN_1_TO_1,
            comp.attenuation,
            wcorr.AttenuationParam,
            comment="-ln(I / I0), I0 from the air, a value or the image",
        ),
        FeatureSpec(
            "attenuation-with-flat",
            "Attenuation with a flat image",
            GROUP_PHYSICS,
            PATTERN_2_TO_1,
            comp.attenuation_with_flat,
            wcorr.FlatAttenuationParam,
            obj2_name="flat image",
            comment="-ln(I / F), F acquired without object",
        ),
        FeatureSpec(
            "equivalent-thickness",
            "Equivalent thickness",
            GROUP_PHYSICS,
            PATTERN_1_TO_1,
            comp.equivalent_thickness,
            wcorr.ThicknessParam,
            comment="Thickness of a material from an attenuation image",
        ),
    )


FEATURES = _catalog()
check_catalog(FEATURES)
