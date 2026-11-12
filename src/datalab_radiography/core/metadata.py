"""Namespaced metadata conventions of the radiography plugin.

Structured values (magnifications, coordinate frame, geometry...) are stored
as versioned JSON strings: they survive DataLab HDF5 workspaces unchanged and
remain readable by other tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .. import PLUGIN_ID

METADATA_PREFIX = f"plugin.{PLUGIN_ID}"
_LOCAL_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*$")


def metadata_key(local_key: str) -> str:
    """Return a stable, plugin-namespaced metadata key.

    Args:
        local_key: lowercase key local to the plugin

    Returns:
        Fully namespaced metadata key

    Raises:
        ValueError: if the local key is not stable and machine-readable
    """
    if not isinstance(local_key, str) or not _LOCAL_KEY_PATTERN.fullmatch(local_key):
        raise ValueError(
            "Radiography metadata keys must contain lowercase letters, digits, "
            "'.', '_' or '-'"
        )
    return f"{METADATA_PREFIX}.{local_key}"


#: Magnifications defined on an image (JSON, see :mod:`.magnification`)
MAGNIFICATIONS_KEY = metadata_key("magnifications")
#: Coordinate frame of the image coordinates (JSON, see :mod:`.magnification`)
COORDINATE_FRAME_KEY = metadata_key("coordinate_frame")
#: Detector pixel pitch in millimeters, along X and Y (JSON list)
DETECTOR_PITCH_KEY = metadata_key("detector_pitch_mm")
#: Acquisition geometry (JSON object: distances in mm, focal spot, voltage...)
GEOMETRY_KEY = metadata_key("geometry")
#: Radiographic display profile applied to the image (JSON object)
DISPLAY_PROFILE_KEY = metadata_key("display_profile")
#: Subset of the DICOM attributes of the image (JSON object)
DICOM_KEY = metadata_key("dicom")
#: Ground truth of a simulated image (JSON object)
SIMULATION_TRUTH_KEY = metadata_key("simulation_truth")
#: Dataset, license and DOI of a real example image (JSON object)
ATTRIBUTION_KEY = metadata_key("attribution")
#: Role of an image in a correction campaign: "raw", "dark" or "flat"
FRAME_ROLE_KEY = metadata_key("frame_role")
#: Role of an output object of a recipe
OUTPUT_ROLE_KEY = metadata_key("output_role")

SCHEMA_VERSION = 1
FRAME_ROLES = ("raw", "dark", "flat")


class MetadataSchemaError(ValueError):
    """Raised when plugin metadata cannot be read."""


def dumps(payload: dict[str, Any]) -> str:
    """Serialize a structured value with its schema version.

    Args:
        payload: JSON-compatible mapping (without ``schema_version``)

    Returns:
        Compact, deterministic JSON text
    """
    data = {"schema_version": SCHEMA_VERSION, **payload}
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads(text: str | dict | None) -> dict[str, Any] | None:
    """Read a structured value written by :func:`dumps`.

    Args:
        text: JSON text, an already decoded mapping, or None

    Returns:
        The mapping without its ``schema_version``, or None if ``text`` is None

    Raises:
        MetadataSchemaError: if the value is not valid JSON or has an
         unsupported schema version
    """
    if text is None:
        return None
    if isinstance(text, dict):
        data = dict(text)
    else:
        try:
            data = json.loads(text)
        except (TypeError, ValueError) as exc:
            raise MetadataSchemaError(f"Invalid plugin metadata: {exc}") from exc
    if not isinstance(data, dict):
        raise MetadataSchemaError("Plugin metadata must be a JSON object")
    version = data.pop("schema_version", None)
    if version != SCHEMA_VERSION:
        raise MetadataSchemaError(
            f"Unsupported plugin metadata schema version {version!r} "
            f"(expected {SCHEMA_VERSION}): update the plugin"
        )
    return data
