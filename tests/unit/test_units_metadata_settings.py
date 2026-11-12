"""Tests of length units, plugin metadata and preferences."""

from __future__ import annotations

import json

import pytest

from datalab_radiography.core import metadata as md
from datalab_radiography.core import units
from datalab_radiography.workflow.settings import RadiographySettings, SettingsStore


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mm", "mm"),
        ("µm", "µm"),
        ("μm", "µm"),
        ("um", "µm"),
        ("Micron", "µm"),
        ("cm", "cm"),
        ("inch", "in"),
        ('"', "in"),
        ("", None),
        ("pixels", None),
        (None, None),
    ],
)
def test_normalize_unit(text, expected) -> None:
    """Unit names and common aliases are recognized."""
    assert units.normalize_unit(text) == expected


def test_conversions() -> None:
    """Lengths and areas convert consistently."""
    assert units.to_mm(1.5, "cm") == pytest.approx(15.0)
    assert units.from_mm(25.4, "in") == pytest.approx(1.0)
    assert units.convert_length(150.0, "µm", "mm") == pytest.approx(0.15)
    assert units.convert_area(1.0, "cm", "mm") == pytest.approx(100.0)
    assert units.format_length(12.3456, "cm") == "1.235 cm"
    assert units.format_length(float("nan"), "cm") == "-"
    with pytest.raises(ValueError):
        units.unit_mm("px")


def test_metadata_keys_and_json() -> None:
    """Keys are namespaced and structured values carry their schema version."""
    assert md.MAGNIFICATIONS_KEY == "plugin.org.datalab.radiography.magnifications"
    with pytest.raises(ValueError):
        md.metadata_key("Bad Key")
    text = md.dumps({"b": 1, "a": [1.0, 2.0]})
    assert json.loads(text)["schema_version"] == md.SCHEMA_VERSION
    assert md.loads(text) == {"a": [1.0, 2.0], "b": 1}
    assert md.loads(None) is None
    with pytest.raises(md.MetadataSchemaError, match="schema version"):
        md.loads(json.dumps({"schema_version": 99}))
    with pytest.raises(md.MetadataSchemaError):
        md.loads("not json")
    with pytest.raises(md.MetadataSchemaError):
        md.loads("[1, 2]")


def test_settings_defaults_and_validation() -> None:
    """Invalid preferences fall back to their defaults."""
    settings = RadiographySettings()
    assert settings.unit == "cm" and settings.film_convention is False
    bad = RadiographySettings.from_dict(
        {
            "unit": "parsec",
            "low_percentile": 50.0,
            "high_percentile": 10.0,
            "film_convention": "yes",
            "detector_pitch_mm": -1.0,
            "unknown": 1,
        }
    )
    assert bad.unit == "cm"
    assert (bad.low_percentile, bad.high_percentile) == (0.5, 99.5)
    assert bad.film_convention is False
    assert bad.detector_pitch_mm is None
    assert RadiographySettings(unit="um").unit == "µm"


def test_settings_store_roundtrip(tmp_path) -> None:
    """Preferences survive a save and reload; invalid files are ignored."""
    path = tmp_path / "sub" / "prefs.json"
    store = SettingsStore(path)
    store.update(film_convention=True, unit="mm", detector_pitch_mm=0.1496)
    reloaded = SettingsStore(path).settings
    assert reloaded.film_convention is True
    assert reloaded.unit == "mm"
    assert reloaded.detector_pitch_mm == pytest.approx(0.1496)
    path.write_text("[]", encoding="utf-8")
    assert SettingsStore(path).settings == RadiographySettings()
    memory = SettingsStore()
    memory.update(unit="m")
    assert memory.settings.unit == "m"
