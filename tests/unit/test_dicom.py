"""Tests of the translation and application of DICOM attributes."""

from __future__ import annotations

import numpy as np
import pytest
from sigima.objects import create_image

from datalab_radiography.core import dicom as cdicom
from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import dicom as wdicom
from datalab_radiography.workflow import display as wdisplay
from datalab_radiography.workflow import objects
from datalab_radiography.workflow.settings import RadiographySettings


class MultiValue(list):
    """Stand-in for pydicom multi-valued elements."""


HEADER = {
    "PatientName": "Doe^John",
    "PatientID": "123",
    "Modality": "DX",
    "PhotometricInterpretation": "MONOCHROME1",
    "Rows": 40,
    "Columns": 50,
    "BitsStored": "12",
    "RescaleSlope": "2",
    "RescaleIntercept": "-100",
    "WindowCenter": MultiValue(["1900", "2500"]),
    "WindowWidth": MultiValue(["1000", "2000"]),
    "ImagerPixelSpacing": MultiValue(["0.20", "0.15"]),
    "PixelSpacing": MultiValue(["0.16", "0.12"]),
    "PixelSpacingCalibrationType": "GEOMETRY",
    "DistanceSourceToDetector": "1000",
    "DistanceSourceToPatient": "800",
    "FocalSpots": MultiValue(["0.6"]),
    "KVP": "120",
    "PixelData": b"\x00\x01",
}


def test_attributes_are_technical_and_plain() -> None:
    """Only technical attributes are kept, as JSON-compatible values."""
    attrs = cdicom.attributes(HEADER)
    assert "PatientName" not in attrs and "PatientID" not in attrs
    assert "PixelData" not in attrs
    assert attrs["BitsStored"] == 12 and attrs["RescaleSlope"] == 2
    assert attrs["WindowCenter"] == 1900 and attrs["WindowWidth"] == 1000
    assert attrs["ImagerPixelSpacing"] == [0.2, 0.15]
    assert attrs["PhotometricInterpretation"] == "MONOCHROME1"
    assert md.loads(md.dumps(attrs)) == attrs
    assert cdicom.attributes({"KVP": "", "Modality": None}) == {}
    assert cdicom.attributes({"KVP": float("nan")}) == {}


def test_calibration_and_geometry() -> None:
    """Spacings are read as (X, Y); calibrated spacings give the object plane."""
    attrs = cdicom.attributes(HEADER)
    calibration = cdicom.calibration(attrs)
    assert calibration.pitch_mm == (0.15, 0.2)
    assert calibration.object_spacing_mm == (0.12, 0.16)
    assert calibration.factor == pytest.approx(1.25)
    assert calibration.calibration_type == "GEOMETRY"
    same = cdicom.calibration(
        {"ImagerPixelSpacing": [0.2, 0.2], "PixelSpacing": [0.2, 0.2]}
    )
    assert same.object_spacing_mm is None and same.factor is None
    ratio = cdicom.calibration(
        {"ImagerPixelSpacing": [0.2, 0.2], "PixelSpacing": [0.1, 0.1]}
    )
    assert ratio.factor == pytest.approx(2.0)
    estimated = cdicom.calibration(
        {"EstimatedRadiographicMagnificationFactor": 1.1, **attrs}
    )
    assert estimated.factor == 1.1
    assert cdicom.calibration({"ImagerPixelSpacing": [0.0, 0.2]}).pitch_mm is None
    assert cdicom.calibration({"ImagerPixelSpacing": "0.2"}).pitch_mm is None
    assert cdicom.geometry(attrs) == {
        "sdd_mm": 1000.0,
        "sod_mm": 800.0,
        "voltage_kv": 120.0,
        "focal_spot_mm": 0.6,
    }
    assert cdicom.is_dicom_header(b"\0" * 128 + b"DICM")
    assert not cdicom.is_dicom_header(b"\0" * 128 + b"TIFF")
    assert not cdicom.is_dicom_header(b"DICM")


def test_apply_attributes_to_an_image() -> None:
    """Attributes give the pitch, the magnification, the window and the inversion."""
    attrs = cdicom.attributes(HEADER)
    image = create_image("dicom", np.arange(2000, dtype=np.uint16).reshape(40, 50))
    assert wdicom.matches(image, attrs)
    assert not wdicom.matches(create_image("other", np.zeros((10, 10))), attrs)
    assert wdicom.matches(image, {"Rows": "40"})
    changes = wdicom.apply_attributes(image, attrs)
    assert changes[0].startswith("detector pixel pitch")
    assert objects.get_detector_pitch(image) == pytest.approx((0.15, 0.2))
    magnifications = objects.get_magnifications(image)
    assert magnifications.active == "DICOM"
    item = magnifications.active_item
    assert item.scale_mm == pytest.approx((0.12, 0.16))
    assert item.factor == pytest.approx(1.25)
    assert item.details["calibration_type"] == "GEOMETRY"
    assert objects.get_geometry(image)["sdd_mm"] == 1000.0
    profile = wdisplay.apply_profile(image)
    assert profile.invert and profile.window_source == wdisplay.WINDOW_DICOM
    assert (image.zscalemin, image.zscalemax) == pytest.approx((750.0, 1249.5))
    objects.set_detector_pitch(image, 0.5)
    wdicom.apply_attributes(image, attrs)
    assert objects.get_detector_pitch(image) == pytest.approx((0.5, 0.5))
    changes = wdicom.apply_attributes(image, attrs, RadiographySettings(), force=True)
    assert objects.get_detector_pitch(image) == pytest.approx((0.15, 0.2))
    assert changes[-1] == "display"
    plain = create_image("plain", np.zeros((40, 50)))
    wdicom.apply_attributes(plain, {"ImagerPixelSpacing": [0.2, 0.2]})
    assert not len(objects.get_magnifications(plain))
    wdicom.apply_attributes(plain, {"EstimatedRadiographicMagnificationFactor": 2})
    assert objects.get_magnifications(plain).active_item.factor == 2.0
