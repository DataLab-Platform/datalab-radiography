"""Checks on the local corpus of real radiographs (skipped without it).

Set ``DATALAB_RADIOGRAPHY_CORPUS`` to the corpus folder (see CONTRIBUTING.md).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pytest
import sigima.io as sio
from datalab.plugins.recipes import RecipeExecutionContext

from datalab_radiography.core import dicom as cdicom
from datalab_radiography.core import magnification as mag
from datalab_radiography.workflow import computations, objects
from datalab_radiography.workflow import correction_recipes as recipes
from datalab_radiography.workflow import corrections as wcorr
from datalab_radiography.workflow import dicom as wdicom
from datalab_radiography.workflow import display as wdisplay

CORPUS = os.environ.get("DATALAB_RADIOGRAPHY_CORPUS", "")
pytestmark = pytest.mark.skipif(
    not CORPUS or not Path(CORPUS).is_dir(),
    reason="DATALAB_RADIOGRAPHY_CORPUS is not set",
)


def corpus_path(*parts: str) -> Path:
    """Return a path of the corpus, skipping the test if it is missing."""
    path = Path(CORPUS, *parts)
    if not path.exists():
        pytest.skip(f"{path} is missing from the corpus")
    return path


def _read_settings(path: Path) -> dict[str, float]:
    """Return the numeric entries of a CWI FleX-ray scan settings file."""
    values = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*([^:]+?)\s*:\s*([-+0-9.eE]+)\s*$", line)
        if match:
            values.setdefault(match[1], float(match[2]))
    return values


def test_cwi_walnut_correction_and_geometry() -> None:
    """CWI walnut: transmission of 1 in the air, magnification of 3.015."""
    folder = corpus_path("cwi-walnuts", "Walnut1", "Projections", "tubeV2")
    settings = _read_settings(folder / "scan settings.txt")
    factor = mag.magnification_from_distances(settings["SDD"], settings["SOD"])
    assert factor == pytest.approx(settings["Magnification"], rel=1e-5)
    raw = sio.read_image(str(folder / "scan_000000.tif"))
    dark = sio.read_image(str(folder / "di000000.tif"))
    flats = [sio.read_image(str(folder / f"io00000{i}.tif")) for i in (0, 1)]
    for image, role in [(raw, "raw"), (dark, "dark")] + [(f, "flat") for f in flats]:
        objects.set_frame_role(image, role)
    inputs = wcorr.suggest_frame_bindings([raw, dark, *flats])
    param = recipes.OffsetGainRecipeParam()
    param.output = "transmission"
    outcome = recipes.run_offset_gain(inputs, param, RecipeExecutionContext())
    transmission = outcome.objects[0].value.data
    # The corners of the projection are in the air
    corners = np.concatenate(
        [
            transmission[:40, :40].ravel(),
            transmission[:40, -40:].ravel(),
            transmission[-40:, :40].ravel(),
            transmission[-40:, -40:].ravel(),
        ]
    )
    # The brightest pixels are in the air; the corners, darkened by the
    # vignetting of the detector, keep a few percent of residual error
    assert np.nanpercentile(transmission, 99) == pytest.approx(1.0, abs=0.01)
    assert np.nanmedian(corners) == pytest.approx(1.0, abs=0.05)
    assert np.nanmin(transmission) > 0.2
    # Magnification of the walnut axis from its geometry, pitch 0.1496 mm
    objects.set_detector_pitch(raw, settings["Binned pixel size"])
    objects.set_geometry(
        raw,
        {
            "sdd_mm": settings["SDD"],
            "planes": [{"name": "Axis", "sod_mm": settings["SOD"]}],
        },
    )
    from datalab_radiography.workflow import magnification as wmag

    item = wmag.define_from_metadata(raw)[0]
    assert item.object_pixel_mm == pytest.approx(
        settings["Voxel size"] / 1000.0, rel=1e-4
    )


def test_keyboard_laminography_free_beam() -> None:
    """LUT circuit board: the air has a near-zero attenuation."""
    folder = corpus_path("lut-keyboard-laminography", "20250612_CL_Keyboard_Data")
    pca = (folder / "20250612_keyboard.pca").read_text(encoding="utf-8")
    free_ray = float(re.search(r"FreeRay=([0-9.]+)", pca)[1])
    fdd = float(re.search(r"FDD=([0-9.]+)", pca)[1])
    fod = float(re.search(r"FOD=([0-9.]+)", pca)[1])
    assert mag.magnification_from_distances(fdd, fod) == pytest.approx(6.952, rel=1e-3)
    image = sio.read_image(str(next(folder.glob("*_00001.tif"))))
    param = wcorr.AttenuationParam()
    param.free_beam, param.i0 = wcorr.I0_VALUE, free_ray
    values = computations.attenuation(image, param).data
    air = np.nanpercentile(values, 2)
    assert abs(air) < 0.1
    assert np.nanpercentile(values, 99) > 0.5


def test_tcia_dicom_attributes() -> None:
    """Real CR and DX files: inversion, pixel pitch and window from attributes."""
    pydicom = pytest.importorskip("pydicom")
    files = sorted(corpus_path("tcia-covid19-ar").rglob("*.dcm"))
    assert len(files) >= 20
    photometrics, pitches = set(), set()
    for path in files:
        image = sio.read_image(str(path))
        header = pydicom.dcmread(path, stop_before_pixels=True)
        attrs = cdicom.attributes({key: header.get(key) for key in cdicom.KEYWORDS})
        assert wdicom.matches(image, attrs), path
        wdicom.apply_attributes(image, attrs)
        profile = wdisplay.apply_profile(image)
        photometric = attrs["PhotometricInterpretation"]
        photometrics.add(photometric)
        assert profile.invert == (photometric == "MONOCHROME1"), path
        spacing = attrs["ImagerPixelSpacing"]
        pitch = objects.get_detector_pitch(image)
        assert pitch == pytest.approx((spacing[1], spacing[0])), path
        pitches.add(round(pitch[0], 3))
        if "WindowCenter" in attrs:
            assert profile.window_source == wdisplay.WINDOW_DICOM
            low, high = profile.window.low, profile.window.high
            assert low < np.max(image.data) and high > np.min(image.data), path
    assert photometrics == {"MONOCHROME1", "MONOCHROME2"}
    assert len(pitches) >= 2
