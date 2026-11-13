"""Build the real-data examples of the plugin from the local test corpus.

Usage::

    python scripts/build_examples.py [CORPUS_FOLDER]

The corpus folder defaults to the ``DATALAB_RADIOGRAPHY_CORPUS`` environment
variable (see CONTRIBUTING.md). Each example is a DataLab workspace of less
than 1 MiB, written to ``src/datalab_radiography/examples``, whose images carry
their detector pixel pitch, acquisition geometry, magnification, frame role,
radiographic display and attribution (dataset, authors, license, DOI).
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from collections.abc import Callable
from pathlib import Path

import numpy as np
import scipy.io
import sigima.io as sio
from datalab.config import Conf
from datalab.env import execenv
from datalab.tests import datalab_test_app_context
from sigima.objects import ImageObj, create_image

from datalab_radiography.core import metadata as md
from datalab_radiography.demo import REAL_EXAMPLE_FILES
from datalab_radiography.workflow import display as wdisplay
from datalab_radiography.workflow import magnification as wmag
from datalab_radiography.workflow import objects

OUTPUT_DIR = Path(__file__).parents[1] / "src" / "datalab_radiography" / "examples"
MAX_SIZE = 1024 * 1024
#: Namespace of the stable UUIDs of the example images
UUID_NAMESPACE = uuid.UUID("4f6d2b1e-6c0a-4d8e-9a57-7b3a0f3c2e11")
CC_BY = "CC BY 4.0"


def binned(data: np.ndarray, factor: int = 2) -> np.ndarray:
    """Return the mean of factor × factor pixel blocks (the edges are cropped)."""
    rows, cols = (size - size % factor for size in data.shape)
    blocks = np.asarray(data, dtype=np.float64)[:rows, :cols]
    return blocks.reshape(rows // factor, factor, cols // factor, factor).mean((1, 3))


def as_uint16(data: np.ndarray) -> np.ndarray:
    """Round binned detector values back to unsigned 16-bit integers."""
    return np.clip(np.round(data), 0, 65535).astype(np.uint16)


def radiograph(
    example_id: str,
    title: str,
    data: np.ndarray,
    pitch_mm: float,
    geometry: dict,
    attribution: dict,
    role: str = "raw",
) -> ImageObj:
    """Return an example image with its plugin metadata."""
    image = create_image(title, data, units=("", "", "DN"))
    image.metadata["__uuid"] = str(uuid.uuid5(UUID_NAMESPACE, f"{example_id}/{title}"))
    objects.set_detector_pitch(image, pitch_mm)
    objects.set_frame_role(image, role)
    objects.set_geometry(image, geometry)
    image.metadata[md.ATTRIBUTION_KEY] = md.dumps(attribution)
    if role == "raw":
        wmag.define_from_metadata(image)
    wdisplay.apply_profile(image)
    return image


def _settings(path: Path) -> dict[str, float]:
    """Return the numeric entries of a ``key : value`` or ``key=value`` file."""
    values: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^\s*([^:=]+?)\s*[:=]\s*([-+0-9.eE]+)\s*$", line)
        if match:
            try:
                values.setdefault(match[1], float(match[2]))
            except ValueError:
                continue
    return values


def walnut(corpus: Path) -> list[ImageObj]:
    """Raw flat-panel projection of a walnut, with its dark and flat images."""
    folder = corpus / "cwi-walnuts" / "Walnut1" / "Projections" / "tubeV2"
    settings = _settings(folder / "scan settings.txt")
    # 0.1496 mm is the pitch of the detector binned 2 × 2; binned again here
    pitch = 0.1496 * 2
    attribution = {
        "dataset": "A Cone-Beam X-Ray CT Data Collection Designed for Machine "
        "Learning (Walnut 1), CWI FleX-ray Lab",
        "authors": "H. Der Sarkissian, F. Lucka, M. van Eijnatten, G. Colacicco, "
        "S. B. Coban, K. J. Batenburg",
        "reference": "Sci. Data 6, 215 (2019)",
        "doi": "10.5281/zenodo.2686726",
        "license": CC_BY,
        "changes": "First projection of source position 2, first dark and flat "
        "images, binned 2 × 2 and cropped",
    }
    geometry = {
        "sdd_mm": settings["SDD"],
        "voltage_kv": settings["Tube voltage"],
        "planes": [{"name": "Rotation axis", "sod_mm": settings["SOD"]}],
    }
    crop = (slice(8, 376), slice(12, 460))
    images = []
    for title, name, role in (
        ("Walnut", "scan_000000.tif", "raw"),
        ("Walnut dark", "di000000.tif", "dark"),
        ("Walnut flat", "io000000.tif", "flat"),
    ):
        data = as_uint16(binned(sio.read_image(str(folder / name)).data)[crop])
        images.append(
            radiograph("walnut", title, data, pitch, geometry, attribution, role)
        )
    return images


def circuit_board(corpus: Path) -> list[ImageObj]:
    """Projection of a circuit board tilted by 45° (laminography)."""
    folder = corpus / "lut-keyboard-laminography" / "20250612_CL_Keyboard_Data"
    settings = _settings(folder / "20250612_keyboard.pca")
    source = next(folder.glob("*_00001.tif"))
    attribution = {
        "dataset": "Cone-Beam Computed Laminography Dataset of a Keyboard Circuit "
        "Board, LUT University",
        "doi": "10.5281/zenodo.17084482",
        "license": CC_BY,
        "changes": "First projection, binned 2 × 2",
        "free_beam_dn": settings["FreeRay"],
        "notes": "The board is tilted by 45° (laminography): lengths along the "
        "tilt are shortened by cos 45° in the plane of the rotation axis",
    }
    geometry = {
        "sdd_mm": settings["FDD"],
        "voltage_kv": settings["Voltage"],
        "planes": [{"name": "Rotation axis", "sod_mm": settings["FOD"]}],
    }
    data = as_uint16(binned(sio.read_image(str(source)).data))
    pitch = settings["PixelsizeX"] * 2
    return [
        radiograph("circuit-board", "Circuit board", data, pitch, geometry, attribution)
    ]


def knee_phantom(corpus: Path) -> list[ImageObj]:
    """Cone-beam CT projection of a knee phantom."""
    path = next((corpus / "uef-knee-phantom-cbct").glob("*.mat"))
    names = ["proj", "flatValue", "dPitch", "sourceToCRot", "sourceToDetector"]
    mat = scipy.io.loadmat(path, variable_names=names)
    attribution = {
        "dataset": "Cone beam computed tomography dataset of a knee phantom "
        "(Planmeca Viso G7, 100 kV, 80 mAs)",
        "doi": "10.5281/zenodo.12722386",
        "license": CC_BY,
        "changes": "First projection, binned 2 × 2",
        "free_beam": float(np.ravel(mat["flatValue"])[0]),
    }
    sod = float(np.ravel(mat["sourceToCRot"])[0])
    geometry = {
        "sdd_mm": float(np.ravel(mat["sourceToDetector"])[0]),
        "voltage_kv": 100.0,
        "planes": [{"name": "Rotation axis", "sod_mm": sod}],
    }
    data = binned(mat["proj"][:, :, 0]).astype(np.float32)
    pitch = float(np.ravel(mat["dPitch"])[0]) * 2
    return [
        radiograph("knee-phantom", "Knee phantom", data, pitch, geometry, attribution)
    ]


BUILDERS: dict[str, Callable[[Path], list[ImageObj]]] = {
    "walnut": walnut,
    "circuit-board": circuit_board,
    "knee-phantom": knee_phantom,
}


def main() -> None:
    """Build every real-data example."""
    default = os.environ.get("DATALAB_RADIOGRAPHY_CORPUS", "")
    corpus = Path(sys.argv[1] if len(sys.argv) > 1 else default)
    if not default and len(sys.argv) < 2 or not corpus.is_dir():
        raise SystemExit("Give the corpus folder, or set DATALAB_RADIOGRAPHY_CORPUS")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (
        Conf.plugins_enabled.context(False),
        execenv.context(unattended=True),
        datalab_test_app_context(console=False, exec_loop=False) as window,
    ):
        for example_id, builder in BUILDERS.items():
            images = builder(corpus)
            window.imagepanel.remove_all_objects()
            for image in images:
                window.imagepanel.add_object(image)
            path = OUTPUT_DIR / REAL_EXAMPLE_FILES[example_id]
            window.save_h5_workspace(str(path))
            size = path.stat().st_size
            print(f"{path.name}: {len(images)} images, {size / 1024:.0f} KiB")
            if size > MAX_SIZE:
                raise SystemExit(f"{path.name} exceeds {MAX_SIZE // 1024} KiB")


if __name__ == "__main__":
    main()
