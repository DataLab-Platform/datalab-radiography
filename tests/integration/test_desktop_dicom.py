"""DICOM images in DataLab Desktop."""

from __future__ import annotations

import os

import numpy as np
import pytest
from datalab.objectmodel import get_uuid

from datalab_radiography.core import metadata as md
from datalab_radiography.workflow import objects

from .helpers import plot_item, radiography_app, the_plugin

pydicom = pytest.importorskip("pydicom")


def write_dicom(path: str, data: np.ndarray, **attributes) -> str:
    """Write a minimal digital X-ray DICOM file."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    sop_class = "1.2.840.10008.5.1.4.1.1.1.1"  # Digital X-Ray Image, for presentation
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = Dataset()
    dataset.file_meta = meta
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.PatientName = "Doe^John"
    dataset.Modality = "DX"
    dataset.Rows, dataset.Columns = data.shape
    dataset.SamplesPerPixel = 1
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    for keyword, value in attributes.items():
        setattr(dataset, keyword, value)
    dataset.PixelData = data.astype("<u2").tobytes()
    pydicom.dcmwrite(path, dataset, enforce_file_format=True)
    return path


def dicom_data() -> np.ndarray:
    """Return a ramp of stored values."""
    return (np.arange(64 * 80).reshape(64, 80) % 4000).astype(np.uint16)


ATTRIBUTES = {
    "PhotometricInterpretation": "MONOCHROME1",
    "ImagerPixelSpacing": [0.2, 0.2],
    "PixelSpacing": [0.16, 0.16],
    "WindowCenter": 2000,
    "WindowWidth": 1001,
    "RescaleSlope": 1,
    "RescaleIntercept": 0,
}


def test_dicom_attributes_of_new_images(tmp_path) -> None:
    """New DICOM images get their pitch, magnification, window and inversion."""
    path = write_dicom(
        os.fspath(tmp_path / "radiograph.dcm"), dicom_data(), **ATTRIBUTES
    )
    with radiography_app() as window:
        window.imagepanel.load_from_files([path])
        image = window.imagepanel.objmodel.get_all_objects()[-1]
        attrs = md.loads(image.metadata[md.DICOM_KEY])
        assert "PatientName" not in attrs
        assert objects.get_detector_pitch(image) == pytest.approx((0.2, 0.2))
        active = objects.get_magnifications(image).active_item
        assert active.factor == pytest.approx(1.25)
        assert (image.zscalemin, image.zscalemax) == pytest.approx((1499.5, 2499.5))
        item = plot_item(window, image)
        assert item.param.colormap == "gray" and item.param.invert_colormap
        assert not any("PatientName" in str(key) for key in image.metadata)


def test_dicom_attributes_on_demand(tmp_path) -> None:
    """With the preference disabled, attributes are read on demand."""
    path = write_dicom(os.fspath(tmp_path / "plate.dcm"), dicom_data(), **ATTRIBUTES)
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.settings_store.update(apply_dicom_attributes=False)
        window.imagepanel.load_from_files([path])
        image = window.imagepanel.objmodel.get_all_objects()[-1]
        assert md.DICOM_KEY not in image.metadata
        assert objects.get_detector_pitch(image) is None
        window.imagepanel.objview.select_objects([get_uuid(image)])
        done = plugin.dicom.apply_to_selection()
        assert done == [image]
        assert objects.get_magnifications(image).active_item.factor == 1.25
        assert plot_item(window, image).param.invert_colormap
        other = image.copy(title="Not DICOM")
        other.metadata.pop("source", None)
        window.imagepanel.add_object(other)
        window.imagepanel.objview.select_objects([get_uuid(other)])
        assert plugin.dicom.apply_to_selection() == []
