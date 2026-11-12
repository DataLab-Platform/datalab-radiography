"""Magnification management in DataLab Desktop."""

from __future__ import annotations

import os

import numpy as np
import pytest
from datalab.objectmodel import get_uuid
from sigima.objects import create_image

from datalab_radiography.core import magnification as mag
from datalab_radiography.workflow import magnification as wmag
from datalab_radiography.workflow import objects

from ..unit.test_circlefit import projected_object
from .helpers import radiography_app, the_plugin

EXPRESS_ID = "org.datalab.radiography:express-coordinates"


def _ball_image():
    """Return a radiograph of a calibration ball (40 pixels wide)."""
    data, xc, yc = projected_object(20.0, "sphere", 1.0)
    image = create_image("ball", (data * 3000).astype(np.float32))
    return image, xc, yc


def test_manager_defines_and_persists_magnifications(tmp_path) -> None:
    """The manager defines magnifications saved in the workspace."""
    filename = os.fspath(tmp_path / "magnifications.h5")
    with radiography_app() as window:
        plugin = the_plugin()
        image, xc, yc = _ball_image()
        window.imagepanel.add_object(image)
        window.imagepanel.objview.select_objects([get_uuid(image)])
        dialog = plugin.magnification_manager()
        assert dialog is not None and dialog.obj is image
        pitch = wmag.DetectorPitchParam()
        pitch.pitch_x = pitch.pitch_y = 0.15
        assert dialog.edit_pitch(pitch)
        assert plugin.settings_store.settings.detector_pitch_mm == pytest.approx(0.15)
        value = wmag.MagnificationValueParam()
        value.name, value.factor = "Object", 2.5
        dialog.add_value(value)
        ball = wmag.MagnificationBallParam()
        ball.true_diameter, ball.roi_index = 2.0, 0
        dialog.add_ball(ball, box=(xc - 30, yc - 30, xc + 30, yc + 30))
        assert dialog.table.rowCount() == 2
        assert dialog.table.item(1, 0).text() == "✓"
        assert dialog.set_active("Object") is not None
        assert dialog.table.item(0, 0).text() == "✓"
        assert dialog.changed
        dialog.reject()
        mset = objects.get_magnifications(image)
        assert mset.get("Calibration ball").factor == pytest.approx(3.0, rel=5e-3)
        window.save_h5_workspace(filename)
        window.load_h5_workspace([filename], reset_all=True)
        loaded = window.imagepanel.objmodel.get_all_objects()[0]
        restored = objects.check_magnifications(loaded)
        assert restored.names == ("Object", "Calibration ball")
        assert restored.active == "Object"


def test_errors_are_reported_without_changes() -> None:
    """A failed definition leaves the image unchanged."""
    with radiography_app() as window:
        plugin = the_plugin()
        image = create_image("flat", np.ones((50, 50)))
        window.imagepanel.add_object(image)
        dialog = plugin.magnification_manager()
        value = wmag.MagnificationValueParam()
        value.name, value.factor = "Object", 2.0
        assert dialog.add_value(value) is None
        assert not dialog.changed
        assert not len(objects.get_magnifications(image))


def test_express_coordinates_feature_and_pitch_action() -> None:
    """The registered feature converts coordinates in DataLab."""
    with radiography_app() as window:
        plugin = the_plugin()
        panel = window.imagepanel
        image = create_image("radiograph", np.random.default_rng(0).random((60, 80)))
        panel.add_object(image)
        panel.objview.select_objects([get_uuid(image)])
        pitch = wmag.DetectorPitchParam()
        pitch.pitch_x = pitch.pitch_y = 0.2
        assert plugin.set_detector_pitch(pitch) == pytest.approx((0.2, 0.2))
        value = wmag.MagnificationValueParam()
        value.name, value.factor = "Weld", 2.0
        wmag.define_value(image, value)
        assert panel.processor.get_feature(EXPRESS_ID) is not None
        param = wmag.ExpressCoordinatesParam()
        param.target, param.unit = mag.FRAME_OBJECT, "mm"
        panel.processor.run_feature(EXPRESS_ID, param)
        result = panel.objmodel.get_all_objects()[-1]
        assert result is not image
        assert result.dx == pytest.approx(0.1) and result.xunit == "mm"
        frame = objects.get_coordinate_frame(result)
        assert frame.plane == "Weld" and frame.magnification == pytest.approx(2.0)


def test_geometry_calculator() -> None:
    """The calculator reports the derived quantities."""
    with radiography_app():
        param = wmag.GeometryCalculatorParam()
        param.sdd, param.sod = 1000.0, 400.0
        report = dict(the_plugin().geometry_calculator(param))
        assert report["Magnification M = SDD / SOD"] == "2.5"
