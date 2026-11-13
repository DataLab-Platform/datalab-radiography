"""Measurement tool and report in DataLab Desktop."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from datalab.adapters_metadata import TableAdapter
from datalab.objectmodel import get_uuid
from datalab.plugins.tools import PluginTool
from plotpy.builder import make
from sigima.objects import create_image
from sigima.objects.annotations import SegmentAnnotation, annotation_to_dict

from datalab_radiography.workflow import magnification as wmag
from datalab_radiography.workflow import measurements as wmeas
from datalab_radiography.workflow import objects

from .helpers import radiography_app, the_plugin

REPORT_ID = "org.datalab.radiography:measurement-report"


def _calibrated_image():
    """Return an image with a 0.2 mm pitch and a magnification of 2."""
    image = create_image("radiograph", np.random.default_rng(0).random((200, 200)))
    objects.set_detector_pitch(image, 0.2)
    param = wmag.MagnificationValueParam()
    param.name, param.factor = "Weld", 2.0
    wmag.define_value(image, param)
    return image


def test_measurement_tool(tmp_path) -> None:
    """Drawn shapes are measured, exported, and kept on the image."""
    with radiography_app() as window:
        plugin = the_plugin()
        image = _calibrated_image()
        window.imagepanel.add_object(image)
        window.imagepanel.objview.select_objects([get_uuid(image)])
        tool = plugin.measurement_tool()
        assert tool is not None and tool.dialog is not None
        assert tool.table.rowCount() == 0
        plot = tool.dialog.get_plot()
        plot.add_item(make.annotated_segment(0, 0, 100, 0, title="Gap"))
        assert tool.table.rowCount() == 1
        rows = tool.table_rows()
        assert rows[0][:3] == ["Shape", "Quantity", "Image (px)"]
        assert rows[1][0] == "Gap"
        assert rows[1][-1] == "1 cm"
        tool.mode.setCurrentIndex(3)
        assert tool.table.columnCount() == 3
        assert "Weld" in tool.table.horizontalHeaderItem(2).text()
        assert "1 cm" in tool.copy_to_clipboard()
        filename = tool.export_csv(str(tmp_path / "measurements.csv"))
        with open(filename, encoding="utf-8") as file:
            exported = list(csv.reader(file))
        assert exported[1][:2] == ["Gap", "Length"]
        assert float(exported[1][-1]) == pytest.approx(1.0)
        assert tool.add_result()
        tables = list(TableAdapter.iterate_from_obj(image))
        assert len(tables) == 1 and tables[0].result.data[0][0] == "Gap"
        assert tool.keep_annotations() == 1
        kept = image.get_annotations()[-1]
        assert kept["kind"] == "segment" and "Length 1 cm (Weld)" in kept["title"]


def test_measurement_tool_window_size() -> None:
    """The window makes room for the table, then keeps the user's geometry."""
    with radiography_app() as window:
        plugin = the_plugin()
        assert [tool.id for tool in plugin.get_tools()] == [
            "xray-simulator",
            "measurement-tool",
            "magnifications",
            "geometry-calculator",
        ]
        # The tools stay in the submenus where users expect them
        assert plugin.measure_action.text() == "Measurement tool..."
        assert plugin.magnification_action.text() == "Magnifications..."
        if hasattr(PluginTool, "menu"):
            assert not plugin.get_tool("measurement-tool").menu
        image = _calibrated_image()
        window.imagepanel.add_object(image)
        window.imagepanel.objview.select_objects([get_uuid(image)])
        plugin.settings_store.update(measurement_window=None)
        tool = plugin.measurement_tool()
        plot_size = window.imagepanel.get_dialog_size()
        assert tool.dialog.width() >= min(
            plot_size[0] + 460,
            int(0.9 * tool.dialog.screen().availableGeometry().width()),
        )
        tool.dialog.resize(900, 640)
        tool._store_geometry()  # pylint: disable=protected-access
        assert plugin.settings_store.settings.measurement_window
        tool.dialog.close()
        again = plugin.measurement_tool()
        assert (again.dialog.width(), again.dialog.height()) == (900, 640)
        again.dialog.close()


def test_measurement_report_feature() -> None:
    """The registered report attaches a table to the image."""
    with radiography_app() as window:
        image = _calibrated_image()
        image.set_annotations(
            [annotation_to_dict(SegmentAnnotation(x0=0, y0=0, x1=100, y1=0))]
        )
        window.imagepanel.add_object(image)
        window.imagepanel.objview.select_objects([get_uuid(image)])
        param = wmeas.MeasurementReportParam()
        param.unit = "mm"
        window.imagepanel.processor.run_feature(REPORT_ID, param)
        tables = list(TableAdapter.iterate_from_obj(image))
        assert len(tables) == 1
        assert tables[0].result.data[0][2:] == pytest.approx([100.0, 20.0, 10.0])
        html = tables[0].to_html()
        assert "Segment 1" in html
