"""Interactive measurement tool of DataLab Desktop."""

from __future__ import annotations

import csv
import io
from typing import TYPE_CHECKING

from datalab.adapters_metadata import TableAdapter
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from guidata.qthelpers import exec_dialog
from plotpy.tools import (
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedPointTool,
    AnnotatedPolygonTool,
    AnnotatedRectangleTool,
    AnnotatedSegmentTool,
    MultiLineTool,
)
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj
from sigima.objects.annotations import annotation_to_dict

from ..core import measurement as meas
from ..workflow import measurements as wmeas
from .desktop_plot import drawn_annotations, image_dialog

if TYPE_CHECKING:
    from plotpy.plot import PlotDialog

    from .desktop import XRayRadiographyNDTPlugin

ALL_PLANES = "All planes"
DRAWING_TOOLS = (
    AnnotatedSegmentTool,
    MultiLineTool,
    AnnotatedCircleTool,
    AnnotatedEllipseTool,
    AnnotatedRectangleTool,
    AnnotatedPolygonTool,
    AnnotatedPointTool,
)
HELP = (
    "Draw shapes with the tools of the toolbar: segment (distance), polyline "
    "(length; three points give an angle), circle, ellipse, rectangle, polygon "
    "and point. Values are given in image coordinates, in the detector plane and "
    "in the object plane of each magnification."
)


class MeasurementTool:
    """Measure shapes drawn on a radiograph, in every plane.

    Args:
        plugin: radiography plugin
        obj: image to measure
    """

    def __init__(self, plugin: XRayRadiographyNDTPlugin, obj: ImageObj) -> None:
        self.plugin = plugin
        self.obj = obj
        self.planes: list[meas.Plane] = []
        self.rows: list[meas.MeasurementRow] = []
        self.dialog: PlotDialog | None = None
        self.table: QW.QTableWidget | None = None
        self.mode: QW.QComboBox | None = None

    @property
    def unit(self) -> str:
        """Display unit of lengths."""
        return self.plugin.settings_store.settings.unit

    def create_dialog(self) -> PlotDialog | None:
        """Create the dialog of the tool."""
        panel = self.plugin.imagepanel
        created = image_dialog(
            panel,
            self.obj,
            f"Measurements - {self.obj.title}",
            "radiography_measure",
            edit=False,
        )
        if created is None:
            return None
        dlg, _item = created
        for tool_class in DRAWING_TOOLS:
            dlg.manager.add_tool(tool_class)
        side = QW.QWidget(dlg)
        layout = QW.QVBoxLayout(side)
        help_label = QW.QLabel(HELP, side)
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        self.mode = QW.QComboBox(side)
        self.mode.addItem(ALL_PLANES)
        for plane in wmeas.planes_for(self.obj):
            self.mode.addItem(plane.title)
        self.mode.currentIndexChanged.connect(lambda _index: self.update_table())
        form = QW.QFormLayout()
        form.addRow("Planes:", self.mode)
        layout.addLayout(form)
        self.table = QW.QTableWidget(0, 0, side)
        self.table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        buttons = QW.QGridLayout()
        for index, (title, slot) in enumerate(
            (
                ("Copy", self.copy_to_clipboard),
                ("Export CSV...", self.export_csv),
                ("Add as result", self.add_result),
                ("Keep as annotations", self.keep_annotations),
            )
        ):
            button = QW.QPushButton(title, side)
            button.clicked.connect(lambda _checked=False, func=slot: func())
            buttons.addWidget(button, index // 2, index % 2)
        layout.addLayout(buttons)
        close = QW.QDialogButtonBox(QW.QDialogButtonBox.Close, side)
        close.rejected.connect(dlg.reject)
        layout.addWidget(close)
        side.setMinimumWidth(460)
        dlg.add_widget(side, 0, 1)
        plot = dlg.get_plot()
        for signal in (
            plot.SIG_ITEMS_CHANGED,
            plot.SIG_ITEM_MOVED,
            plot.SIG_ITEM_HANDLE_MOVED,
            plot.SIG_ANNOTATION_CHANGED,
            plot.SIG_ITEM_REMOVED,
        ):
            signal.connect(lambda *_args: self.update_table())
        self.dialog = dlg
        self.update_table()
        return dlg

    def exec(self) -> None:
        """Show the tool (modal)."""
        if self.create_dialog() is not None and not execenv.unattended:
            exec_dialog(self.dialog)

    # -- Measurements --------------------------------------------------------

    def shapes(self) -> list[meas.Shape]:
        """Return the shapes drawn in the dialog."""
        if self.dialog is None:
            return []
        return wmeas.shapes_from_annotations(self.obj, drawn_annotations(self.dialog))

    def selected_planes(self) -> list[int]:
        """Return the indexes of the planes shown in the table."""
        if self.mode is None or self.mode.currentIndex() <= 0:
            return list(range(len(self.planes)))
        return [self.mode.currentIndex() - 1]

    def update_table(self) -> None:
        """Measure the drawn shapes and fill the table."""
        self.planes, self.rows = wmeas.measure(
            self.obj, self.shapes(), wmeas.PLANES_ALL, self.unit
        )
        if self.table is None:
            return
        indexes = self.selected_planes()
        headers = ["Shape", "Quantity"] + [
            wmeas.plane_header(self.planes[index], self.unit) for index in indexes
        ]
        self.table.clear()
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setRowCount(len(self.rows))
        for row_index, row in enumerate(self.rows):
            texts = [row.shape, wmeas.quantity_label(row)] + [
                self._format(row, index) for index in indexes
            ]
            for column, text in enumerate(texts):
                self.table.setItem(row_index, column, QW.QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def _format(self, row: meas.MeasurementRow, plane_index: int) -> str:
        """Format a value of a row in a plane."""
        value = row.values[plane_index]
        plane = self.planes[plane_index]
        if plane.scale is None and row.dimension in (meas.LENGTH, meas.COORDINATE):
            return f"{value:.5g}" if value == value else "-"
        unit = wmeas.image_unit(self.obj) if plane.scale is None else self.unit
        return wmeas.format_value(value, row, unit)

    def table_rows(self) -> list[list[str]]:
        """Return the table shown, header first."""
        if self.table is None:
            return []
        columns = self.table.columnCount()
        rows = [[self.table.horizontalHeaderItem(c).text() for c in range(columns)]]
        for row in range(self.table.rowCount()):
            rows.append([self.table.item(row, c).text() for c in range(columns)])
        return rows

    # -- Actions ---------------------------------------------------------------

    def copy_to_clipboard(self) -> str:
        """Copy the table to the clipboard (tab-separated values)."""
        text = "\n".join("\t".join(row) for row in self.table_rows())
        QW.QApplication.clipboard().setText(text)
        return text

    def export_csv(self, filename: str | None = None) -> str | None:
        """Export the table of every plane to a CSV file."""
        if filename is None:
            filename, _filter = QW.QFileDialog.getSaveFileName(
                self.dialog, "Export measurements", "", "CSV files (*.csv)"
            )
            if not filename:
                return None
        table = wmeas.measurement_table(self.planes, self.rows, self.unit)
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(table.headers)
        writer.writerows(table.data)
        with open(filename, "w", encoding="utf-8", newline="") as file:
            file.write(buffer.getvalue())
        return filename

    def add_result(self) -> bool:
        """Attach the measurements to the image as a DataLab result."""
        if not self.rows:
            self.plugin.inform("Draw shapes to measure first.")
            return False
        table = wmeas.measurement_table(self.planes, self.rows, self.unit)
        TableAdapter(table).add_to(self.obj)
        self.plugin.imagepanel.refresh_plot(get_uuid(self.obj), True, False)
        self.plugin.imagepanel.SIG_OBJECT_MODIFIED.emit()
        return True

    def keep_annotations(self) -> int:
        """Add the drawn shapes to the image annotations, titled with their values."""
        if self.dialog is None:
            return 0
        annotations = drawn_annotations(self.dialog)
        shapes = wmeas.shapes_from_annotations(self.obj, annotations)
        index = self.mode.currentIndex() - 1 if self.mode is not None else -1
        plane_index = index if index >= 0 else len(self.planes) - 1
        stored = list(self.obj.get_annotations())
        for annotation, shape in zip(annotations, shapes):
            rows = [row for row in self.rows if row.shape == shape.title]
            text = ", ".join(
                f"{row.quantity} {self._format(row, plane_index)}"
                for row in rows
                if row.dimension in (meas.LENGTH, meas.AREA, meas.ANGLE)
            )
            plane = self.planes[plane_index].title
            title = f"{shape.title}: {text} ({plane})" if text else shape.title
            data = annotation_to_dict(annotation)
            data["title"] = title
            stored.append(data)
        self.obj.set_annotations(stored)
        self.plugin.imagepanel.refresh_plot(get_uuid(self.obj), True, False)
        self.plugin.imagepanel.SIG_OBJECT_MODIFIED.emit()
        return len(annotations)
