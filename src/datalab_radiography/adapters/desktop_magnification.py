"""Magnification manager and geometry calculator of DataLab Desktop."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import guidata.dataset as gds
from plotpy.tools import AnnotatedRectangleTool, AnnotatedSegmentTool
from qtpy import QtCore as QC
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj
from sigima.objects.annotations import RectangleAnnotation, SegmentAnnotation

from ..core import magnification as mag
from ..core import units
from ..workflow import magnification as wmag
from ..workflow import objects
from .desktop_plot import pick_shapes

if TYPE_CHECKING:
    from .desktop import XRayRadiographyNDTPlugin

COLUMNS = ("Active", "Name", "Method", "M", "Object pixel", "Remarks")


def _edit(param: gds.DataSet, parent: QW.QWidget) -> bool:
    """Edit a parameter set in a dialog."""
    return bool(param.edit(parent=parent))


class MagnificationManagerDialog(QW.QDialog):
    """Define, choose and copy the magnifications of an image.

    Changes apply to the image immediately.

    Args:
        plugin: radiography plugin
        obj: image
    """

    def __init__(self, plugin: XRayRadiographyNDTPlugin, obj: ImageObj) -> None:
        super().__init__(plugin.main)
        self.plugin = plugin
        self.obj = obj
        self.changed = False
        self.setWindowTitle(f"Magnifications - {obj.title}")
        self.setMinimumSize(760, 360)
        layout = QW.QVBoxLayout(self)
        self.frame_label = QW.QLabel(self)
        self.frame_label.setWordWrap(True)
        layout.addWidget(self.frame_label)
        self.table = QW.QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QW.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QW.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QW.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemDoubleClicked.connect(lambda _item: self.set_active())
        layout.addWidget(self.table)
        buttons = QW.QHBoxLayout()
        add_button = QW.QPushButton("Add", self)
        add_menu = QW.QMenu(add_button)
        for title, slot in (
            ("Magnification value...", self.add_value),
            ("Source, object and detector distances...", self.add_geometry),
            ("Known length on the image...", self.add_length),
            ("Calibration ball or disk...", self.add_ball),
            ("From the acquisition metadata", self.add_from_metadata),
        ):
            add_menu.addAction(title, slot)
        add_button.setMenu(add_menu)
        buttons.addWidget(add_button)
        for title, slot in (
            ("Remove", self.remove),
            ("Set active", self.set_active),
            ("Detector pitch...", self.edit_pitch),
            ("Copy to selected images", self.copy_to_selection),
        ):
            button = QW.QPushButton(title, self)
            button.clicked.connect(lambda _checked=False, func=slot: func())
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)
        box = QW.QDialogButtonBox(QW.QDialogButtonBox.Close, self)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.refresh()

    @property
    def unit(self) -> str:
        """Display unit of lengths."""
        return self.plugin.settings_store.settings.unit

    def refresh(self) -> None:
        """Show the coordinate frame and the magnifications of the image."""
        frame = objects.get_coordinate_frame(self.obj)
        pitch = objects.get_detector_pitch(self.obj)
        pitch_text = (
            "unknown"
            if pitch is None
            else " × ".join(units.format_length(value, "mm") for value in pitch)
        )
        self.frame_label.setText(
            f"Image coordinates: <b>{frame.describe()}</b> — detector pixel pitch: "
            f"<b>{pitch_text}</b>"
        )
        try:
            mset = objects.check_magnifications(self.obj)
        except mag.MagnificationError as exc:
            self.frame_label.setText(f"{self.frame_label.text()}<br>{exc}")
            mset = mag.MagnificationSet()
        self.table.setRowCount(len(mset))
        for row, item in enumerate(mset):
            texts = wmag.describe_magnification(item, self.unit)
            values = (
                "✓" if item.name == mset.active else "",
                texts["name"],
                texts["method"],
                texts["factor"],
                texts["pixel"],
                texts["remarks"],
            )
            for column, value in enumerate(values):
                cell = QW.QTableWidgetItem(value)
                if column == 0:
                    cell.setTextAlignment(QC.Qt.AlignCenter)
                self.table.setItem(row, column, cell)
        self.table.resizeColumnsToContents()

    def current_name(self) -> str | None:
        """Return the name of the selected magnification."""
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 1) is None:
            return None
        return self.table.item(row, 1).text()

    def _run(self, function: Callable[[], object]) -> object | None:
        """Run an operation on the image, reporting errors to the user."""
        try:
            result = function()
        except ValueError as exc:
            self.plugin.warn(str(exc), self.windowTitle())
            return None
        self.changed = True
        self.refresh()
        return result

    def _entry_unit(self, param: gds.DataSet) -> gds.DataSet:
        """Prefill the length unit of a definition."""
        if hasattr(param, "unit"):
            param.unit = wmag.DEFAULT_ENTRY_UNIT
        return param

    def _ensure_pitch(self) -> bool:
        """Ask the detector pitch if it is needed and unknown."""
        if objects.detector_mm_per_unit(self.obj) is not None:
            return True
        self.plugin.inform(
            "The detector pixel pitch of the image is unknown: enter it first.",
            self.windowTitle(),
        )
        return self.edit_pitch()

    # -- Definitions -------------------------------------------------------

    def add_value(self, param: wmag.MagnificationValueParam | None = None) -> object:
        """Define a magnification from its value."""
        if param is None:
            if not self._ensure_pitch():
                return None
            param = wmag.MagnificationValueParam()
            if not _edit(param, self):
                return None
        return self._run(lambda: wmag.define_value(self.obj, param))

    def add_geometry(
        self, param: wmag.MagnificationGeometryParam | None = None
    ) -> object:
        """Define a magnification from the acquisition distances."""
        if param is None:
            if not self._ensure_pitch():
                return None
            param = self._entry_unit(wmag.MagnificationGeometryParam())
            if not _edit(param, self):
                return None
        return self._run(lambda: wmag.define_geometry(self.obj, param))

    def add_length(self, param: wmag.MagnificationLengthParam | None = None) -> object:
        """Define a magnification from a segment of known length."""
        if param is None:
            param = self._entry_unit(wmag.MagnificationLengthParam())
            shapes = pick_shapes(
                self.plugin.imagepanel,
                self.obj,
                "Known length",
                AnnotatedSegmentTool,
                "Draw a segment between two marks of known distance, then click OK.",
            )
            if shapes is None:
                return None
            segments = [s for s in shapes if isinstance(s, SegmentAnnotation)]
            if segments:
                segment = segments[-1]
                param.x0, param.y0 = segment.x0, segment.y0
                param.x1, param.y1 = segment.x1, segment.y1
            if not _edit(param, self):
                return None
        return self._run(lambda: wmag.define_length(self.obj, param))

    def add_ball(
        self,
        param: wmag.MagnificationBallParam | None = None,
        box: tuple[float, float, float, float] | None = None,
    ) -> object:
        """Define a magnification from a calibration ball or disk."""
        if param is None:
            param = self._entry_unit(wmag.MagnificationBallParam())
            if self.obj.roi is None or self.obj.roi.is_empty():
                shapes = pick_shapes(
                    self.plugin.imagepanel,
                    self.obj,
                    "Calibration ball",
                    AnnotatedRectangleTool,
                    "Draw a rectangle around the ball (with some background), "
                    "then click OK.",
                )
                if shapes is None:
                    return None
                rectangles = [s for s in shapes if isinstance(s, RectangleAnnotation)]
                if rectangles:
                    rect = rectangles[-1]
                    box = (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
                param.roi_index = 0
            if not _edit(param, self):
                return None
        result = self._run(lambda: wmag.define_ball(self.obj, param, box))
        if result is not None:
            item, fit = result
            self.plugin.inform(
                f"Fitted diameter: {fit.diameter:.5g} (image coordinates), "
                f"residual {fit.rms_residual:.3g}.<br>"
                f"Pixel size in the object plane: "
                f"{units.format_length(item.object_pixel_mm, self.unit)}",
                self.windowTitle(),
            )
        return result

    def add_from_metadata(self) -> object:
        """Define the magnifications given by the acquisition metadata."""
        return self._run(lambda: wmag.define_from_metadata(self.obj))

    # -- Management ----------------------------------------------------------

    def remove(self, name: str | None = None) -> object:
        """Remove the selected magnification."""
        name = name or self.current_name()
        if name is None:
            return None
        return self._run(lambda: wmag.remove_magnification(self.obj, name))

    def set_active(self, name: str | None = None) -> object:
        """Make the selected magnification active."""
        name = name or self.current_name()
        if name is None:
            return None
        return self._run(lambda: wmag.set_active_magnification(self.obj, name))

    def edit_pitch(self, param: wmag.DetectorPitchParam | None = None) -> bool:
        """Edit the detector pixel pitch of the image."""
        if param is None:
            param = wmag.detector_pitch_param(
                self.obj, self.plugin.settings_store.settings.detector_pitch_mm
            )
            if not _edit(param, self):
                return False
        pitch = self._run(lambda: wmag.set_detector_pitch(self.obj, param))
        if pitch is None:
            return False
        self.plugin.settings_store.update(detector_pitch_mm=pitch[0])
        return True

    def copy_to_selection(self) -> object:
        """Copy the magnifications to the other selected images."""
        targets = [
            obj for obj in self.plugin.display.selected_images() if obj is not self.obj
        ]
        if not targets:
            self.plugin.inform("Select other images in the image panel.")
            return None
        return self._run(lambda: wmag.copy_magnifications(self.obj, targets))


def show_geometry_calculator(
    plugin: XRayRadiographyNDTPlugin,
    param: wmag.GeometryCalculatorParam | None = None,
) -> list[tuple[str, str]] | None:
    """Edit an acquisition geometry and show the derived quantities."""
    if param is None:
        param = wmag.GeometryCalculatorParam()
        pitch = plugin.settings_store.settings.detector_pitch_mm
        if pitch:
            param.pitch = pitch
        if not param.edit(parent=plugin.main):
            return None
    try:
        report = wmag.geometry_report(param)
    except ValueError as exc:
        plugin.warn(str(exc))
        return None
    rows = "".join(
        f"<tr><td>{name}</td><td align='right'><b>{value}</b></td></tr>"
        for name, value in report
    )
    plugin.inform(f"<table cellspacing='6'>{rows}</table>")
    return report
