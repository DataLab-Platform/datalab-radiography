"""PlotPy dialogs showing a radiograph, for DataLab Desktop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from datalab.adapters_plotpy import create_adapter_from_object
from datalab.env import execenv
from datalab.objectmodel import get_uuid
from guidata.qthelpers import exec_dialog
from plotpy.items import AnnotatedShape
from plotpy.plot import PlotDialog
from qtpy import QtWidgets as QW
from sigima.objects import ImageObj
from sigima.objects.annotations import GraphicalAnnotation
from sigima.viz.annotation_plotpy import plotpy_item_to_annotation

if TYPE_CHECKING:
    from datalab.gui.panel.image import ImagePanel


def image_dialog(
    panel: ImagePanel, obj: ImageObj, title: str, name: str, edit: bool = True
) -> tuple[PlotDialog, object] | None:
    """Return a plot dialog showing an image with its current display settings.

    Args:
        panel: image panel
        obj: image to show
        title: dialog title
        name: dialog object name
        edit: if True, the dialog has OK and Cancel buttons

    Returns:
        Dialog and image item, or None if the dialog cannot be created
    """
    dlg = panel.create_new_dialog(edit=edit, toolbar=True, title=title, name=name)
    if dlg is None:
        return None
    current = panel.plothandler.get(get_uuid(obj))
    item = create_adapter_from_object(obj).make_item(update_from=current)
    item.set_selectable(False)
    item.set_readonly(True)
    dlg.get_plot().add_item(item, z=0)
    dlg.get_plot().do_autoscale()
    return dlg, item


def drawn_annotations(dlg: PlotDialog) -> list[GraphicalAnnotation]:
    """Return the shapes drawn in a plot dialog, as Sigima annotations."""
    annotations = []
    for item in dlg.get_plot().get_items():
        if isinstance(item, AnnotatedShape):
            annotation = plotpy_item_to_annotation(item)
            if annotation is not None:
                annotations.append(annotation)
    return annotations


def pick_shapes(
    panel: ImagePanel,
    obj: ImageObj,
    title: str,
    tool_class: type,
    message: str,
) -> list[GraphicalAnnotation] | None:
    """Let the user draw shapes on an image and return them.

    Args:
        panel: image panel
        obj: image to show
        title: dialog title
        tool_class: PlotPy tool class activated to draw the shapes
        message: instructions shown above the plot

    Returns:
        Drawn shapes (the last one last), or None if the dialog was cancelled
    """
    created = image_dialog(panel, obj, title, "radiography_pick_shapes")
    if created is None:
        return None
    dlg, _item = created
    label = QW.QLabel(message, dlg)
    label.setWordWrap(True)
    dlg.add_widget(label, 1, 0)
    tool = dlg.manager.add_tool(tool_class)
    tool.activate()
    if execenv.unattended:
        return []
    if not exec_dialog(dlg):
        return None
    return drawn_annotations(dlg)
