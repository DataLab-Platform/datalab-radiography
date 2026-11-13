"""Regenerate the documentation screenshots in DataLab Desktop.

Run with the plugin installed in DataLab's Python environment::

    python scripts/take_screenshots.py
"""

from __future__ import annotations

import os

# The screenshots illustrate the English documentation
os.environ["LANG"] = "en"

from pathlib import Path  # noqa: E402

from datalab.config import Conf, ensure_initialized  # noqa: E402
from datalab.env import execenv  # noqa: E402
from datalab.gui.plugins.recipe_runner import RecipeRunner  # noqa: E402
from datalab.objectmodel import get_uuid  # noqa: E402
from datalab.plugins import PluginRegistry  # noqa: E402
from datalab.tests import datalab_test_app_context  # noqa: E402
from qtpy import QtWidgets as QW  # noqa: E402
from qtpy.QtTest import QTest  # noqa: E402

from datalab_radiography import PLUGIN_ID  # noqa: E402
from datalab_radiography.simulator import SIMULATOR_TOOL  # noqa: E402
from datalab_radiography.workflow.ndt_recipes import INDICATION_RECIPE  # noqa: E402
from datalab_radiography.workflow.quality import IndicationParam  # noqa: E402

OUTPUT_DIR = Path(__file__).parents[1] / "doc" / "images"
WINDOW_SIZE = (1280, 800)
SIMULATOR_SIZE = (1280, 720)
#: Leaves time for plots and the simulator's first live frame to be drawn (ms)
RENDER_DELAY = 1_000


def _registered_plugin():
    """Return the plugin instance registered by DataLab at startup."""
    for plugin in PluginRegistry.get_plugins():
        if plugin.plugin_id == PLUGIN_ID:
            return plugin
    raise RuntimeError("Install the plugin in DataLab's Python environment first")


def _save(widget: QW.QWidget, name: str) -> None:
    """Save a screenshot of a widget once it is rendered."""
    QTest.qWait(RENDER_DELAY)
    path = OUTPUT_DIR / name
    if not widget.grab().save(str(path)):
        raise OSError(f"Cannot write {path}")
    print(f"Saved {path}")


def main() -> None:
    """Detect the indications of the simulated weld, then open the simulator."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    execenv.unattended = True
    ensure_initialized(load_user_config=False)
    with (
        Conf.color_mode.context("light"),
        Conf.plugins_enabled.context(True),
        datalab_test_app_context(
            size=WINDOW_SIZE, console=False, exec_loop=False
        ) as window,
    ):
        window.reload_plugins()
        plugin = _registered_plugin()
        plugin.open_example("simulated-weld")
        panel = window.imagepanel
        images = panel.objmodel.get_all_objects()
        param = IndicationParam()
        values = plugin.example_parameter_values(INDICATION_RECIPE.recipe_id, images)
        for name, value in values.items():
            setattr(param, name, value)
        outcome = RecipeRunner(window).run(
            INDICATION_RECIPE, {"images": (images[0],)}, param
        )
        window.set_current_panel("image")
        panel.objview.select_objects([get_uuid(outcome.objects[0].value)])
        _save(window, "overview.png")

        simulator = plugin.launch_tool(SIMULATOR_TOOL.id)
        simulator.resize(*SIMULATOR_SIZE)
        _save(simulator, "simulator.png")
        simulator.close()


if __name__ == "__main__":
    main()
