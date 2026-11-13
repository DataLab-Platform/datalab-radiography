"""Installed distribution, reload and persistence qualification."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from datalab.adapters_metadata import TableAdapter
from datalab.gui.plugins.recipe_runner import RecipeRunner
from datalab.objectmodel import get_uuid
from datalab.plugins.recipes import RECIPE_RUN_RECORD_OPTION, RecipeRunRecord
from sigima.objects.annotations import SegmentAnnotation

from datalab_radiography import PLUGIN_ID, demo
from datalab_radiography.workflow import enhancement as wenh
from datalab_radiography.workflow import measurements as wmeas
from datalab_radiography.workflow import ndt_recipes as recipes
from datalab_radiography.workflow import objects
from datalab_radiography.workflow import quality as wquality

from .helpers import radiography_app, the_plugin

PROJECT_ROOT = Path(__file__).parents[2]
#: Size and SHA-256 of the packaged examples (update after scripts/build_examples.py)
EXAMPLE_FILES = {
    "walnut": (
        1_019_544,
        "3CAF228299608A64890E409341C5EE53ECCC9D73C8758A3B45AC392418A732A1",
    ),
    "circuit-board": (
        518_440,
        "400B481AF1B91BEA996996074CE39F3E31242016E80216E9C563FE2E6D330100",
    ),
    "knee-phantom": (
        473_048,
        "29EEFCCFAA4F54B5D14F93147A060176D4D7EFD18E9E983D4EF75AC143AA29D4",
    ),
}
ICONS = [
    "display.svg",
    "magnification.svg",
    "measure.svg",
    "radiography.svg",
    "radiography_demo.svg",
    "xray_simulator.svg",
]


def test_wheel_installs_and_resolves_entry_point_outside_checkout(tmp_path) -> None:
    """A built wheel imports its plugin and examples from an isolated target."""
    wheel_dir = tmp_path / "wheel"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", os.fspath(wheel_dir)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheel_dir.glob("*.whl"))
    assert len(wheels) == 1
    install_dir = tmp_path / "installed"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--target",
            os.fspath(install_dir),
            os.fspath(wheels[0]),
        ],
        check=True,
    )
    probe = """
import hashlib
import importlib
import importlib.metadata as metadata
import importlib.resources as resources
import json

distribution = next(metadata.distributions(path=[INSTALL_DIR]))
entry_point = next(
    value for value in distribution.entry_points if value.group == "datalab.plugins"
)
plugin_class = entry_point.load()
module = importlib.import_module(plugin_class.__module__)
web_entry_point = next(
    value for value in distribution.entry_points
    if value.group == "datalab.web_plugins"
)
web_plugin_class = web_entry_point.load()
web_module = importlib.import_module(web_plugin_class.__module__)
examples = {}
for example in plugin_class.get_examples():
    if example.resource is not None:
        payload = example.resolve().read_bytes()
        digest = hashlib.sha256(payload).hexdigest().upper()
        examples[example.id] = [len(payload), digest]
package = resources.files("datalab_radiography")
print(json.dumps({
    "entry_point": entry_point.name,
    "examples": examples,
    "icons": sorted(icon.name for icon in (package / "icons").iterdir()),
    "module_file": module.__file__,
    "plugin_id": plugin_class.get_plugin_id(),
    "requires": sorted(distribution.requires or []),
    "sources": (package / "examples" / "SOURCES.md").is_file(),
    "web_entry_point": web_entry_point.name,
    "web_module_file": web_module.__file__,
    "web_plugin_id": web_plugin_class.get_plugin_id(),
}))
""".replace("INSTALL_DIR", repr(os.fspath(install_dir)))
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        (os.fspath(install_dir), environment.get("PYTHONPATH", ""))
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)
    assert result["entry_point"] == "datalab_radiography"
    assert result["plugin_id"] == PLUGIN_ID
    assert result["module_file"] == os.fspath(
        install_dir / "datalab_radiography" / "adapters" / "desktop.py"
    )
    assert result["web_entry_point"] == "datalab_radiography"
    assert result["web_plugin_id"] == PLUGIN_ID
    assert result["web_module_file"] == os.fspath(
        install_dir / "datalab_radiography" / "adapters" / "web.py"
    )
    assert result["icons"] == ICONS
    assert result["sources"]
    assert result["examples"] == {
        key: list(value) for key, value in EXAMPLE_FILES.items()
    }
    assert any(
        item.startswith("pydicom") and 'sys_platform != "emscripten"' in item
        for item in result["requires"]
    )


def test_examples_and_sources_are_packaged() -> None:
    """Every real-data example is a small packaged file listed in SOURCES.md."""
    folder = PROJECT_ROOT / "src" / "datalab_radiography" / "examples"
    sources = (folder / "SOURCES.md").read_text(encoding="utf-8")
    for example in demo.REAL_EXAMPLES:
        filename = demo.REAL_EXAMPLE_FILES[example.id]
        assert (folder / filename).stat().st_size < 1024 * 1024
        assert f"`{filename}`" in sources
        assert example.resolve().is_file()


def test_magnifications_measurements_and_provenance_survive_h5(tmp_path) -> None:
    """Magnifications, measurement tables and recipe provenance survive HDF5."""
    filename = os.fspath(tmp_path / "radiography-round-trip.h5")
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("simulated-weld")
        images = window.imagepanel.objmodel.get_all_objects()
        weld = images[0]
        magnifications = objects.get_magnifications(weld)
        weld.add_graphical_annotation(
            SegmentAnnotation(x0=100.0, y0=100.0, x1=225.0, y1=100.0, title="Length")
        )
        window.imagepanel.objview.select_objects([get_uuid(weld)])
        window.imagepanel.processor.run_feature(
            f"{PLUGIN_ID}:measurement-report", wmeas.MeasurementReportParam()
        )
        measurements = [
            table.result.to_dict() for table in TableAdapter.iterate_from_obj(weld)
        ]
        assert measurements
        feature = window.imagepanel.processor.get_feature(
            f"{PLUGIN_ID}:flatten-background"
        )
        assert None not in feature.plugin_origin.values()
        window.imagepanel.processor.run_feature(
            feature.feature_id, wenh.FlattenBackgroundParam()
        )
        flattened = window.imagepanel.objmodel.get_all_objects()[-1]
        param = wquality.DuplexParam()
        for name, value in plugin.example_parameter_values(
            recipes.DUPLEX_RECIPE.recipe_id, images
        ).items():
            setattr(param, name, value)
        outcome = RecipeRunner(window).run(
            recipes.DUPLEX_RECIPE, {"image": (weld,)}, param
        )
        profile = outcome.objects[0].value
        record = RecipeRunRecord.from_dict(
            profile.get_metadata_option(RECIPE_RUN_RECORD_OPTION)
        )
        tables = {
            table.title: table.result.to_dict()
            for table in TableAdapter.iterate_from_obj(profile)
        }
        window.save_h5_workspace(filename)
        window.load_h5_workspace([filename], reset_all=True)
        loaded_weld = window.find_object_by_uuid(get_uuid(weld))
        assert objects.get_magnifications(loaded_weld) == magnifications
        assert window.find_object_by_uuid(get_uuid(flattened)) is not None
        assert [
            table.result.to_dict()
            for table in TableAdapter.iterate_from_obj(loaded_weld)
        ] == measurements
        loaded = window.find_object_by_uuid(get_uuid(profile))
        assert (
            RecipeRunRecord.from_dict(
                loaded.get_metadata_option(RECIPE_RUN_RECORD_OPTION)
            )
            == record
        )
        assert {
            table.title: table.result.to_dict()
            for table in TableAdapter.iterate_from_obj(loaded)
        } == tables
