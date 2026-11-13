"""DataLab-Web adapter contract tests (without DataLab-Web or Qt)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
from datalab.plugins import PluginCapability
from sigima.objects import create_image, create_image_roi
from sigima.objects.annotations import SegmentAnnotation

from datalab_radiography import PLUGIN_ID, __version__, demo
from datalab_radiography.adapters import web
from datalab_radiography.workflow import features as wfeatures
from datalab_radiography.workflow import objects
from datalab_radiography.workflow.recipes import RECIPES


class FakeProcessor:
    """Records the registrations of DataLab-Web's browser processor."""

    def __init__(self) -> None:
        self.features: list[dict] = []

    def _record(self, pattern, func, label, **options) -> None:
        self.features.append(
            {"pattern": pattern, "func": func, "label": label, **options}
        )

    def register_1_to_1(
        self,
        func,
        label,
        paramclass=None,
        icon_name=None,
        menu_path=None,
        feature_id=None,
        owner_plugin_id=None,
    ):  # noqa: E501
        self._record(
            "1_to_1",
            func,
            label,
            paramclass=paramclass,
            menu_path=menu_path,
            feature_id=feature_id,
            owner_plugin_id=owner_plugin_id,
        )  # noqa: E501

    def register_1_to_0(
        self,
        func,
        label,
        paramclass=None,
        icon_name=None,
        menu_path=None,
        feature_id=None,
        owner_plugin_id=None,
    ):  # noqa: E501
        self._record(
            "1_to_0",
            func,
            label,
            paramclass=paramclass,
            menu_path=menu_path,
            feature_id=feature_id,
            owner_plugin_id=owner_plugin_id,
        )  # noqa: E501

    def register_2_to_1(
        self,
        func,
        label,
        paramclass=None,
        icon_name=None,
        menu_path=None,
        obj2_name="Operand",
        skip_xarray_compat=False,
        feature_id=None,
        owner_plugin_id=None,
    ):  # noqa: E501
        self._record(
            "2_to_1",
            func,
            label,
            paramclass=paramclass,
            menu_path=menu_path,
            obj2_name=obj2_name,
            feature_id=feature_id,
            owner_plugin_id=owner_plugin_id,
        )  # noqa: E501

    def register_n_to_1(
        self,
        func,
        label,
        paramclass=None,
        icon_name=None,
        menu_path=None,
        feature_id=None,
        owner_plugin_id=None,
    ):  # noqa: E501
        self._record(
            "n_to_1",
            func,
            label,
            paramclass=paramclass,
            menu_path=menu_path,
            feature_id=feature_id,
            owner_plugin_id=owner_plugin_id,
        )  # noqa: E501


class LegacyProcessor(FakeProcessor):
    """DataLab-Web processor without analyses (1-to-0 features)."""

    register_1_to_0 = None


def web_plugin(processor=None, images=()) -> web.XRayRadiographyNDTWebPlugin:
    """Return the Web plugin attached to a fake browser host."""
    plugin = web.XRayRadiographyNDTWebPlugin()
    plugin.main = SimpleNamespace(
        imagepanel=SimpleNamespace(processor=processor or FakeProcessor())
    )
    plugin.messages = []

    async def message(text: str) -> None:
        plugin.messages.append(text)

    plugin.get_selected_objects = lambda: list(images)
    plugin.show_info_async = plugin.show_warning_async = message
    return plugin


def edited(**values):
    """Return an ``edit`` coroutine accepting a dialog with given values."""

    async def edit(param) -> bool:
        for name, value in values.items():
            setattr(param, name, value)
        return True

    return edit


def test_manifest_and_declarations() -> None:
    """The manifest lists the recipes, examples, tools and features."""
    manifest = web.get_web_manifest()
    assert manifest["plugin_id"] == PLUGIN_ID
    assert manifest["plugin_version"] == __version__
    assert manifest["web_status"] == web.WEB_STATUS
    assert manifest["pyodide_version"] == "0.26.4"
    assert [item["recipe_id"] for item in manifest["recipes"]] == [
        recipe.recipe_id for recipe in RECIPES
    ]
    assert manifest["examples"] == [example.id for example in demo.EXAMPLES]
    assert "xray-simulator" in manifest["tools"]
    assert len(manifest["features"]) == len(wfeatures.features_for(wfeatures.HOST_WEB))
    plugin_class = web.XRayRadiographyNDTWebPlugin
    assert plugin_class.get_plugin_id() == PLUGIN_ID
    assert PluginCapability.APPLICATION in plugin_class.PLUGIN_INFO.capabilities
    assert plugin_class.materialize_example("walnut") is None
    weld = plugin_class.materialize_example("simulated-weld")
    assert objects.get_magnifications(weld.objects[0]).active is not None
    assert isinstance(web_plugin().xray_simulator(), web.RadiographySimulator)


def test_features_are_registered_with_menu_paths() -> None:
    """Every feature of the catalog is registered under the plugin menu."""
    plugin = web_plugin()
    plugin.create_actions()
    registered = plugin.main.imagepanel.processor.features
    catalog = wfeatures.features_for(wfeatures.HOST_WEB)
    assert [item["feature_id"] for item in registered] == [
        feature.feature_id for feature in catalog
    ]
    for item, feature in zip(registered, catalog):
        assert item["pattern"] == feature.pattern
        assert item["menu_path"] == feature.menu_path
        assert item["menu_path"].startswith(f"Plugins/{web.PLUGIN_NAME}/")
        assert item["owner_plugin_id"] == PLUGIN_ID
    legacy = web_plugin(LegacyProcessor())
    legacy.create_actions()
    patterns = {item["pattern"] for item in legacy.main.imagepanel.processor.features}
    assert "1_to_0" not in patterns and "1_to_1" in patterns


def radiograph() -> object:
    """Return a radiograph with a detector pitch."""
    image = create_image("radiograph", np.full((64, 80), 1000.0))
    objects.set_detector_pitch(image, 0.1)
    return image


def test_magnification_tools() -> None:
    """Magnification tools edit their parameters and change the images."""
    first, second = radiograph(), radiograph()
    plugin = web_plugin(images=(first, second))
    plugin.edit = edited(name="Weld", factor=1.25)
    assert asyncio.run(plugin.define_magnification_value()) == [first, second]
    assert objects.get_magnifications(second).active_item.factor == 1.25
    plugin.edit = edited(name="Axis", sdd=1000.0, distance=500.0)
    asyncio.run(plugin.define_magnification_geometry())
    assert objects.get_magnifications(first).get("Axis").factor == pytest.approx(2.0)
    plugin.edit = edited(name="Weld")
    asyncio.run(plugin.set_active_magnification())
    assert objects.get_magnifications(first).active == "Weld"
    first.add_graphical_annotation(SegmentAnnotation(x0=0, y0=0, x1=50, y1=0))
    single = web_plugin(images=(first,))
    captured = {}

    async def edit_length(param) -> bool:
        captured["x1"] = param.x1
        param.name, param.true_length, param.unit = "Ruler", 4.0, "mm"
        return True

    single.edit = edit_length
    asyncio.run(single.define_magnification_length())
    assert captured["x1"] == 50.0
    assert objects.get_magnifications(first).get("Ruler").factor == pytest.approx(1.25)
    third = radiograph()
    copier = web_plugin(images=(first, third))
    assert asyncio.run(copier.copy_magnifications()) == [third]
    assert (
        objects.get_magnifications(third).names
        == objects.get_magnifications(first).names
    )
    plugin.edit = edited(pitch_x=0.2, pitch_y=0.2, unit="mm")
    asyncio.run(plugin.set_detector_pitch())
    assert objects.get_detector_pitch(first) == pytest.approx((0.2, 0.2))


def test_ball_metadata_and_errors() -> None:
    """The ball tool fits a ROI; failures are reported, not raised."""
    weld = demo.materialize_generated_example("simulated-weld").objects[0]
    weld.roi = create_image_roi("rectangle", [395.0, 55.0, 65.0, 60.0])
    plugin = web_plugin(images=(weld,))
    plugin.edit = edited(name="Ball", true_diameter=3.0, unit="mm")
    assert asyncio.run(plugin.define_magnification_ball()) == [weld]
    assert objects.get_magnifications(weld).get("Ball").factor == pytest.approx(
        1.0417, rel=0.01
    )
    plain = create_image("plain", np.zeros((20, 20)))
    failing = web_plugin(images=(plain,))
    assert asyncio.run(failing.define_magnifications_from_metadata()) == []
    assert "plain" in failing.messages[0]
    cancelled = web_plugin(images=(weld,))

    async def cancel(_param) -> bool:
        return False

    cancelled.edit = cancel
    assert asyncio.run(cancelled.define_magnification_value()) == []
    calculator = web_plugin()
    calculator.edit = edited(sdd=1000.0, sod=800.0, focal_spot=0.4, pitch=0.1)
    report = asyncio.run(calculator.geometry_calculator())
    assert report and "1.25" in calculator.messages[0]


def test_display_tools() -> None:
    """Display tools apply the profile, the film convention and windows."""
    image = radiograph()
    image.data = np.linspace(0.0, 1000.0, image.data.size).reshape(image.data.shape)
    plugin = web_plugin(images=(image,))
    image.metadata.update(colormap="viridis", _dlw_lut_range=[0.0, 1.0])
    asyncio.run(plugin.apply_radiographic_display())
    assert image.get_metadata_option("colormap") == "gray"
    assert "colormap" not in image.metadata
    assert "_dlw_lut_range" not in image.metadata
    plugin.edit = edited(film_convention=True)
    asyncio.run(plugin.set_film_convention())
    assert plugin.film_convention and image.get_metadata_option("invert_colormap")
    plugin.edit = edited(method="manual", level=500.0, width=200.0)
    asyncio.run(plugin.edit_window())
    assert (image.zscalemin, image.zscalemax) == pytest.approx((400.0, 600.0))
