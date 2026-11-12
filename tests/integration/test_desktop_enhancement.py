"""Enhancement features in DataLab Desktop."""

from __future__ import annotations

import numpy as np
from datalab.objectmodel import get_uuid

from datalab_radiography.workflow import display as wdisplay
from datalab_radiography.workflow import enhancement as wenh
from datalab_radiography.workflow import features as wfeatures

from .helpers import radiography_app, submenu_texts, the_plugin

FEATURE_PARAMS = {
    "flatten-background": wenh.FlattenBackgroundParam.create(size=5.0, unit="mm"),
    "unsharp-mask": wenh.UnsharpMaskParam.create(sigma=0.5, unit="mm"),
    "radiographic-clahe": wenh.RadiographicClaheParam(),
    "radiographic-denoise": wenh.RadiographicDenoiseParam(),
}


def new_images(window, before: set[str]) -> list:
    """Return the images added since ``before`` (set of UUIDs)."""
    return [obj for obj in window.imagepanel.objmodel if get_uuid(obj) not in before]


def test_enhancement_features_in_desktop() -> None:
    """Enhancement features have menu actions and keep the radiographic display."""
    with radiography_app() as window:
        plugin = the_plugin()
        processor = window.imagepanel.processor
        enhancement = [
            feature
            for feature in wfeatures.features_for(wfeatures.HOST_DESKTOP)
            if feature.group == wfeatures.GROUP_ENHANCEMENT
        ]
        assert {feature.local_id for feature in enhancement} == {
            *FEATURE_PARAMS,
            "robust-average",
        }
        for feature in enhancement:
            assert processor.get_feature(feature.feature_id) is not None
        assert [
            text.removesuffix("...")
            for text in submenu_texts(window, wfeatures.GROUP_ENHANCEMENT)
        ] == [feature.title for feature in enhancement]
        plugin.open_example("simulated-weld")
        weld = window.imagepanel.objmodel.get_all_objects()[0]
        for local_id, param in FEATURE_PARAMS.items():
            window.imagepanel.objview.select_objects([get_uuid(weld)])
            before = {get_uuid(obj) for obj in window.imagepanel.objmodel}
            window.imagepanel.processor.run_feature(
                f"{plugin.plugin_id}:{local_id}", param
            )
            added = new_images(window, before)
            assert len(added) == 1, local_id
            assert added[0].data.shape == weld.data.shape
            assert np.isfinite(added[0].data).mean() > 0.99
            assert wdisplay.has_profile(added[0]), local_id


def test_robust_average_in_desktop() -> None:
    """Robust averaging combines the selected frames into one image."""
    with radiography_app() as window:
        plugin = the_plugin()
        plugin.open_example("simulated-weld")
        weld = window.imagepanel.objmodel.get_all_objects()[0]
        frames = []
        for index in range(3):
            frame = weld.copy(title=f"Frame {index}")
            frame.data = np.asarray(weld.data, dtype=float) + index
            window.imagepanel.add_object(frame)
            frames.append(frame)
        frames[1].data[5, 5] = 1e6
        window.imagepanel.objview.select_objects([get_uuid(obj) for obj in frames])
        before = {get_uuid(obj) for obj in window.imagepanel.objmodel}
        window.imagepanel.processor.run_feature(
            f"{plugin.plugin_id}:robust-average", wenh.RobustAverageParam()
        )
        added = new_images(window, before)
        assert len(added) == 1
        assert np.allclose(added[0].data[10:, 10:], weld.data[10:, 10:] + 1.0)
        assert added[0].data[5, 5] < 1e5
