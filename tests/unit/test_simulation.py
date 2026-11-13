"""Tests of the radiography simulator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from datalab_radiography import demo, simulator
from datalab_radiography.core import simulation as sim


def _noise_free(scene, **detector_options):
    """Return a noise-free simulation without detector artifacts."""
    detector = sim.Detector(
        gain_nonuniformity=0.0,
        offset_nonuniformity_dn=0.0,
        heel=0.0,
        dead_fraction=0.0,
        hot_fraction=0.0,
        blur_px=0.0,
        **detector_options,
    )
    return sim.simulate(scene, detector=detector, darks=1, flats=1, noise=False)


def test_magnifications_and_validation() -> None:
    """Each plane has its magnification; invalid scenes are refused."""
    scene = sim.weld_scene(800.0, 960.0)
    factors = sim.magnifications(scene, sim.Geometry(sdd_mm=1000.0))
    assert factors == pytest.approx({"Weld": 1.25, "Calibration ball": 1000 / 960})
    with pytest.raises(ValueError, match="between source and detector"):
        sim.magnifications(scene, sim.Geometry(sdd_mm=900.0))
    with pytest.raises(ValueError, match="Unknown plane"):
        sim.Scene("bad", {"A": 100.0}, (sim.Box("Box", "B"),))
    with pytest.raises(ValueError, match="Unknown material"):
        sim.Scene("bad", {"A": 100.0}, (sim.Box("Box", "A", "unobtainium"),))


def test_beer_lambert_and_frames() -> None:
    """Noise-free frames follow the attenuation model."""
    scene = sim.Scene("slab", {"Slab": 900.0}, (sim.Box("Slab", "Slab", depth=10.0),))
    result = _noise_free(scene)
    detector = sim.Detector()
    mu, beta = sim.MATERIALS["steel"]
    expected_transmission = math.exp(-mu * 10.0 / (1.0 + beta * 10.0))
    raw = result.radiographs[0].astype(float)
    dark = result.darks[0].astype(float)
    flat = result.flats[0].astype(float)
    assert np.allclose(dark, detector.offset_dn)
    transmission = (raw - dark) / (flat - dark)
    assert np.median(transmission) == pytest.approx(expected_transmission, abs=1e-3)
    assert result.truth["components"][0]["thickness_mm"] == 10.0


def test_tube_voltage_changes_attenuation() -> None:
    """Coefficients follow the voltage; lower voltages give more contrast."""
    assert sim.material_coefficients("steel") == sim.MATERIALS["steel"]
    mu80, beta80 = sim.material_coefficients("steel", 80.0)
    assert mu80 == pytest.approx(0.15 * 2**1.5)
    assert beta80 == pytest.approx(0.010 * 2**1.5)
    with pytest.raises(ValueError, match="between 40 and 450 kV"):
        sim.material_coefficients("steel", 1000.0)
    scene = sim.Scene("slab", {"Slab": 900.0}, (sim.Box("Slab", "Slab", depth=10.0),))
    transmissions = [
        np.median(
            sim.plane_transmissions(scene, sim.Geometry(voltage_kv=v), sim.Detector())[
                "Slab"
            ]
        )
        for v in (80.0, 160.0, 300.0)
    ]
    assert transmissions[0] < transmissions[1] < transmissions[2]


def test_wire_rendering_conserves_area() -> None:
    """Thin wires are rendered with their exact cross-section."""
    detector = sim.Detector(rows=50, cols=200)
    wires = sim.Wires(
        "Wires", "P", "platinum", centers=(-3.0, 3.0), diameters=(0.05, 0.4)
    )
    thickness, slices = sim.render_thickness(wires, detector, factor=1.0)
    row = thickness[thickness.shape[0] // 2]
    pixel = detector.pitch_mm
    # Integral of the thickness across a wire = area of its cross-section
    area = row.sum() * pixel
    expected = math.pi * (0.025**2 + 0.2**2)
    assert area == pytest.approx(expected, rel=1e-6)
    assert slices[1].start > 0


def test_detector_artifacts_and_determinism() -> None:
    """The fixed pattern is stable; frames depend on their seed only."""
    detector = sim.Detector(rows=100, cols=120, dead_fraction=0.01, hot_fraction=0.01)
    pattern = detector.fixed_pattern()
    assert pattern["dead"].sum() > 0 and pattern["hot"].sum() > 0
    assert not (pattern["dead"] & pattern["hot"]).any()
    scene = sim.plate_scene()
    first = sim.simulate(scene, detector=detector, seed=5)
    second = sim.simulate(scene, detector=detector, seed=5)
    other = sim.simulate(scene, detector=detector, seed=6)
    assert np.array_equal(first.radiographs[0], second.radiographs[0])
    assert not np.array_equal(first.radiographs[0], other.radiographs[0])
    frame = first.radiographs[0]
    assert frame.dtype == np.uint16
    assert (frame[pattern["dead"]] == 0).all()
    assert (frame[pattern["hot"]] == detector.max_dn).all()
    assert (
        sim.defect_mask(detector).sum() == pattern["dead"].sum() + pattern["hot"].sum()
    )


def test_geometric_unsharpness_blurs_far_planes() -> None:
    """A plane far from the detector is blurred by the focal spot."""
    detector = sim.Detector(rows=60, cols=200)
    edge = sim.Box("Edge", "P", depth=5.0, x0=0.0)
    profiles = {}
    for focal in (0.0, 1.0):
        scene = sim.Scene("edge", {"P": 500.0}, (edge,))
        transmissions = sim.plane_transmissions(
            scene, sim.Geometry(sdd_mm=1000.0, focal_spot_mm=focal), detector
        )
        profiles[focal] = transmissions["P"][30]
    sharp = np.count_nonzero(np.abs(np.diff(profiles[0.0])) > 1e-3)
    blurred = np.count_nonzero(np.abs(np.diff(profiles[1.0])) > 1e-3)
    assert sharp <= 2 < blurred


def test_scenes_and_truth() -> None:
    """Every scene simulates, with its planes and components in the truth."""
    for name, factory in sim.SCENES.items():
        result = sim.simulate(factory(), detector=sim.Detector(rows=64, cols=80))
        truth = result.truth
        assert truth["scene"] == name
        assert truth["geometry"]["planes"]
        assert len(truth["components"]) == len(factory().components)
    pipe = [c for c in sim.pipe_scene().components if isinstance(c, sim.Pipe)][0]
    assert pipe.truth()["wall_thickness_mm"] == pytest.approx(3.5)
    wedge = sim.step_wedge_scene().components[0]
    assert len(wedge.truth()["step_boxes_mm"]) == 10
    duplex = sim.duplex_iqi("D", "P", 0.0, 0.0)
    assert len(duplex.diameters) == 26 and duplex.diameters[-1] == 0.05


def test_simulator_instrument() -> None:
    """The instrument previews and acquires frames with their metadata."""
    instrument = simulator.RadiographySimulator()
    instrument.settings.rows, instrument.settings.cols = 80, 100
    frame = instrument.preview()
    assert "Weld: M = 1.25" in frame.summary
    low, high = frame.value_range
    assert low < high
    instrument.settings.radiographs, instrument.settings.darks = 2, 1
    acquisition = instrument.acquire()
    assert [obj.title for obj in acquisition.objects] == ["Weld 1", "Weld 2", "Dark"]
    for scene, _title in simulator.SCENE_CHOICES:
        instrument.settings.scene = scene
        assert instrument.preview().objects
    instrument.settings.radiographs = instrument.settings.darks = 0
    with pytest.raises(ValueError, match="at least one"):
        instrument.acquire()
    instrument.settings.radiographs, instrument.settings.rows = 100, 4096
    instrument.settings.cols = 4096
    with pytest.raises(ValueError, match="million pixels"):
        instrument.acquire()


def test_examples_are_declared_and_generated() -> None:
    """Examples are deterministic and carry their metadata."""
    generated = {example.id for example in demo.EXAMPLES if example.resource is None}
    assert generated == set(demo.GENERATED_EXAMPLES)
    assert {example.id for example in demo.REAL_EXAMPLES} == set(
        demo.REAL_EXAMPLE_FILES
    )
    first = demo.materialize_generated_example("simulated-weld").objects[0]
    second = demo.materialize_generated_example("simulated-weld").objects[0]
    assert np.array_equal(first.data, second.data)
    assert demo.materialize_generated_example("unknown") is None
