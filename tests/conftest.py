"""Test configuration shared by all tests."""

from __future__ import annotations

import os
import tempfile

import pytest

# Qt widgets are created off screen, also on Windows
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# The plugin never reads or writes the preferences of the user during tests
SETTINGS_ENV = "DATALAB_RADIOGRAPHY_SETTINGS"
os.environ.setdefault(
    SETTINGS_ENV,
    os.path.join(tempfile.mkdtemp(prefix="radiography-tests-"), "settings.json"),
)


@pytest.fixture(autouse=True)
def default_preferences():
    """Start every test with the default preferences of the plugin."""
    path = os.environ[SETTINGS_ENV]
    if os.path.exists(path):
        os.remove(path)
    yield
    if os.path.exists(path):
        os.remove(path)
