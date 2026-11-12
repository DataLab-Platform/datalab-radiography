"""Dependency rules for host-independent plugin layers.

``core`` holds the scientific code (NumPy and SciPy only). ``workflow``,
``simulator`` and ``demo`` may use Sigima, guidata data sets and DataLab's
headless plugin contracts. None of them may import a GUI toolkit, a host
adapter or a module unavailable in the browser (DataLab-Web runs Pyodide).
"""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parents[2] / "src" / "datalab_radiography"
PACKAGE_NAME = "datalab_radiography"
HOST_MODULES = (
    "PyQt5",
    "PyQt6",
    "PySide6",
    "qtpy",
    "qwt",
    "plotpy",
    "sigimax",
    "guidata.qthelpers",
    "guidata.widgets",
    "guidata.dataset.qtwidgets",
    "guidata.dataset.qtitemwidgets",
    "datalab.gui",
    "datalab.adapters_plotpy",
    "datalab.widgets",
    "datalab.config",
    "pydicom",
    f"{PACKAGE_NAME}.adapters",
    "js",
    "pyodide",
)
HEADLESS_MODULES = (
    f"{PACKAGE_NAME}.simulator",
    f"{PACKAGE_NAME}.demo",
)
FORBIDDEN_LAYER_IMPORTS = {
    "core": (
        "datalab",
        "sigima",
        "guidata",
        "h5py",
        "skimage",
        f"{PACKAGE_NAME}.workflow",
        *HEADLESS_MODULES,
    ),
    "workflow": HEADLESS_MODULES,
}
TOP_LEVEL_HEADLESS_FILES = ("simulator.py", "demo.py")


def _module_imports(filename: Path) -> set[str]:
    """Return direct imports declared by one Python source file."""
    tree = ast.parse(filename.read_text(encoding="utf-8"), filename=str(filename))
    relative_parts = filename.relative_to(PACKAGE_ROOT).with_suffix("").parts
    package_parts = (PACKAGE_NAME, *relative_parts[:-1])
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0:
                if node.module is not None:
                    imports.add(node.module)
                continue
            base_parts = package_parts[: len(package_parts) - node.level + 1]
            if node.module is not None:
                imports.add(".".join((*base_parts, *node.module.split("."))))
            else:
                imports.update(
                    ".".join((*base_parts, *alias.name.split(".")))
                    for alias in node.names
                )
    return imports


def _violations(filenames: list[Path], forbidden: tuple[str, ...]) -> list[str]:
    """Return the forbidden imports of the given files."""
    violations: list[str] = []
    for filename in filenames:
        for imported_module in sorted(_module_imports(filename)):
            if any(
                imported_module == prefix or imported_module.startswith(f"{prefix}.")
                for prefix in forbidden
            ):
                violations.append(
                    f"{filename.relative_to(PACKAGE_ROOT)} imports {imported_module}"
                )
    return violations


def test_package_root_does_not_import_host_adapters() -> None:
    """Importing package identity must not load a host adapter."""
    imports = _module_imports(PACKAGE_ROOT / "__init__.py")
    assert not any(name.startswith(f"{PACKAGE_NAME}.adapters") for name in imports)


def test_import_scanner_resolves_relative_imports() -> None:
    """Relative imports cannot bypass the architecture checks."""
    imports = _module_imports(PACKAGE_ROOT / "workflow" / "__init__.py")
    assert f"{PACKAGE_NAME}.workflow.recipes" in imports


def test_headless_layers_do_not_import_host_modules() -> None:
    """Core and workflow remain independent from Desktop and Web hosts."""
    violations: list[str] = []
    for layer in ("core", "workflow"):
        forbidden = HOST_MODULES + FORBIDDEN_LAYER_IMPORTS[layer]
        filenames = sorted((PACKAGE_ROOT / layer).rglob("*.py"))
        violations.extend(_violations(filenames, forbidden))
    assert not violations, "\n".join(violations)


def test_headless_modules_do_not_import_host_modules() -> None:
    """The simulator and the examples run in DataLab Desktop and DataLab-Web."""
    filenames = [
        PACKAGE_ROOT / name
        for name in TOP_LEVEL_HEADLESS_FILES
        if (PACKAGE_ROOT / name).is_file()
    ]
    assert not _violations(filenames, HOST_MODULES)


def test_only_desktop_adapters_import_qt() -> None:
    """The Web adapter stays importable without Qt, PlotPy or pydicom."""
    web_adapter = PACKAGE_ROOT / "adapters" / "web.py"
    forbidden = tuple(
        name for name in HOST_MODULES if not name.startswith(PACKAGE_NAME)
    ) + (f"{PACKAGE_NAME}.adapters.desktop",)
    assert not _violations([web_adapter], forbidden)
