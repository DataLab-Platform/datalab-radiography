# Architecture

## Layers

Dependencies point inward:

```text
adapters -> workflow -> core
simulator, demo -> workflow -> core
```

| Layer | Content | May import |
| --- | --- | --- |
| `core` | Scientific code: units, metadata schema, magnification model, circle fit, display windows, measurements, simulation, corrections, attenuation, enhancement, image quality, wall thickness, indications, DICOM attribute translation | NumPy, SciPy |
| `workflow` | Sigima computation functions, parameter sets, recipes, the catalog of features, metadata accessors of images, preferences | Sigima, guidata data sets, headless DataLab contracts, scikit-image |
| `simulator.py`, `demo.py` | Simulator instrument, generated and real-data examples | as `workflow` |
| `adapters/desktop*.py` | DataLab Desktop plugin: menus, dialogs, display controller, DICOM header reading | Qt, PlotPy, pydicom, DataLab GUI |
| `adapters/web.py` | DataLab-Web plugin | as `workflow`, plus the DataLab-Web plugin API |

`core` runs unchanged in DataLab Desktop, DataLab-Web (Pyodide) and plain Python. `workflow`, `simulator` and `demo` never import a GUI toolkit, pydicom or an adapter. `tests/unit/test_architecture.py` checks these rules by parsing the imports of every module.

## Features and Recipes

`workflow/features.py` declares each processing feature once (`FeatureSpec`: identifier, title, menu group, pattern 1→1, 1→0, 2→1 or n→1, Sigima function, parameter class, hosts). Each adapter registers the catalog with its host: Desktop in `register_computations()`, with stable feature identifiers `org.datalab.radiography:<local id>` owned by the plugin; DataLab-Web in `create_actions()`, with menu paths.

The Sigima functions (`workflow/computations.py`) declare an operation identifier and a contract version for DataLab provenance: increase the contract version whenever a change alters a result.

Recipes (`workflow/correction_recipes.py`, `workflow/ndt_recipes.py`) are headless DataLab recipes: typed input slots, parameters, binding suggestions and diagnostics. They run identically in both hosts.

## Metadata

The plugin stores its data in image metadata, as JSON strings with a schema version, under keys prefixed with `plugin.org.datalab.radiography.` (`core/metadata.py`): magnifications, coordinate frame, detector pixel pitch, acquisition geometry, display profile, DICOM attributes, simulation truth, attribution, frame and output roles. DataLab saves them in workspaces and copies them to processing results.

## Display

DataLab Desktop writes its default display options into an image when it first displays it, before plugins are notified. The display controller (`adapters/desktop_display.py`) therefore applies the radiographic profile to new images whose colormap is still the default one, and leaves images with explicit settings (including images read from workspaces) unchanged. New-image hooks run before the profile: the DICOM controller uses one to apply the attributes of DICOM files.
