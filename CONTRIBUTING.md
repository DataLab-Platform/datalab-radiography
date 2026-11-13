# Contributing

## Setup and Checks

Install the project in editable mode, in a Python environment where DataLab 1.4 or later is available:

```bash
python -m pip install -e ".[test]"
```

Before submitting a change, run:

```bash
python -m ruff format .
python -m ruff check .
python -m pytest
```

CI runs the published DataLab 1.4.0, Sigima 1.4.0 and SigimaX 1.1.0 stack on Linux and Windows. A manual run can select a development branch of DataLab, Sigima and SigimaX with `stack-ref`; release tags run the checks before the draft release is prepared (see [RELEASING.md](RELEASING.md)).

To reproduce the CI environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install "sigima==1.4.0" "sigimax==1.1.0" "datalab-platform[qt]==1.4.0"
.venv\Scripts\python.exe -m pip install -e ".[test]"
```

## Trying the Plugin in DataLab

With DataLab 1.4 or later, add this project folder in **Plugins > Configure plugins... > Advanced**, then reload the plugins: DataLab loads the plugin from `src/datalab_radiography/adapters/desktop.py`, without installing the project. The packages the plugin depends on (NumPy, SciPy and pydicom) must be available in DataLab's environment.

## Architecture

Dependencies point inward:

```text
adapters -> workflow -> core
simulator, demo -> workflow -> core
```

- `core` holds the scientific code, in NumPy and SciPy only. It does not import DataLab, Sigima or guidata, so that it runs unchanged in DataLab Desktop, DataLab-Web (Pyodide) and plain Python.
- `workflow` turns the core into Sigima computation functions, headless DataLab recipes and the catalog of the plugin features. It never imports GUI modules, PlotPy, pydicom or the adapters.
- `simulator.py` exposes the radiography simulator as a DataLab instrument; `demo.py` builds the examples. Both follow the rules of `workflow`.
- `adapters/desktop*.py` integrate the plugin in DataLab Desktop (menus, dialogs, display profile, DICOM attributes). `adapters/web.py` integrates it in DataLab-Web and must not import Qt, PlotPy or pydicom.

`tests/unit/test_architecture.py` checks these boundaries; [doc/architecture.md](doc/architecture.md) details them.

## Tests

- `tests/unit`: core, workflow and adapters.
- `tests/validation`: results against the simulator truth, and checks on the local corpus of real radiographs.
- `tests/integration`: DataLab Desktop integration, Web adapter, installed wheel, hot reload and workspace round trip.

### Local Corpus of Real Radiographs

Some validation tests use open real data that is not stored in the repository (several hundred megabytes, including human medical images that must not be redistributed). Set `DATALAB_RADIOGRAPHY_CORPUS` to the folder holding the corpus to run them; they are skipped otherwise, as in CI. The corpus layout and the licenses of its datasets are described in `SOURCES.md` at the root of the corpus folder.

### Real-Data Examples

The packaged examples `src/datalab_radiography/examples/*.h5` are reduced extracts of the corpus (less than 1 MiB each), with their attribution in `examples/SOURCES.md` and in the metadata of each image. Rebuild them with:

```bash
python scripts/build_examples.py
```

Then update the sizes and SHA-256 checksums pinned in `tests/integration/test_distribution_lifecycle.py`.

### Screenshots

`python scripts/take_screenshots.py` regenerates the images of `doc/images`, with the plugin installed in DataLab's environment.

## Branches

Day-to-day work lands on `develop`, the default branch. `main` is the release branch: `develop` is merged into `main` only to cut a release (see [RELEASING.md](RELEASING.md)).
