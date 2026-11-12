# <img src="src/datalab_radiography/icons/radiography.svg" alt="" width="48" align="absmiddle"> X-ray Radiography & NDT

A [DataLab](https://datalab-platform.com/) plugin for X-ray radiography and non-destructive testing (NDT): radiographic display, magnifications and calibrated measurements, detector corrections, physical quantities, enhancement, image quality and NDT measurements.

> **Status: Alpha, under development.** The methods are validated on synthetic data with known truth and checked on open real data. They take inspiration from radiography standards (ISO 17636-2, ISO 19232, ASTM E2597, ISO 20769) without claiming compliance. The plugin has no medical diagnostic purpose.

## Installation

Python 3.10 or later is required in a Python environment; the standalone DataLab application supplies its own interpreter.

- **DataLab Desktop:** the plugin needs DataLab 1.4 or later. Download the wheel (`.whl`) attached to the latest [release](https://github.com/DataLab-Platform/datalab-radiography/releases) and install it with **Plugins > Configure plugins... > From file**. In a Python environment, you may instead install it with pip:

  ```bash
  pip install git+https://github.com/DataLab-Platform/datalab-radiography.git
  ```

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## License

BSD 3-Clause, see [LICENSE](LICENSE).
