# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

Python 3.10 or later and DataLab 1.4 or later are required.

First version of the X-ray Radiography & NDT plugin.

Radiographic display:

- New images are displayed in gray scale with an automatic window between two percentiles of their values, ignoring the saturated free beam and dead pixels. Images with explicit display settings, and images read from a workspace, keep their settings; processing results inherit the display of their source. The DataLab preferences are not changed, and disabling the plugin restores the usual display.
- Film convention (dense matter bright), applied to new images and to the selected images. DICOM images are displayed as their attributes require.
- Window of the selected images: automatic, from the regions of interest, from DICOM attributes, or from a level and a width.
- Preferences of the plugin (automatic display, film convention, percentiles, length unit, DICOM attributes).
