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

Magnifications:

- Several named magnifications per image (one per object plane), defined from their value, from the source, object and detector distances, from a segment of known length drawn on the image, from a calibration ball or disk (sub-pixel fit of a model of the projected ball, within 0.5 % on simulated balls), or from the acquisition metadata. One of them is active.
- Detector pixel pitch of the selected images, and copy of the magnifications to other images acquired with the same geometry.
- Magnifications are saved with the image in DataLab workspaces, inherited by processing results, and remain valid after binning or ROI extraction.
- Express coordinates in a plane: image coordinates in the object plane of a magnification, in the detector plane or in pixels, so that all DataLab measurements give physical lengths. Regions of interest and annotations keep their position.
- Geometry calculator: magnification, object-plane pixel size, geometric and total unsharpness, and optimal magnification.
