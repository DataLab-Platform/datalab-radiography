# Radiographic Display

## Default Display

While the plugin is active, every new image is displayed as a radiograph:

- gray scale, linear;
- window between two percentiles of the image values (0.5 % and 99.5 % by default), ignoring the saturated free beam and dead pixels, so that the part fills the gray levels;
- optional film convention.

Images with explicit display settings, and images read from a workspace, keep their settings. Processing results inherit the display of their source, with a window computed again on their own values. The display is stored in the image (DataLab metadata), not in the DataLab preferences: disabling the plugin restores the usual display of new images.

## Film Convention

Radiographs are displayed as recorded by digital detectors: the more matter, the darker. **Display > Film convention (dense matter bright)** inverts the gray scale, as on a film viewed on a light box, for new images and for the selected images. DICOM images follow their own attributes (`MONOCHROME1`, inverse presentation shape) whatever the convention.

## Windows

The **Display** menu applies to the selected images:

| Action | Window |
| --- | --- |
| Apply radiographic display | Gray scale, inversion and automatic window |
| Automatic window | Between the percentiles of the preferences |
| Window from regions of interest | Between the percentiles of the values in the regions of interest |
| Window... | Percentiles, DICOM window attributes, or level and width |
| Read DICOM attributes | Pixel spacing, magnification, window and gray scale of DICOM files |

## Preferences

**Preferences...** sets the automatic display of new images, the film convention, the percentiles of automatic windows, the display unit of lengths (cm by default) and the reading of DICOM attributes.
