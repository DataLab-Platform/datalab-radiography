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

Measurements:

- Measurement tool: distances, polylines (with the angle of three points), circles, ellipses, rectangles, polygons and points drawn on the radiograph, measured live in image coordinates, in the detector plane and in the object plane of each magnification (areas divided by M²). The table can be copied, exported to CSV, attached to the image as a DataLab result, and the shapes kept as annotations titled with their values.
- Measurement report: the same table for the annotations and regions of interest of the selected images, as a DataLab analysis.

Simulator and examples:

- X-ray radiography simulator, with a live view: butt weld with pores, a crack, wire and duplex image quality indicators (ISO 19232-1 and -5) and a calibration ball; pipe seen tangentially; aluminum step wedge; cast plate with pores. Each object plane has its own magnification and geometric unsharpness; the detector adds blur, a heel effect, gain and offset non-uniformity, Poisson and read noise, defective pixels and saturation. Acquisitions add radiographs, dark and flat frames with their geometry, magnifications and ground truth.
- Simulated examples: weld (two magnifications), pipe (two magnifications), step wedge, and a correction campaign with dark and flat frames. A welcome page tile opens the weld example.

Detector corrections and physical quantities:

- Offset and gain correction of raw radiographs with mean dark and flat frames (detector offset, gain non-uniformity, heel effect), as corrected images or transmissions, with the correction of the defective pixels found in the dark and flat frames and a quality report (saturation, invalid gain, defective pixels).
- Defective pixel map and correction: pixels deviating from their local median, replaced by the median of their valid neighbors.
- Attenuation -ln(I / I0), with I0 measured in the air (regions of interest), given, or taken from the image; attenuation with a flat image; equivalent thickness of a material of known attenuation coefficient.
- Step wedge thickness calibration: one region of interest per step, optional dark and flat frames, polynomial calibration that takes beam hardening into account (within 0.01 mm on the simulated wedge), calibration curve and thickness maps of the wedge and of other radiographs.

Enhancement:

- Sizes are given in pixels or in a length unit of the object plane of the active magnification (or of the detector plane when the image has no magnification).
- Flatten background: removes thickness variations and the heel effect, by a Gaussian background, or by a morphological background that keeps steps of thickness, so that indications of thin and thick regions share one display window (pores of a simulated stepped casting become 3 to 6 times more visible). Gradients are followed up to the borders of the image.
- Unsharp mask, to enhance details smaller than a given size, such as pores and cracks.
- Adaptive equalization (CLAHE) tuned for radiographs: normalization between two percentiles ignoring the free beam, then equalization by tiles of a given size.
- Denoise with presets: median (direct photon hits), Gaussian, total variation or bilateral (edges kept).
- Robust frame averaging: mean, median, or mean with rejection of outliers such as direct photon hits.

Image quality:

- Signal-to-noise ratio in each region of interest, with slow variations removed, and SNR normalized to a basic spatial resolution of 88.6 µm (inspired by ISO 17636-2).
- Contrast-to-noise ratio of indication regions against a background region.
- Basic spatial resolution with a duplex wire IQI (inspired by ISO 19232-5): robust profile across the wire pairs in a rectangle, dip of each pair (annotated on the profile), first pair below 20 % (iSRb) and SRb interpolated at 20 %, in the IQI plane and in the detector plane. Profiles may be read in either direction; defective pixels are ignored.

NDT measurements:

- Wall thickness in tangential radiography (inspired by ISO 20769-1), in the plane of the pipe axis: fit of the projected annulus, which corrects the blur (within 1.1 % for simulated walls of 2 to 5 mm), and steepest gradient edges (within 3 %), with the profile, the fitted model and the walls annotated.
- Indication detection: indications deviating from the local background by several noise standard deviations, in the regions of interest, sized in the plane of a magnification (length, width, area, equivalent diameter), annotated on a copy of the radiograph, with an optional label image. Isolated defective pixels are ignored. No acceptance criterion is applied. Small, low-contrast indications are undersized (a pore of 0.7 mm measures 0.5 mm in the simulated weld).
- The simulated weld and pipe examples propose these measurements with suitable rectangles.
