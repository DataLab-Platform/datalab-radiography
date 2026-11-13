# Getting Started

## Find the Plugin

Once installed (see the [README](../README.md#installation)), the plugin adds the **Plugins > X-ray Radiography & NDT** menu to the image panel of DataLab. The **Applications** catalog, opened from the welcome page, lists the plugin with its examples and its simulator; a welcome page tile opens the simulated weld.

When the plugin is active, new images are displayed as radiographs: gray scale and automatic window (see [Radiographic display](display.md)). Disabling the plugin with **Plugins > Configure plugins...** restores the usual display; the DataLab preferences are never changed.

## First Inspection

1. Choose **Plugins > X-ray Radiography & NDT > Examples > Simulated weld...**. DataLab opens a radiograph of a butt weld with pores, a crack, wire and duplex image quality indicators (IQI), and a calibration ball. The image has two magnifications: the weld plane (active) and the plane of the ball, closer to the detector.
2. Choose **Measurement tool...** in the plugin menu, draw a segment along the crack and read its length in pixels, in the detector plane and in both object planes. **Keep as annotations** stores the segment, titled with its length.
3. Choose **Image quality > Basic spatial resolution (duplex IQI)...**: the parameters propose a rectangle across the duplex IQI. DataLab adds the profile with the dip of each wire pair, and the basic spatial resolution (SRb) in the IQI and detector planes.
4. Choose **NDT measurements > Indication detection...**: DataLab adds a copy of the radiograph with the three pores and the crack annotated, and their table (length, width, area, equivalent diameter in mm in the weld plane).
5. Save the workspace: magnifications, annotations, tables and the provenance of the results are kept.

## Your Own Radiographs

1. Open the radiograph (TIFF, DICOM, PNG, HDF5...). DICOM files give the pixel pitch, the magnification and the window (see [DICOM images](dicom.md)).
2. Give the detector pixel pitch with **Magnification > Detector pixel pitch...** if it is unknown.
3. Define the magnification of each plane of interest with **Magnifications...** in the plugin menu: value, acquisition distances, segment of known length, or calibration ball or disk (see [Magnifications](magnifications.md)).
4. Measure, correct, enhance and inspect: lengths entered in the parameters of the plugin (sizes of filters, indications) refer to the active magnification.

Raw flat-panel radiographs should first be corrected with their dark and flat images (**Detector corrections > Offset and gain correction...**, see [Detector corrections](corrections.md)).
