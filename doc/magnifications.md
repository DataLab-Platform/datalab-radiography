# Magnifications

## Object Planes

In a radiograph, the source projects each object plane onto the detector with its own magnification M = SDD / SOD (source-to-detector distance over source-to-object distance). A length measured on the detector is M times the length in the object plane. A radiograph may hold several planes of interest: the axis of a pipe and a reference placed on the detector, the source and detector sides of a double-wall exposure, a calibration ball and the part.

The plugin stores, with each image, any number of named magnifications, one per object plane, and an active one. Each magnification gives the length, in mm in its object plane, of one unit of image coordinates along X and Y, and the factor M when known. Magnifications are saved in DataLab workspaces, inherited by processing results, and remain valid after binning, resampling or ROI extraction. If the coordinates of an image are changed outside the plugin, the magnifications are refused until defined again.

## Detector Pixel Pitch

**Magnification > Detector pixel pitch...** sets the pitch of the selected images, in mm. It is needed to convert a magnification factor into lengths, and to give lengths in the detector plane. DICOM files and the simulated examples carry it.

## Defining a Magnification

**Magnification > Magnifications...** (also a tool of the **Applications** panel) shows the magnifications of the image, with their method and object-plane pixel size, and defines new ones:

| Method | Input |
| --- | --- |
| Magnification value | The factor M (pixel pitch required) |
| Source, object and detector distances | SDD and SOD, or SDD and object-to-detector distance; the focal spot size gives the geometric unsharpness |
| Known length on the image | A segment drawn on the image, and its true length in the object plane |
| Calibration ball or disk | A rectangle around the ball and its true diameter: the ball is located by a sub-pixel fit of a model of its projection (within 0.5 % from a radius of 10 pixels on simulated balls) |
| Acquisition metadata | Distances or magnification given by the simulator or by DICOM attributes |

The dialog also makes a magnification active, removes one, and copies the magnifications to the other selected images acquired with the same geometry.

## Coordinates in a Plane

**Magnification > Express coordinates in a plane...** changes the coordinates of an image (not its data) so that they read directly in the object plane of a magnification, in the detector plane, or in pixels, in the unit of your choice. All DataLab measurements (cursors, profiles, ROI statistics) then give physical lengths. Regions of interest and annotations keep their position on the image, and the magnifications stay valid.

## Geometry Calculator

**Magnification > Geometry calculator...** (also a tool of the **Applications** panel) computes, from the distances, the focal spot size and the basic spatial resolution of the detector:

- the magnification and the object-plane pixel size;
- the geometric unsharpness Ug = f (M - 1), in the detector plane;
- the total image unsharpness in the object plane, ((Ug³ + (2 SRb)³)^(1/3)) / M;
- the optimal magnification 1 + (2 SRb / f)^1.5, which minimizes it.
