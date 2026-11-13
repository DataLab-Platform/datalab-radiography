# DICOM Images

DataLab reads the pixels of DICOM files (CR, DX, and DICONDE for industrial radiography) but none of their attributes. When a DICOM image is added, the plugin reads the header of its source file with [pydicom](https://pydicom.github.io/) and applies its technical attributes:

| Attribute | Use |
| --- | --- |
| Imager Pixel Spacing | Detector pixel pitch (row and column spacings, read as Y and X) |
| Pixel Spacing, Pixel Spacing Calibration Type | Magnification **DICOM**, when the spacing differs from the imager spacing (calibrated in the object plane) |
| Estimated Radiographic Magnification Factor | Magnification **DICOM** |
| Distance Source to Detector, Distance Source to Patient | Magnification **DICOM** (SDD / SOD), when no factor is given |
| Window Center, Window Width, Rescale Slope, Rescale Intercept | Window, converted to stored pixel values |
| Photometric Interpretation, Presentation LUT Shape | Inverted gray scale for `MONOCHROME1` and `INVERSE` |
| Focal Spots, KVP | Acquisition geometry |

Only technical attributes are copied into the image: no attribute identifying a patient, a part or an operator. Images derived from a DICOM image are not read again, and images whose size differs from the file are ignored.

The **Read the attributes of new DICOM images** preference disables the automatic reading; **Display > Read DICOM attributes** reads the attributes of the selected images on demand, and replaces their pixel pitch and magnifications.

pydicom is installed with the plugin on DataLab Desktop. In DataLab-Web, DICOM attributes are not read.

The reading was checked on 28 real CR and DX chest radiographs of the TCIA COVID-19-AR collection (Fujifilm, Konica Minolta and Philips; `MONOCHROME1` and `MONOCHROME2`; pitches of 0.2, 0.175 and 0.148 mm). These medical images are used for testing only and are not distributed.
