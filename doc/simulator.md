# Simulator and Examples

## X-ray Radiography Simulator

**X-ray radiography simulator...** (in the **Applications** catalog and in the plugin menu of the image panel) shows a live view of a simulated radiograph next to its settings, and acquires radiographs, dark and flat images with their metadata.

Scenes:

- **Butt weld**: steel plate with a weld bead, three pores, a crack, a wire IQI (ISO 19232-1) and a duplex IQI (ISO 19232-5) in the weld plane, and a calibration ball closer to the detector: two magnifications;
- **Pipe**: steel pipe seen tangentially, with a calibration ball on the detector: two magnifications;
- **Step wedge**: aluminum, 2 to 20 mm;
- **Cast plate**: aluminum plate with pores.

The model is deterministic, with known truth:

- each component lies in an object plane, with its magnification M = SDD / SOD;
- attenuation follows a Beer-Lambert law with a beam hardening term, A(t) = μ t / (1 + β t), per material; the coefficients are given at 160 kV and scaled by (160 / V)^1.5 with the tube voltage V (40 to 450 kV): lower voltages give more contrast, and less signal behind thick parts;
- each plane is blurred by the focal spot (geometric unsharpness f (M − 1));
- the detector adds its blur, the heel effect, gain and offset non-uniformity, Poisson and read noise, dead and hot pixels, and 14-bit saturation.

Settings: scene, source-to-object distances, source-to-detector distance, focal spot, voltage, photons per pixel (100 to 30,000: the free beam saturates the detector above about 28,000) and averaged frames, detector size, pitch, blur, read noise, gain non-uniformity, heel effect, defective pixels, and the numbers of radiographs, dark and flat images to acquire. **Restore defaults** (DataLab 1.4.1 or later, DataLab-Web) sets them back to their default values.

Acquired images carry their role (radiograph, dark, flat), detector pixel pitch, geometry, magnifications and ground truth, so that every method of the plugin can be checked against the truth.

## Examples

**Examples** in the plugin menu, or the **Applications** catalog, opens:

| Example | Content | Suggested methods |
| --- | --- | --- |
| Simulated weld | Weld with pores, crack, IQIs and calibration ball | Magnifications, measurements, duplex IQI, indication detection, enhancement |
| Simulated pipe (tangential view) | Pipe and calibration ball | Wall thickness, measurements in two planes |
| Simulated step wedge | Step wedge with one region of interest per step, dark and flat images | Step wedge thickness calibration |
| Simulated correction campaign | Cast plate radiographs, dark and flat images, detector with defective pixels | Offset and gain correction, defective pixels |
| Walnut (raw flat-panel radiograph) | Real raw radiograph with dark and flat images, M = 3.015 | Offset and gain correction, attenuation, measurements |
| Circuit board (laminography, 45° tilt) | Real radiograph of a circuit board, M = 6.95, free beam 1992 DN | Measurements, enhancement, attenuation |
| Knee phantom (cone-beam CT projection) | Real flat-field corrected projection, M = 1.707 | Display, enhancement, measurements |

The circuit board is tilted by 45°: lengths along the tilt are shortened by cos 45° in the plane of the rotation axis. The real-data examples are reduced extracts of open datasets under the CC BY 4.0 license; see [SOURCES.md](../src/datalab_radiography/examples/SOURCES.md) for their attribution.
