# Detector Corrections and Physical Quantities

## Offset and Gain Correction

Flat-panel detectors add an offset to each pixel and have a non-uniform gain; the X-ray tube adds the heel effect. **Detector corrections > Offset and gain correction...** corrects raw radiographs with mean dark images (beam off) and mean flat images (beam on, no object, same settings):

- corrected image: (I − D) / (F − D) × median(F − D), in digital numbers;
- transmission: (I − D) / (F − D), between 0 and 1 in the free beam.

Images are assigned to the radiograph, dark and flat slots from their frame role (set by the simulator and the examples); images without a role are radiographs. Pixels whose gain (F − D) is below a fraction of its median are invalid (NaN). The defective pixels found in the dark and flat images (hot, dead and weak pixels) are corrected with the median of their valid neighbors. A quality table gives, for each radiograph, the dark and gain levels, and the fractions of invalid, below-dark and saturated pixels.

The **Simulated correction campaign** and **Walnut** examples are ready for this recipe: on the walnut, the transmission is 1 in the air within 3 %.

## Defective Pixels

- **Defective pixel map** marks (1) the pixels deviating from their local median by more than a threshold, in robust standard deviations.
- **Correct defective pixels** replaces them with the median of their valid neighbors.

## Physical Quantities

| Feature | Result |
| --- | --- |
| Attenuation | A = −ln((I − offset) / (I0 − offset)); I0 is the median in the regions of interest (air), a value, or an upper percentile of the image |
| Attenuation with a flat image | A = −ln((I − offset) / (F − offset)), with a second, flat image |
| Equivalent thickness | t = A / μ for a material of known effective attenuation coefficient μ (monochromatic approximation) |

## Step Wedge Thickness Calibration

**Physical quantities > Step wedge thickness calibration...** calibrates the thickness of a material against the attenuation, which takes beam hardening into account:

1. Draw one region of interest per step on the radiograph of a step wedge, in the order of the thicknesses (and, optionally, one in the air).
2. Give the step thicknesses, and optionally dark and flat images (or a free-beam region and the dark level).
3. The recipe fits a polynomial t(A) (degree 2 by default, through the origin when the free beam is known from dark and flat images or from a free-beam region), adds the calibration curve to the signal panel, the thickness map of the wedge and of the other selected radiographs, and a table of the residuals.

On the **Simulated step wedge** example (2 to 20 mm of aluminum, with beam hardening), the calibrated thickness of each step is within 1 %. Warnings report large residuals and non-monotonic calibrations.
