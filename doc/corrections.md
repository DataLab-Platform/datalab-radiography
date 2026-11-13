# Detector Corrections and Physical Quantities

## Offset and Gain Correction

Flat-panel detectors add an offset to each pixel and have a non-uniform gain; the X-ray tube adds the heel effect. **Detector corrections > Offset and gain correction...** corrects raw radiographs with mean dark images (beam off) and mean flat images (beam on, no object, same settings):

- corrected image: (I − D) / (F − D) × median(F − D), in digital numbers;
- transmission: (I − D) / (F − D), between 0 and 1 in the free beam.

Images are assigned to the radiograph, dark and flat slots from their frame role (set by the simulator and the examples); images without a role are radiographs. Pixels whose gain (F − D) is below a fraction of its median are invalid (NaN). The defective pixels found in the dark and flat images (hot, dead and weak pixels) are corrected with the median of their valid neighbors. A quality table gives, for each radiograph, the dark and gain levels, and the fractions of invalid, below-dark and saturated pixels.

Saturation is checked in the radiographs and in the flat images, without the defective pixels; a saturated flat image underestimates the gain and makes the correction wrong. A pixel is saturated when it reaches the **Saturation level** of the detector:

- the value given in the parameters, if not zero;
- otherwise, the full scale of the digitizer given by the image: 2^BitsStored − 1 for DICOM images, or the bit depth of the simulator;
- otherwise, the saturation is estimated (heuristic): when the maximum of an image of digital numbers is close to the full scale 2^k − 1 of an 8 to 16-bit digitizer, the pixels within 2 % of this full scale are reported as a *possible saturation*. Give the saturation level of the detector to confirm it.

Less than 0.1 % of saturated pixels is not reported.

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

The calibration is refused, with a message naming the steps involved, when:

- the attenuation does not increase with the step thickness (regions of interest in the wrong order, wrong thicknesses, or thick steps beyond the dynamic range of the detector, which give the same attenuation);
- there are fewer steps of different thicknesses than the polynomial needs;
- the calibration curve does not increase over the calibrated attenuation range (from zero when through the origin): lower the polynomial degree or add steps.

On the **Simulated step wedge** example (2 to 20 mm of aluminum, with beam hardening), the calibrated thickness of each step is within 1 %. Warnings report large residuals, and thickness maps of which more than 1 % of the pixels have an attenuation outside the calibrated range, beyond a margin of 5 % of this range (their thickness is extrapolated).
