# Image Quality

The methods take inspiration from ISO 17636-2 (normalized SNR) and ISO 19232-5 (duplex wire IQI) without claiming compliance.

## Signal-to-Noise Ratio

**Image quality > Signal-to-noise ratio...** computes, in each region of interest of the selected images (the whole image if there is none), the mean, the noise (standard deviation) and SNR = mean / noise. Measure it on an image linear in dose (raw, corrected or transmission), in homogeneous regions:

- **Remove slow variations** fits a plane in each region, so that the heel effect and thickness gradients are not counted as noise;
- with the **basic spatial resolution** of the detector (SRb, measured with a duplex IQI), the table adds the normalized SNR, SNRN = SNR × 88.6 µm / SRb.

## Contrast-to-Noise Ratio

**Image quality > Contrast-to-noise ratio...** compares indication regions with a background region (the last one by default): CNR = |mean(indication) − mean(background)| / noise(background). Draw at least two regions of interest.

## Basic Spatial Resolution (Duplex IQI)

A duplex wire IQI holds 13 pairs of wires, D1 (0.80 mm) to D13 (0.05 mm); in each pair, the gap equals the wire diameter. **Image quality > Basic spatial resolution (duplex IQI)...**:

1. reads a profile across the wire pairs in a rectangle (the first rectangular region of interest of the image, or coordinates), averaged across the wires with a trimmed mean that ignores defective pixels;
2. removes a straight baseline, then assigns the wire peaks to the elements in order, using the nominal spacing of each pair in the IQI plane (the profile may run from D1 to D13 or the reverse);
3. computes the dip of each pair, (A + B − 2C) / (A + B), from the two peaks A and B and the minimum C between them;
4. stops at the first pair with a dip below 20 % (iSRb), and interpolates the wire diameter at a 20 % dip between it and the previous pair: the basic spatial resolution SRb, half the image unsharpness.

The **IQI plane** is the name of the magnification of the plane where the IQI lies (empty: the active magnification; `Detector`: an IQI on the detector). The recipe adds the profile, with each element annotated with its dip, and two tables: the elements, and iSRb with SRb in the IQI plane and in the detector plane (SRb × M).

On the **Simulated weld** example, the duplex IQI gives iSRb = D10 and SRb ≈ 112 µm in the weld plane (140 µm in the detector plane); the noisy and noise-free radiographs agree within one element, and a larger focal spot gives a larger SRb.
