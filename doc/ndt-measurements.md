# NDT Measurements

## Wall Thickness (Tangential Radiography)

In tangential radiography, the X-ray beam grazes the wall of a pipe: the wall appears on both sides of the pipe, as a band whose attenuation peaks at the inner wall. **NDT measurements > Wall thickness (tangential)...** reads a profile across the whole pipe, perpendicular to its axis, from the free beam on one side to the free beam on the other side (rectangle drawn as a region of interest, or coordinates), and measures the wall in the plane of the pipe axis (the magnification given as **Pipe axis plane**, the active one by default). Two methods, inspired by ISO 20769-1:

- **Model fit**: least-squares fit of the projected annulus (outer and inner radii, axis position, Beer-Lambert law with beam hardening, smooth background, Gaussian blur), which corrects the blur. On simulated pipes (outer diameters of 20 and 28 mm, walls of 2 to 5 mm), the wall is measured within 1.1 % and the outer diameter within 0.1 %.
- **Edges**: outer edges at the steepest rise of the attenuation, inner edges at its steepest fall after the maxima, refined below the pixel. Both walls are measured separately, within 3 % on the same pipes (the blur biases thin walls upward).

The recipe adds the profile, with both walls annotated, the fitted model, and a table of both methods. A warning reports methods differing by more than 10 % (profile not perpendicular to the axis, or not crossing the whole pipe). The **Simulated pipe** example proposes a suitable rectangle.

## Indication Detection

**NDT measurements > Indication detection...** finds indications (pores, cracks, inclusions) in the regions of interest of the selected radiographs (the whole image if there is none), or in a rectangle:

1. the contrast to a morphological background (larger than the indications) is expressed in noise standard deviations;
2. isolated pixels (defective pixels, direct photon hits) are ignored by a 3 × 3 median;
3. an indication has at least one pixel beyond the detection threshold (5 σ), and extends to the sizing threshold (2.5 σ);
4. its size is given in the plane of a magnification: length and width (largest and smallest Feret diameters), area and equivalent diameter.

Choose **Brighter** indications for voids on raw or transmission radiographs, **Darker** for dense inclusions (or voids on attenuation and thickness images), or both. The recipe adds a copy of each radiograph with the indications annotated, their table, and optionally the label image. No acceptance criterion is applied.

On the **Simulated weld** example (region proposed around the weld bead), the three pores and the crack are found without false call, within 0.15 mm of their position; the pores of 1.2 and 1.8 mm are sized within 4 %, the crack length within 3 %. Small, low-contrast indications are undersized: the pore of 0.7 mm measures 0.5 mm. The toes of a weld bead produce indications: keep them out of the inspected region.
