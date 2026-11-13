# Qualification

The plugin is **Alpha**: its methods are validated on synthetic data with known truth, and checked on open real data. They take inspiration from radiography standards without claiming compliance: ISO 17636-2 (normalized SNR, magnification technique), ISO 19232-1 and -5 (wire and duplex IQIs), ASTM E2597 (detector characterization), ISO 20769-1 (tangential radiography). The plugin has no medical diagnostic purpose.

## Validation on Simulated Data

The simulator renders every component with its truth (see [Simulator](simulator.md)). `tests/validation` checks:

| Method | Criterion | Result |
| --- | --- | --- |
| Calibration ball or disk | Radius of balls and disks of 6 to 36 pixels, with blur and noise | Within 0.5 % from 10 pixels, 1 % below; center within 0.05 pixel |
| Magnification from a ball in the image | Calibration ball of the weld and pipe examples, in its own plane | Magnification within 0.5 % |
| Magnifications from metadata | Planes of the weld | Exact; measured lengths exact in the object plane |
| Offset and gain correction | Correction campaign | Median transmission within 1 %; residual non-uniformity below 3 % |
| Step wedge calibration | 10 steps of 2 to 20 mm, beam hardening | Each step within 1 % |
| Background flattening | Stepped casting with pores | Pores 3 to 6 times more visible (morphological background) |
| Unsharp mask | Weld pores | Contrast increased by 20 % at least |
| Duplex IQI | Weld, noisy and noise-free; larger focal spot | Same element within one; larger SRb with a larger focal spot |
| Wall thickness | Outer diameters 20 and 28 mm, walls 2 to 5 mm | Model within 2 % (1.1 % observed), edges within 3 % |
| Indication detection | Weld: three pores and a crack | All found, no false call, position within 0.15 mm, pores of 1 mm or more within 10 %, crack length within 10 % |

SNR, CNR, attenuation, equivalent thickness, enhancement filters, Feret diameters and DICOM translations are tested on analytic images with exact results (`tests/unit`).

## Checks on Real Data

With the local corpus (see [Contributing](../CONTRIBUTING.md#local-corpus-of-real-radiographs)):

- **CWI walnut** (raw flat-panel projections, dark and flat images): transmission 1 in the air within 3 % after correction; magnification 3.015 from the scan settings, consistent with the published voxel size.
- **LUT circuit board** (laminography): attenuation close to 0 in the air with the free-beam level of the dataset; magnification 6.952 from the geometry file.
- **TCIA COVID-19-AR** (28 CR and DX DICOM files): gray-scale inversion of `MONOCHROME1`, pixel pitch from Imager Pixel Spacing, window from the attributes.

## Limits

- Magnifications assume object planes parallel to the detector: tilted parts (such as the laminography circuit board) shorten lengths along the tilt.
- The equivalent thickness uses a monochromatic attenuation coefficient; use the step wedge calibration for a polychromatic beam.
- Indications are sized at a fixed contrast threshold: small, low-contrast indications are undersized (a pore of 0.7 mm measures 0.5 mm in the simulated weld). Weld toes produce indications.
- The wall thickness methods require a profile crossing the whole pipe, perpendicular to its axis, with free beam on both sides.
- The duplex IQI analysis assigns peaks with the nominal wire spacings of ISO 19232-5 in the IQI plane: the magnification of that plane must be known.

## Continuous Integration

CI runs the unit, validation and integration tests (except the corpus checks) on Linux and Windows with the published DataLab 1.4.0 stack, including a test that builds the wheel, installs it outside the checkout, loads the plugin from its entry point and reads the packaged examples.
