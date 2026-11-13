# Enhancement

The **Enhancement** menu improves the visibility of indications. Sizes are given in pixels, or as lengths in the object plane of the active magnification (in the detector plane when the image has no magnification): a 2 mm filter keeps its meaning whatever the magnification. Each feature has a live preview.

## Flatten Background

Removes thickness variations and the heel effect, so that indications of thin and thick regions share one display window. The background is estimated:

- **Gaussian**: smooth variations, with a Gaussian of the given size;
- **Morphological**: opening then closing, which removes features smaller than the given size and keeps thickness steps; use about 1.5 times the size of the largest indication.

The background is divided (multiplicative variations, the usual case) or subtracted, and the mean level is kept. The image is extended by odd reflection, so that gradients are followed up to its borders. On a simulated stepped casting, the morphological flattening makes the pores of all steps 3 to 6 times more visible in a single window, where the Gaussian background leaves halos at the steps.

## Unsharp Mask

I + amount × (I − G_σ(I)): enhances details smaller than σ, such as pores and cracks.

## Adaptive Equalization (CLAHE)

Normalizes the image between two percentiles (ignoring the free beam), then equalizes its histogram by tiles of a given size, with a clipping limit. The result lies between 0 and 1. Choose tiles a few times larger than the indications to reveal.

## Denoise

| Method | Use |
| --- | --- |
| Median | Impulse noise, direct photon hits |
| Gaussian | General smoothing |
| Total variation | Noise reduction keeping edges |
| Bilateral | Noise reduction keeping edges |

## Robust Frame Averaging

Combines the selected frames into one image: mean, median, or mean rejecting, for each pixel, the values farther than a threshold (in robust standard deviations) from the median, such as direct photon hits on the detector.
