# Measurements

## Measurement Tool

**Measurements > Measurement tool...** opens the radiograph in a dialog with drawing tools: segments, polylines, circles, ellipses, rectangles, polygons and points. Each shape is measured live:

| Shape | Quantities |
| --- | --- |
| Segment | Length |
| Polyline | Length; angle at the middle point for three points |
| Circle | Diameter, area |
| Ellipse | Major and minor axes, area |
| Rectangle | Width, height, area |
| Polygon | Perimeter, area; angle for three points |
| Point | Position |

The table gives every quantity in image coordinates, in the detector plane (if the pixel pitch is known), and in the object plane of each magnification: lengths are divided by M, areas by M². **Planes** shows all planes or the active one; the display unit is cm by default (see [Preferences](display.md#preferences)).

The table can be copied, exported to CSV, or attached to the image as a DataLab result (**Add as result**). **Keep as annotations** stores the shapes on the image, titled with their values.

## Measurement Report

**Measurements > Measurement report...** measures, without dialog, the annotations and the regions of interest of the selected images, and attaches the table to each image as a DataLab analysis result.

## Physical Coordinates

To use the native DataLab measurement tools (cursors, profiles, statistics) in physical units, express the image coordinates in a plane (see [Magnifications](magnifications.md#coordinates-in-a-plane)).
