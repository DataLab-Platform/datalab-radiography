# DataLab-Web

[DataLab-Web](https://github.com/DataLab-Platform/web) runs DataLab in the browser (Pyodide) and bundles the plugin. The same scientific code runs in both hosts; the differences come from the browser:

| Feature | DataLab Desktop | DataLab-Web |
| --- | --- | --- |
| Processing features (enhancement, corrections, physical quantities, coordinates) | Plugin menu | **Plugins > X-ray Radiography & NDT** menu |
| Analyses (measurement report, SNR, CNR) | Plugin menu | Plugin menu, with DataLab-Web versions supporting analyses |
| Recipes, examples, simulator | Plugin menu and Applications catalog | Applications catalog |
| Magnifications | Manager dialog with drawing (segment, ball rectangle) | Tools in the Applications catalog: value, distances, known length (last segment annotation of the image, or coordinates), calibration ball (in a region of interest), metadata, active magnification, copy, pixel pitch, geometry calculator |
| Measurement tool | Interactive dialog | Not available: use annotations or regions of interest and the measurement report |
| Display | Automatic for new images | Examples and simulated images carry the radiographic display; **Apply radiographic display**, **Film convention...** and **Window...** tools for other images |
| DICOM attributes | Read with pydicom | Not read |
| Preferences | Stored | Not stored: the film convention lasts for the session |
