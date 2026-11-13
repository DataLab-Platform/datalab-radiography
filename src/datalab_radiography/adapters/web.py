"""DataLab-Web adapter of the radiography plugin.

DataLab-Web runs the plugin in the browser (Pyodide). The processing features
of the shared catalog are registered with menu paths in :meth:`create_actions`;
recipes, examples and the simulator are offered by DataLab-Web's generic host;
tools open asynchronous dialogs on the selected images (magnifications,
detector pixel pitch, geometry calculator, display). DICOM attributes are not
read in the browser (pydicom is not available).

This module must not import Qt, PlotPy or pydicom.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import guidata.dataset as gds
from datalab.plugins import PluginBase, PluginCapability, PluginInfo
from datalab.plugins.examples import PluginExampleData
from datalab.plugins.recipes import RecipeObjectType
from datalab.plugins.tools import PluginTool, ToolSelection
from sigima.objects import ImageObj
from sigima.objects.annotations import SegmentAnnotation

from .. import PLUGIN_DESCRIPTION, PLUGIN_ID, PLUGIN_NAME, __version__
from ..demo import EXAMPLES, materialize_generated_example
from ..simulator import SIMULATOR_TOOL, RadiographySimulator
from ..workflow import RECIPES, objects
from ..workflow import display as wdisplay
from ..workflow import features as wfeatures
from ..workflow import magnification as wmag
from ..workflow import parameters as wparams

#: Browser qualification of this adapter
WEB_STATUS = "candidate"
#: First DataLab-Web version bundling the plugin
DATALAB_WEB_VERSION = "0.12.0"
PYODIDE_VERSION = "0.26.4"
PLUGIN_ICON = "datalab_radiography:icons/radiography.svg"
MAGNIFICATION_ICON = "datalab_radiography:icons/magnification.svg"
DISPLAY_ICON = "datalab_radiography:icons/display.svg"
DOCUMENTATION_URL = "https://github.com/DataLab-Platform/datalab-radiography"
#: Metadata keys of the LUT range and colormap chosen in DataLab-Web
DATALAB_WEB_LUT_KEY = "_dlw_lut_range"
DATALAB_WEB_DISPLAY_KEYS = (
    "colormap",
    "colourmap",
    "invert_colormap",
    "colormap_inverted",
    DATALAB_WEB_LUT_KEY,
)


def _image_tool(
    tool_id: str,
    title: str,
    launcher: str,
    description: str,
    icon: str,
    selection: ToolSelection = ToolSelection.AT_LEAST_ONE,
) -> PluginTool:
    """Return a tool working on the selected images."""
    return PluginTool(
        id=tool_id,
        title=title,
        launcher=launcher,
        description=description,
        icon=icon,
        object_type=RecipeObjectType.IMAGE,
        selection=selection,
    )


MAGNIFICATION_TOOLS = (
    _image_tool(
        "magnification-value",
        "Magnification value...",
        "define_magnification_value",
        "Define the magnification of an object plane from its value",
        MAGNIFICATION_ICON,
    ),
    _image_tool(
        "magnification-geometry",
        "Magnification from distances...",
        "define_magnification_geometry",
        "Define a magnification from the source, object and detector distances",
        MAGNIFICATION_ICON,
    ),
    _image_tool(
        "magnification-length",
        "Magnification from a known length...",
        "define_magnification_length",
        "Define a magnification from a segment of known length (the last "
        "segment annotation of the image, or coordinates)",
        MAGNIFICATION_ICON,
        ToolSelection.EXACTLY_ONE,
    ),
    _image_tool(
        "magnification-ball",
        "Magnification from a calibration ball...",
        "define_magnification_ball",
        "Define a magnification from a calibration ball or disk in a region of "
        "interest",
        MAGNIFICATION_ICON,
        ToolSelection.EXACTLY_ONE,
    ),
    _image_tool(
        "magnification-metadata",
        "Magnifications from metadata",
        "define_magnifications_from_metadata",
        "Define the magnifications given by the acquisition metadata",
        MAGNIFICATION_ICON,
    ),
    _image_tool(
        "active-magnification",
        "Active magnification...",
        "set_active_magnification",
        "Choose the active magnification of the selected images",
        MAGNIFICATION_ICON,
    ),
    _image_tool(
        "copy-magnifications",
        "Copy magnifications",
        "copy_magnifications",
        "Copy the magnifications of the first selected image to the others",
        MAGNIFICATION_ICON,
        ToolSelection.AT_LEAST_TWO,
    ),
    _image_tool(
        "detector-pitch",
        "Detector pixel pitch...",
        "set_detector_pitch",
        "Set the detector pixel pitch of the selected images",
        MAGNIFICATION_ICON,
    ),
    PluginTool(
        id="geometry-calculator",
        title="Geometry calculator...",
        launcher="geometry_calculator",
        description="Magnification, unsharpness and optimal magnification",
        icon=MAGNIFICATION_ICON,
    ),
)
DISPLAY_TOOLS = (
    _image_tool(
        "radiographic-display",
        "Apply radiographic display",
        "apply_radiographic_display",
        "Gray scale, inversion and automatic window",
        DISPLAY_ICON,
    ),
    _image_tool(
        "film-convention",
        "Film convention...",
        "set_film_convention",
        "Display dense matter bright (DICOM images follow their attributes)",
        DISPLAY_ICON,
    ),
    _image_tool(
        "window",
        "Window...",
        "edit_window",
        "Percentiles, regions of interest, DICOM attributes, or level and width",
        DISPLAY_ICON,
    ),
)


class XRayRadiographyNDTWebPlugin(PluginBase):
    """X-ray radiography and NDT in DataLab-Web."""

    PLUGIN_INFO = PluginInfo(
        id=PLUGIN_ID,
        name=PLUGIN_NAME,
        version=__version__,
        description=PLUGIN_DESCRIPTION,
        icon=PLUGIN_ICON,
        capabilities=(
            PluginCapability.APPLICATION,
            PluginCapability.PROCESSING,
            PluginCapability.VISUALIZATION,
        ),
        documentation_url=DOCUMENTATION_URL,
    )
    RECIPES = RECIPES
    EXAMPLES = EXAMPLES
    TOOLS = (SIMULATOR_TOOL, *MAGNIFICATION_TOOLS, *DISPLAY_TOOLS)

    def __init__(self) -> None:
        super().__init__()
        #: Film convention of the session (preferences are not stored in the browser)
        self.film_convention = False

    # -- Registration -----------------------------------------------------

    def create_actions(self) -> None:
        """Register the processing features with their DataLab-Web menu paths."""
        processor = self.imagepanel.processor
        # Hosts older than the 1-to-0 support of DataLab-Web skip analyses
        supports_1_to_0 = callable(getattr(type(processor), "register_1_to_0", None))
        for feature in wfeatures.features_for(wfeatures.HOST_WEB):
            options = {
                "paramclass": feature.paramclass,
                "menu_path": feature.menu_path,
                "feature_id": feature.feature_id,
                "owner_plugin_id": self.plugin_id,
            }
            if feature.pattern == wfeatures.PATTERN_1_TO_1:
                processor.register_1_to_1(feature.function, feature.title, **options)
            elif feature.pattern == wfeatures.PATTERN_1_TO_0:
                if supports_1_to_0:
                    processor.register_1_to_0(
                        feature.function, feature.title, **options
                    )
            elif feature.pattern == wfeatures.PATTERN_2_TO_1:
                processor.register_2_to_1(
                    feature.function,
                    feature.title,
                    obj2_name=feature.obj2_name or "Operand",
                    **options,
                )
            else:
                processor.register_n_to_1(feature.function, feature.title, **options)

    # -- Examples and simulator -------------------------------------------

    @classmethod
    def materialize_example(cls, example_id: str) -> PluginExampleData | None:
        """Generate the simulated examples; real-data examples are packaged."""
        cls.get_example(example_id)
        return materialize_generated_example(example_id)

    def xray_simulator(self) -> RadiographySimulator:
        """Return a new radiography simulator."""
        return RadiographySimulator()

    # -- Helpers ----------------------------------------------------------

    def selected_images(self) -> list[ImageObj]:
        """Return the images selected when the running tool was launched."""
        return [obj for obj in self.get_selected_objects() if isinstance(obj, ImageObj)]

    async def edit(self, param: gds.DataSet) -> bool:
        """Edit a parameter set in a browser dialog."""
        return bool(await param.edit_async(self.main))

    async def apply(
        self, images: Sequence[ImageObj], function: Callable[[ImageObj], object]
    ) -> list[ImageObj]:
        """Apply an operation to images, reporting the failures."""
        done, errors = [], []
        for obj in images:
            try:
                function(obj)
            except ValueError as exc:
                errors.append(f"{obj.title}: {exc}")
            else:
                done.append(obj)
        if errors:
            await self.show_warning_async("\n".join(errors))
        return done

    # -- Magnification tools ----------------------------------------------

    async def define_magnification_value(self) -> list[ImageObj]:
        """Define a magnification of the selected images from its value."""
        param = wmag.MagnificationValueParam()
        if not await self.edit(param):
            return []
        return await self.apply(
            self.selected_images(), lambda obj: wmag.define_value(obj, param)
        )

    async def define_magnification_geometry(self) -> list[ImageObj]:
        """Define a magnification of the selected images from the distances."""
        param = wmag.MagnificationGeometryParam()
        if not await self.edit(param):
            return []
        return await self.apply(
            self.selected_images(), lambda obj: wmag.define_geometry(obj, param)
        )

    async def define_magnification_length(self) -> list[ImageObj]:
        """Define a magnification from a segment of known length."""
        images = self.selected_images()
        param = wmag.MagnificationLengthParam()
        segments = [
            item
            for item in images[0].get_graphical_annotations()
            if isinstance(item, SegmentAnnotation)
        ]
        if segments:
            segment = segments[-1]
            param.x0, param.y0, param.x1, param.y1 = (
                segment.x0,
                segment.y0,
                segment.x1,
                segment.y1,
            )
        if not await self.edit(param):
            return []
        return await self.apply(images, lambda obj: wmag.define_length(obj, param))

    async def define_magnification_ball(self) -> list[ImageObj]:
        """Define a magnification from a calibration ball in a region of interest."""
        param = wmag.MagnificationBallParam()
        if not await self.edit(param):
            return []
        return await self.apply(
            self.selected_images(), lambda obj: wmag.define_ball(obj, param)
        )

    async def define_magnifications_from_metadata(self) -> list[ImageObj]:
        """Define the magnifications given by the metadata of the selected images."""
        return await self.apply(self.selected_images(), wmag.define_from_metadata)

    async def set_active_magnification(self) -> list[ImageObj]:
        """Choose the active magnification of the selected images."""
        images = self.selected_images()
        param = wmag.MagnificationNameParam()
        names = objects.get_magnifications(images[0]).names
        if names:
            param.name = objects.get_magnifications(images[0]).active or names[0]
        if not await self.edit(param):
            return []
        return await self.apply(
            images, lambda obj: wmag.set_active_magnification(obj, param.name)
        )

    async def copy_magnifications(self) -> list[ImageObj]:
        """Copy the magnifications of the first selected image to the others."""
        source, *targets = self.selected_images()
        try:
            wmag.copy_magnifications(source, targets)
        except ValueError as exc:
            await self.show_warning_async(str(exc))
            return []
        return targets

    async def set_detector_pitch(self) -> list[ImageObj]:
        """Set the detector pixel pitch of the selected images."""
        images = self.selected_images()
        param = wmag.detector_pitch_param(images[0], None)
        if not await self.edit(param):
            return []
        return await self.apply(images, lambda obj: wmag.set_detector_pitch(obj, param))

    async def geometry_calculator(self) -> list[tuple[str, str]]:
        """Show the magnification and unsharpness of an acquisition geometry."""
        param = wmag.GeometryCalculatorParam()
        if not await self.edit(param):
            return []
        try:
            report = wmag.geometry_report(param)
        except ValueError as exc:
            await self.show_warning_async(str(exc))
            return []
        await self.show_info_async(
            "\n".join(f"{name}: {value}" for name, value in report)
        )
        return report

    # -- Display tools ----------------------------------------------------

    @staticmethod
    def clear_host_display(obj: ImageObj) -> None:
        """Remove the display choices made in DataLab-Web, for the plugin display.

        DataLab-Web stores them under its own keys, which take precedence over
        the display options of the image.
        """
        for key in DATALAB_WEB_DISPLAY_KEYS:
            obj.metadata.pop(key, None)

    async def apply_radiographic_display(self) -> list[ImageObj]:
        """Apply the radiographic display to the selected images."""

        def apply(obj: ImageObj) -> None:
            self.clear_host_display(obj)
            wdisplay.apply_profile(obj, self.film_convention)

        return await self.apply(self.selected_images(), apply)

    async def set_film_convention(self) -> list[ImageObj]:
        """Choose the film convention and apply it to the selected images."""
        param = wparams.InversionParam()
        param.film_convention = self.film_convention
        if not await self.edit(param):
            return []
        self.film_convention = bool(param.film_convention)

        def invert(obj: ImageObj) -> None:
            for key in ("invert_colormap", "colormap_inverted"):
                obj.metadata.pop(key, None)
            obj.set_metadata_option(
                "invert_colormap", wdisplay.inversion_for(obj, self.film_convention)
            )

        return await self.apply(self.selected_images(), invert)

    async def edit_window(self) -> list[ImageObj]:
        """Edit the window of the selected images."""
        param = wparams.WindowParam()
        if not await self.edit(param):
            return []

        def window(obj: ImageObj) -> None:
            obj.metadata.pop(DATALAB_WEB_LUT_KEY, None)
            wparams.apply_window_param(obj, param)

        return await self.apply(self.selected_images(), window)


def get_web_manifest() -> dict[str, object]:
    """Return the browser bundle and compatibility contract of the plugin."""
    return {
        "plugin_id": PLUGIN_ID,
        "plugin_version": __version__,
        "web_status": WEB_STATUS,
        "datalab_web_version": DATALAB_WEB_VERSION,
        "pyodide_version": PYODIDE_VERSION,
        "recipes": [
            {"recipe_id": recipe.recipe_id, "recipe_version": recipe.version}
            for recipe in RECIPES
        ],
        "examples": [example.id for example in EXAMPLES],
        "tools": [tool.id for tool in XRayRadiographyNDTWebPlugin.TOOLS],
        "features": [
            feature.feature_id for feature in wfeatures.features_for(wfeatures.HOST_WEB)
        ],
    }


__all__ = [
    "DATALAB_WEB_VERSION",
    "PYODIDE_VERSION",
    "WEB_STATUS",
    "XRayRadiographyNDTWebPlugin",
    "get_web_manifest",
]
