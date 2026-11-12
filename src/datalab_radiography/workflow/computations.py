"""Sigima computation functions of the radiography plugin.

Each function is registered as a processing feature of DataLab Desktop and
DataLab-Web (see :mod:`.features`). Its ``operation_id`` and
``contract_version`` identify its result in DataLab provenance: increase the
contract version whenever a change alters the result.
"""

from __future__ import annotations

from sigima.objects import ImageObj
from sigima.proc.contracts import EXACT, ReplayQualification
from sigima.proc.decorator import computation_function
from sigima.proc.image.base import dst_1_to_1

from .. import PLUGIN_ID
from ..core import magnification as mag
from ..core import metadata as md
from . import objects
from .magnification import ExpressCoordinatesParam

OPERATION_PREFIX = PLUGIN_ID


def operation_id(name: str) -> str:
    """Return the operation identifier of a computation function."""
    return f"{OPERATION_PREFIX}.{name}"


_FRAME_METADATA = (
    md.MAGNIFICATIONS_KEY,
    md.COORDINATE_FRAME_KEY,
    md.DETECTOR_PITCH_KEY,
)


@computation_function(
    operation_id=operation_id("express_coordinates"),
    contract_version=1,
    qualification=ReplayQualification(contract_version=1, comparison=EXACT),
    reads_metadata={"src": _FRAME_METADATA},
)
def express_coordinates(src: ImageObj, p: ExpressCoordinatesParam) -> ImageObj:
    """Express the image coordinates in an object plane, the detector plane or pixels.

    The data are unchanged. ROIs and annotations keep their position on the
    image, and the magnifications stay valid.

    Args:
        src: source image
        p: target coordinate frame

    Returns:
        Image with converted coordinates
    """
    frame = objects.get_coordinate_frame(src)
    pitch = objects.get_detector_pitch(src)
    mset = objects.check_magnifications(src)
    item = None
    if p.target == mag.FRAME_OBJECT:
        name = (p.magnification or "").strip() or mset.active
        if name is None:
            raise mag.MagnificationError("The image has no magnification")
        item = mset.get(name)
    unit = None if p.target == mag.FRAME_PIXEL else p.unit
    (fx, fy), new_frame = mag.express_factors(frame, pitch, p.target, unit, item)
    dst = dst_1_to_1(src, "express_coordinates", new_frame.describe())
    objects.scale_coordinates(dst, fx, fy)
    dst.xunit = dst.yunit = new_frame.unit or ""
    objects.set_coordinate_frame(dst, new_frame)
    if len(mset):
        objects.set_magnifications(dst, mset.rescaled(fx, fy))
    return dst
