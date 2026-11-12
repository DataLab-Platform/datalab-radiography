"""Headless recipe registry."""

from __future__ import annotations

from datalab.plugins.recipes import RecipeDescriptor

from .correction_recipes import OFFSET_GAIN_RECIPE, STEP_WEDGE_RECIPE
from .ndt_recipes import DUPLEX_RECIPE, INDICATION_RECIPE, WALL_THICKNESS_RECIPE

RECIPES: tuple[RecipeDescriptor, ...] = (
    OFFSET_GAIN_RECIPE,
    STEP_WEDGE_RECIPE,
    DUPLEX_RECIPE,
    WALL_THICKNESS_RECIPE,
    INDICATION_RECIPE,
)

__all__ = [
    "DUPLEX_RECIPE",
    "INDICATION_RECIPE",
    "OFFSET_GAIN_RECIPE",
    "RECIPES",
    "STEP_WEDGE_RECIPE",
    "WALL_THICKNESS_RECIPE",
]
