"""Headless recipe registry."""

from __future__ import annotations

from datalab.plugins.recipes import RecipeDescriptor

from .correction_recipes import OFFSET_GAIN_RECIPE, STEP_WEDGE_RECIPE

RECIPES: tuple[RecipeDescriptor, ...] = (OFFSET_GAIN_RECIPE, STEP_WEDGE_RECIPE)

__all__ = ["OFFSET_GAIN_RECIPE", "RECIPES", "STEP_WEDGE_RECIPE"]
