"""Aristotel degradation-teacher package."""

from .recipes import DamageRecipe, build_default_recipes
from .runner import AristotelRunner, FileCorrupter

__all__ = [
    "AristotelRunner",
    "DamageRecipe",
    "FileCorrupter",
    "build_default_recipes",
]
