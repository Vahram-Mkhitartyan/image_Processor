"""Shared constants and deterministic geometry helpers for ScribeTrace."""

import re

EXPERT_NAME = "scribetrace"
SUPPORTED_THRESHOLD_MODES = {"auto", "binary", "fixed", "otsu"}
NEIGHBOR_OFFSETS = (
    (-1, -1), (0, -1), (1, -1),
    (-1, 0), (1, 0),
    (-1, 1), (0, 1), (1, 1),
)


def coordinate_key(point):
    """Return the deterministic document-order key for a point."""
    return (point.y, point.x)


def sanitize_identifier(value):
    """Convert an arbitrary unit identifier into a safe filename fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown_unit"))
    return cleaned.strip("._") or "unknown_unit"


def edge_key(point_a, point_b):
    """Return a stable undirected key for an edge between two points."""
    return tuple(sorted((point_a.to_tuple(), point_b.to_tuple())))
