"""Normalized symbolic geometry record produced from ScribeTrace output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .constants import UNKNOWN_ID

@dataclass
class ScriLogSignature:
    """
    Raw structural signature extracted from ScribeTrace output.

    This is NOT the logic result yet.

    Think of it as:

        ScribeTrace says:
            - this glyph has 1 loop
            - 2 endpoints
            - 0 junctions
            - right-side exit
            - selected reconstruction phase = 2

        ScriLogSignature stores those measured facts.

    Later:
        ScriLogFactBase converts this signature into facts.
        ScriLogEngine derives symbolic families from those facts.
    """

    # --------------------------------------------------------
    # Identity
    # --------------------------------------------------------

    unit_id: str = UNKNOWN_ID
    selected_hypothesis_id: str = UNKNOWN_ID

    # --------------------------------------------------------
    # Core topology
    # --------------------------------------------------------

    loop_count: int = 0
    endpoint_count: int = 0
    junction_count: int = 0
    path_count: int = 0
    component_count: int = 0

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    width: float = 0.0
    height: float = 0.0
    aspect_ratio: float = 0.0
    ink_pixels: int = 0

    # --------------------------------------------------------
    # Structural flags
    # --------------------------------------------------------

    has_ascender: bool = False
    has_descender: bool = False

    has_left_exit: bool = False
    has_right_exit: bool = False

    has_top_contact: bool = False
    has_bottom_contact: bool = False

    # --------------------------------------------------------
    # Zone maps
    # --------------------------------------------------------

    endpoint_zones: Dict[str, int] = field(default_factory=dict)
    junction_zones: Dict[str, int] = field(default_factory=dict)
    loop_zones: Dict[str, int] = field(default_factory=dict)

    # --------------------------------------------------------
    # Direction features
    # --------------------------------------------------------

    direction_ratios: Dict[str, float] = field(default_factory=dict)

    # --------------------------------------------------------
    # Reconstruction metadata
    # --------------------------------------------------------

    reconstruction_phase: int = 0

    line_removal_applied: bool = False

    downstream_bridge_required: bool = False
    downstream_bridge_completed: bool = False

    # --------------------------------------------------------
    # Dataset / source metadata
    # --------------------------------------------------------

    source_kind: str = "unknown"

    # Useful for debugging adapter compatibility.
    raw_keys_seen: List[str] = field(default_factory=list)

    @property
    def is_reconstructed(self) -> bool:
        """
        True if this came from a non-h0 reconstruction phase.
        """
        return self.reconstruction_phase > 0

    @property
    def is_wide(self) -> bool:
        """
        Wide glyphs have more horizontal spread than vertical height.
        """
        return self.aspect_ratio >= 1.35

    @property
    def is_tall(self) -> bool:
        """
        Tall glyphs have much more height than width.
        """
        return self.aspect_ratio > 0.0 and self.aspect_ratio <= 0.75

    @property
    def is_compact(self) -> bool:
        """
        Compact glyphs are roughly balanced.
        """
        return 0.75 < self.aspect_ratio < 1.35

    @property
    def bridge_state(self) -> str:
        """
        Human-readable bridge/reconstruction state.

        This is important because ScriLog must flag unsafe geometry.
        """
        if not self.downstream_bridge_required:
            return "not_required"

        if self.downstream_bridge_completed:
            return "required_completed"

        return "required_incomplete"

    def to_dict(self) -> Dict[str, Any]:
        """
        JSON-safe export.
        """

        return {
            "unit_id": self.unit_id,
            "selected_hypothesis_id": self.selected_hypothesis_id,

            "loop_count": self.loop_count,
            "endpoint_count": self.endpoint_count,
            "junction_count": self.junction_count,
            "path_count": self.path_count,
            "component_count": self.component_count,

            "width": self.width,
            "height": self.height,
            "aspect_ratio": self.aspect_ratio,
            "ink_pixels": self.ink_pixels,

            "has_ascender": self.has_ascender,
            "has_descender": self.has_descender,

            "has_left_exit": self.has_left_exit,
            "has_right_exit": self.has_right_exit,

            "has_top_contact": self.has_top_contact,
            "has_bottom_contact": self.has_bottom_contact,

            "endpoint_zones": dict(self.endpoint_zones),
            "junction_zones": dict(self.junction_zones),
            "loop_zones": dict(self.loop_zones),

            "direction_ratios": dict(self.direction_ratios),

            "reconstruction_phase": self.reconstruction_phase,
            "line_removal_applied": self.line_removal_applied,

            "downstream_bridge_required": self.downstream_bridge_required,
            "downstream_bridge_completed": self.downstream_bridge_completed,
            "bridge_state": self.bridge_state,

            "source_kind": self.source_kind,
            "raw_keys_seen": list(self.raw_keys_seen),
        }

