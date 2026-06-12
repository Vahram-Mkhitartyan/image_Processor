"""Validated ScribeTrace behavior settings."""

from .trace_common import SUPPORTED_THRESHOLD_MODES

class TraceSettings:
    """Store and validate all behavior controls used by ScribeTrace."""

    def __init__(
        self,
        enabled=True,
        save_debug=True,
        save_json=True,
        debug_draw_labels=False,
        enable_mask_repair=True,
        ink_threshold_mode="otsu",
        fixed_threshold_value=128,
        minimum_ink_pixels=5,
        maximum_component_count_for_full_trace=50,
        minimum_trace_path_points=8,
        short_path_merge_max_angle_degrees=35.0,
        short_path_tangent_points=3,
        short_path_merge_min_advantage_degrees=10.0,
        local_extrema_min_prominence=2,
        local_extrema_min_spacing=3,
        
    ):
        self.enabled = bool(enabled)
        self.save_debug = bool(save_debug)
        self.save_json = bool(save_json)
        self.debug_draw_labels = bool(debug_draw_labels)
        self.enable_mask_repair = bool(enable_mask_repair)
        self.ink_threshold_mode = str(ink_threshold_mode).strip().lower()
        self.fixed_threshold_value = int(fixed_threshold_value)
        self.minimum_ink_pixels = int(minimum_ink_pixels)
        self.maximum_component_count_for_full_trace = int(
            maximum_component_count_for_full_trace
        )
        self.minimum_trace_path_points = int(minimum_trace_path_points)
        self.short_path_merge_max_angle_degrees = float(
            short_path_merge_max_angle_degrees
        )
        self.short_path_tangent_points = int(short_path_tangent_points)
        self.short_path_merge_min_advantage_degrees = float(
            short_path_merge_min_advantage_degrees
        )
        self.local_extrema_min_prominence = int(local_extrema_min_prominence)
        self.local_extrema_min_spacing = int(local_extrema_min_spacing)

        self.validate()

    @classmethod
    def from_dict(cls, settings):
        """Build settings from a possibly partial dictionary."""
        settings = settings or {}
        return cls(
            enabled=settings.get("enabled", True),
            save_debug=settings.get("save_debug", True),
            save_json=settings.get("save_json", True),
            debug_draw_labels=settings.get("debug_draw_labels", False),
            enable_mask_repair=settings.get("enable_mask_repair", True),
            ink_threshold_mode=settings.get("ink_threshold_mode", "otsu"),
            fixed_threshold_value=settings.get("fixed_threshold_value", 128),
            minimum_ink_pixels=settings.get("minimum_ink_pixels", 5),
            maximum_component_count_for_full_trace=settings.get(
                "maximum_component_count_for_full_trace", 50
            ),
            minimum_trace_path_points=settings.get("minimum_trace_path_points", 8),
            short_path_merge_max_angle_degrees=settings.get(
                "short_path_merge_max_angle_degrees", 35.0
            ),
            short_path_tangent_points=settings.get("short_path_tangent_points", 3),
            short_path_merge_min_advantage_degrees=settings.get(
                "short_path_merge_min_advantage_degrees", 10.0
            ),
            local_extrema_min_prominence=settings.get("local_extrema_min_prominence", 2),
            local_extrema_min_spacing=settings.get("local_extrema_min_spacing", 3),
        )

    def validate(self):
        """Reject unsupported modes and unsafe numeric ranges."""
        if self.ink_threshold_mode not in SUPPORTED_THRESHOLD_MODES:
            modes = ", ".join(sorted(SUPPORTED_THRESHOLD_MODES))
            raise ValueError(
                f"Unsupported ink_threshold_mode {self.ink_threshold_mode!r}; "
                f"expected one of: {modes}."
            )
        if not 0 <= self.fixed_threshold_value <= 255:
            raise ValueError("fixed_threshold_value must be between 0 and 255.")
        if self.minimum_ink_pixels < 1:
            raise ValueError("minimum_ink_pixels must be at least 1.")
        if self.maximum_component_count_for_full_trace < 1:
            raise ValueError(
                "maximum_component_count_for_full_trace must be at least 1."
            )
        if self.minimum_trace_path_points < 1:
            raise ValueError("minimum_trace_path_points must be at least 1.")
        if not 0 <= self.short_path_merge_max_angle_degrees <= 180:
            raise ValueError(
                "short_path_merge_max_angle_degrees must be between 0 and 180."
            )
        if self.short_path_tangent_points < 2:
            raise ValueError("short_path_tangent_points must be at least 2.")
        if self.short_path_merge_min_advantage_degrees < 0:
            raise ValueError(
                "short_path_merge_min_advantage_degrees cannot be negative."
            )
        if self.local_extrema_min_prominence < 1:
            raise ValueError("local_extrema_min_prominence must be at least 1.")

        if self.local_extrema_min_spacing < 1:
            raise ValueError("local_extrema_min_spacing must be at least 1.")


    def to_dict(self):
        """Return JSON-ready settings metadata."""
        return {
            "enabled": self.enabled,
            "save_debug": self.save_debug,
            "save_json": self.save_json,
            "debug_draw_labels": self.debug_draw_labels,
            "enable_mask_repair": self.enable_mask_repair,
            "ink_threshold_mode": self.ink_threshold_mode,
            "fixed_threshold_value": self.fixed_threshold_value,
            "minimum_ink_pixels": self.minimum_ink_pixels,
            "maximum_component_count_for_full_trace":
                self.maximum_component_count_for_full_trace,
            "minimum_trace_path_points": self.minimum_trace_path_points,
            "short_path_merge_max_angle_degrees":
                self.short_path_merge_max_angle_degrees,
            "short_path_tangent_points": self.short_path_tangent_points,
            "short_path_merge_min_advantage_degrees":
                self.short_path_merge_min_advantage_degrees,
            "local_extrema_min_prominence": self.local_extrema_min_prominence,
            "local_extrema_min_spacing": self.local_extrema_min_spacing,
        }




def normalize_trace_settings(settings=None):
    """Normalize a settings object or dictionary into TraceSettings."""
    if isinstance(settings, TraceSettings):
        settings.validate()
        return settings
    if isinstance(settings, dict):
        return TraceSettings.from_dict(settings)
    if settings is None:
        return TraceSettings()
    raise ValueError("Unsupported settings type.")
