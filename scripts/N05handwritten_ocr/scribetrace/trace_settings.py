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
        enable_theoretical_reconstruction=False,
        reconstruction_max_hypotheses=5,
        reconstruction_max_accepted=3,
        reconstruction_max_bridge_length_px=12.0,
        reconstruction_max_bridge_angle_degrees=45.0,
        reconstruction_tangent_points=4,
        reconstruction_min_endpoint_separation_px=2.0,
        reconstruction_min_topology_gain=0.08,
        reconstruction_min_acceptance_score=0.58,
        reconstruction_max_added_ink_ratio=0.12,
        reconstruction_confidence_weight=0.20,
        reconstruction_topology_weight=0.55,
        reconstruction_geometry_weight=0.25,
        reconstruction_use_recognition_verification=True,
        reconstruction_min_confidence_gain=0.0,
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
        self.enable_theoretical_reconstruction = bool(
            enable_theoretical_reconstruction
        )
        self.reconstruction_max_hypotheses = int(
            reconstruction_max_hypotheses
        )
        self.reconstruction_max_accepted = int(reconstruction_max_accepted)
        self.reconstruction_max_bridge_length_px = float(
            reconstruction_max_bridge_length_px
        )
        self.reconstruction_max_bridge_angle_degrees = float(
            reconstruction_max_bridge_angle_degrees
        )
        self.reconstruction_tangent_points = int(
            reconstruction_tangent_points
        )
        self.reconstruction_min_endpoint_separation_px = float(
            reconstruction_min_endpoint_separation_px
        )
        self.reconstruction_min_topology_gain = float(
            reconstruction_min_topology_gain
        )
        self.reconstruction_min_acceptance_score = float(
            reconstruction_min_acceptance_score
        )
        self.reconstruction_max_added_ink_ratio = float(
            reconstruction_max_added_ink_ratio
        )
        self.reconstruction_confidence_weight = float(
            reconstruction_confidence_weight
        )
        self.reconstruction_topology_weight = float(
            reconstruction_topology_weight
        )
        self.reconstruction_geometry_weight = float(
            reconstruction_geometry_weight
        )
        self.reconstruction_use_recognition_verification = bool(
            reconstruction_use_recognition_verification
        )
        self.reconstruction_min_confidence_gain = float(
            reconstruction_min_confidence_gain
        )

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
            enable_theoretical_reconstruction=settings.get(
                "enable_theoretical_reconstruction",
                False,
            ),
            reconstruction_max_hypotheses=settings.get(
                "reconstruction_max_hypotheses",
                5,
            ),
            reconstruction_max_accepted=settings.get(
                "reconstruction_max_accepted",
                3,
            ),
            reconstruction_max_bridge_length_px=settings.get(
                "reconstruction_max_bridge_length_px",
                12.0,
            ),
            reconstruction_max_bridge_angle_degrees=settings.get(
                "reconstruction_max_bridge_angle_degrees",
                45.0,
            ),
            reconstruction_tangent_points=settings.get(
                "reconstruction_tangent_points",
                4,
            ),
            reconstruction_min_endpoint_separation_px=settings.get(
                "reconstruction_min_endpoint_separation_px",
                2.0,
            ),
            reconstruction_min_topology_gain=settings.get(
                "reconstruction_min_topology_gain",
                0.08,
            ),
            reconstruction_min_acceptance_score=settings.get(
                "reconstruction_min_acceptance_score",
                0.58,
            ),
            reconstruction_max_added_ink_ratio=settings.get(
                "reconstruction_max_added_ink_ratio",
                0.12,
            ),
            reconstruction_confidence_weight=settings.get(
                "reconstruction_confidence_weight",
                0.20,
            ),
            reconstruction_topology_weight=settings.get(
                "reconstruction_topology_weight",
                0.55,
            ),
            reconstruction_geometry_weight=settings.get(
                "reconstruction_geometry_weight",
                0.25,
            ),
            reconstruction_use_recognition_verification=settings.get(
                "reconstruction_use_recognition_verification",
                True,
            ),
            reconstruction_min_confidence_gain=settings.get(
                "reconstruction_min_confidence_gain",
                0.0,
            ),
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
        if self.reconstruction_max_hypotheses < 1:
            raise ValueError("reconstruction_max_hypotheses must be at least 1.")
        if self.reconstruction_max_accepted < 1:
            raise ValueError("reconstruction_max_accepted must be at least 1.")
        if self.reconstruction_max_accepted > self.reconstruction_max_hypotheses:
            raise ValueError(
                "reconstruction_max_accepted cannot exceed "
                "reconstruction_max_hypotheses."
            )
        if self.reconstruction_max_bridge_length_px <= 0:
            raise ValueError(
                "reconstruction_max_bridge_length_px must be positive."
            )
        if not 0 <= self.reconstruction_max_bridge_angle_degrees <= 90:
            raise ValueError(
                "reconstruction_max_bridge_angle_degrees must be between 0 and 90."
            )
        if self.reconstruction_tangent_points < 2:
            raise ValueError("reconstruction_tangent_points must be at least 2.")
        if self.reconstruction_min_endpoint_separation_px < 1:
            raise ValueError(
                "reconstruction_min_endpoint_separation_px must be at least 1."
            )
        for name in (
            "reconstruction_min_topology_gain",
            "reconstruction_min_acceptance_score",
            "reconstruction_max_added_ink_ratio",
            "reconstruction_confidence_weight",
            "reconstruction_topology_weight",
            "reconstruction_geometry_weight",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1.")
        weight_sum = (
            self.reconstruction_confidence_weight
            + self.reconstruction_topology_weight
            + self.reconstruction_geometry_weight
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError("Reconstruction score weights must sum to 1.0.")
        if not -1 <= self.reconstruction_min_confidence_gain <= 1:
            raise ValueError(
                "reconstruction_min_confidence_gain must be between -1 and 1."
            )


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
            "enable_theoretical_reconstruction":
                self.enable_theoretical_reconstruction,
            "reconstruction_max_hypotheses":
                self.reconstruction_max_hypotheses,
            "reconstruction_max_accepted":
                self.reconstruction_max_accepted,
            "reconstruction_max_bridge_length_px":
                self.reconstruction_max_bridge_length_px,
            "reconstruction_max_bridge_angle_degrees":
                self.reconstruction_max_bridge_angle_degrees,
            "reconstruction_tangent_points":
                self.reconstruction_tangent_points,
            "reconstruction_min_endpoint_separation_px":
                self.reconstruction_min_endpoint_separation_px,
            "reconstruction_min_topology_gain":
                self.reconstruction_min_topology_gain,
            "reconstruction_min_acceptance_score":
                self.reconstruction_min_acceptance_score,
            "reconstruction_max_added_ink_ratio":
                self.reconstruction_max_added_ink_ratio,
            "reconstruction_confidence_weight":
                self.reconstruction_confidence_weight,
            "reconstruction_topology_weight":
                self.reconstruction_topology_weight,
            "reconstruction_geometry_weight":
                self.reconstruction_geometry_weight,
            "reconstruction_use_recognition_verification":
                self.reconstruction_use_recognition_verification,
            "reconstruction_min_confidence_gain":
                self.reconstruction_min_confidence_gain,
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
