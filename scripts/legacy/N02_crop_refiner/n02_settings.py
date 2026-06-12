"""Settings helpers for N02 crop refinement."""

from dataclasses import asdict, dataclass

from n02_io import load_json


@dataclass
class RefinerSettings:
    """Configuration values used by the rebuilt N02 crop refiner.

    Args:
        input_group_mode: How to interpret scribemap groups for refinement.
        scribemap_2_layers_to_refine: Scribemap layers to use for refinement.
        accepted_score_threshold: Score needed to send a crop forward normally.
        review_score_threshold: Minimum useful score before rejection.
        line_bucket_max_center_y_distance: Max y-center distance for same row.
        line_bucket_min_vertical_overlap: Minimum overlap for same row.
        text_unit_max_horizontal_gap_px: Max left/right gap for merging.
        text_unit_min_vertical_overlap: Min y-overlap for text-unit merging.
        text_unit_max_center_y_distance: Max y-center gap for merging.
        text_unit_max_height_ratio: Max relative height difference for merging.
        text_unit_max_merged_width: Hard width cap for one OCR text unit.
        text_unit_max_merged_height: Hard height cap for one OCR text unit.
        text_unit_max_merged_area: Hard area cap for one OCR text unit.
        text_unit_max_groups_per_cluster: Max source boxes per text unit.
        border_profile_band_px: Number of outer border pixels to scan.
        border_profile_min_aligned_pixels: Minimum matching 0/1 border pixels.
        border_profile_tolerance_px: Allowed coordinate mismatch between profiles.
        border_profile_max_horizontal_gap_px: Max x-gap for left/right profile merge.
        border_profile_max_vertical_gap_px: Max y-gap for top/bottom profile merge.
        border_profile_min_overlap_ratio: Required perpendicular overlap.
        crop_padding_px: Small final padding around accepted text units.
        debug_preview_enabled: Whether to write visual inspection previews.
        ink_threshold: Pixel values below this are treated as ink.
        edge_density_high: High edge density can mean the crop clips text.
        center_density_low: Very low center density can mean empty/dead crop.
        min_ink_density: Low total ink density can mean non-text or empty crop.
        max_ink_density: Very high total density can mean dense artifact.

    Returns:
        RefinerSettings instance.
    """

    input_group_mode: str = "legacy_groups"
    scribemap_2_layers_to_refine: tuple = ("blue",)
    accepted_score_threshold: float = 85.0
    review_score_threshold: float = 60.0

    line_bucket_max_center_y_distance: int = 12
    line_bucket_min_vertical_overlap: float = 0.35

    text_unit_max_horizontal_gap_px: int = 32
    text_unit_min_vertical_overlap: float = 0.45
    text_unit_max_center_y_distance: int = 10
    text_unit_max_height_ratio: float = 2.25
    text_unit_max_merged_width: int = 420
    text_unit_max_merged_height: int = 90
    text_unit_max_merged_area: int = 32000
    text_unit_max_groups_per_cluster: int = 4

    border_profile_band_px: int = 1
    border_profile_min_aligned_pixels: int = 2
    border_profile_tolerance_px: int = 2
    border_profile_max_horizontal_gap_px: int = 18
    border_profile_max_vertical_gap_px: int = 24
    border_profile_min_overlap_ratio: float = 0.20

    crop_padding_px: int = 2
    debug_preview_enabled: bool = True

    ink_threshold: int = 180
    edge_density_high: float = 0.22
    center_density_low: float = 0.03
    min_ink_density: float = 0.005
    max_ink_density: float = 0.75

    def to_dict(self):
        """Return settings as a plain dictionary.

        Args:
            None.

        Returns:
            Dictionary version of this settings object.
        """
        return asdict(self)


def coerce_settings(settings):
    """Normalize user-provided settings into a RefinerSettings object.

    Args:
        settings: None, RefinerSettings, or dictionary of overrides.

    Returns:
        RefinerSettings instance.
    """
    if settings is None:
        return RefinerSettings()

    if isinstance(settings, RefinerSettings):
        return settings

    if isinstance(settings, dict):
        allowed = set(RefinerSettings.__dataclass_fields__.keys())
        clean = {
            key: value
            for key, value in settings.items()
            if key in allowed
        }
        return RefinerSettings(**clean)

    raise TypeError("settings must be None, dict, or RefinerSettings")


def load_refiner_settings(settings_path):
    """Load refiner settings from JSON with unknown-key tolerance.

    Args:
        settings_path: Path to a JSON settings file.

    Returns:
        RefinerSettings instance.
    """
    return coerce_settings(load_json(settings_path))
