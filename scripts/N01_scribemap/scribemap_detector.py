import cv2

from scribemap_io import load_image, ensure_output_folders, save_image, save_json
from scribemap_components import detect_micro_components
from scribemap_grouping import build_component_groups, filter_line_like_groups
from scribemap_crops import crop_components, crop_groups
from scribemap_preview import (
    normalize_to_grayscale,
    draw_components_preview,
    draw_groups_preview,
    draw_rejected_groups_preview,
    draw_line_masks_preview,
)


class ScribeMapBWDetector:
    """
    Orchestrator for Node 01.

    Responsibilities:
    - load prepared artifacts from Node 00
    - run component extraction and grouping helpers
    - optionally save crops
    - render debug previews
    - persist metadata JSON
    """

    def __init__(self, settings=None):
        """Initialize the object and store configuration.
        
        Args:
            settings: Optional configuration dictionary used to override defaults.
        
        Returns:
            None.
        """
        if settings is None:
            settings = {}
        self.settings = settings

    def _load_prepared_inputs(self, prepared_bw_image_path, content_ink_mask_path):
        # Node 00 produces both:
        # - prepared_bw_image_path: grayscale/bw image for crop visualization
        # - content_ink_mask_path: ink-only mask used for connected components
        """Load the prepared image and content mask artifacts.
        
        Args:
            prepared_bw_image_path: Path to the prepared crop image from node 00.
            content_ink_mask_path: Path to the content-ink mask from node 00.
        
        Returns:
            Tuple of grayscale input image and content mask.
        """
        bw_input = load_image(prepared_bw_image_path)
        gray_input = normalize_to_grayscale(bw_input)

        content_ink_mask = load_image(content_ink_mask_path)
        content_ink_mask = normalize_to_grayscale(content_ink_mask)

        return gray_input, content_ink_mask

    def _build_groups_from_content_mask(self, content_ink_mask):
        # Core math pipeline delegated to helper modules.
        """Detect components and groups from the content mask.
        
        Args:
            content_ink_mask: Binary content-ink mask produced by file preparation.
        
        Returns:
            Tuple of components, raw groups, kept groups, and rejected groups.
        """
        components = detect_micro_components(content_ink_mask, self.settings)
        raw_groups = build_component_groups(components, self.settings)
        groups, rejected_groups = filter_line_like_groups(raw_groups, self.settings)

        return components, raw_groups, groups, rejected_groups
    
    def _build_groups_for_layer(self, layer_mask, layer_name):
        """Run the normal ScribeMap grouping pipeline on one color layer mask.

        Args:
            layer_mask: Binary layer mask.
            layer_name: Name of the layer, e.g. blue, red, black.

        Returns:
            Dictionary with components, raw groups, groups, and rejected groups.
        """
        components, raw_groups, groups, rejected_groups = self._build_groups_from_content_mask(
            layer_mask
        )

        for component in components:
            component["layer"] = layer_name

        for group in raw_groups:
            group["layer"] = layer_name
            group["group_uid"] = f"{layer_name}_raw_{group['group_id']:04d}"

        for group in groups:
            group["layer"] = layer_name
            group["group_uid"] = f"{layer_name}_{group['group_id']:04d}"

        for group in rejected_groups:
            group["layer"] = layer_name
            group["group_uid"] = f"{layer_name}_rejected_{group['group_id']:04d}"

        return {
            "layer": layer_name,
            "component_count": len(components),
            "raw_group_count": len(raw_groups),
            "group_count": len(groups),
            "rejected_group_count": len(rejected_groups),
            "components": components,
            "raw_groups": raw_groups,
            "groups": groups,
            "rejected_groups": rejected_groups,
        }

    def _build_crop_outputs(self, gray_input, components, groups, folders):
        # Caches human-inspectable crops for downstream review/model diagnostics.
        """Create optional component and group crop outputs.
        
        Args:
            gray_input: Value used by this function.
            components: List of detected micro-component dictionaries.
            groups: List of accepted group dictionaries.
            folders: Dictionary of output folder paths.
        
        Returns:
            Tuple of component output metadata and group output metadata.
        """
        if self.settings.get("save_component_crops", False):
            components_output = crop_components(
                gray_input,
                components,
                folders["components_dir"],
                self.settings.get("component_margin", 4),
                save_image,
            )
        else:
            components_output = components

        if self.settings.get("save_group_crops", True):
            groups_output = crop_groups(
                gray_input,
                groups,
                folders["groups_dir"],
                self.settings.get("group_crop_margin", 10),
                save_image,
            )
        else:
            groups_output = groups

        return components_output, groups_output

    def _build_preview_artifacts(
        self,
        gray_input,
        content_ink_mask,
        components,
        raw_groups,
        groups,
        rejected_groups,
        folders,
    ):
        # Visual debugging outputs:
        # - component boxes
        # - raw groups (before rejection)
        # - filtered groups (final candidates)
        # - rejected groups with reject_reason labels
        """Render ScribeMap preview images for debugging.
        
        Args:
            gray_input: Value used by this function.
            content_ink_mask: Binary content-ink mask produced by file preparation.
            components: List of detected micro-component dictionaries.
            raw_groups: List of groups before final filtering.
            groups: List of accepted group dictionaries.
            rejected_groups: List of rejected group dictionaries.
            folders: Dictionary of output folder paths.
            black_ink_mask_path: Optional path to the black-ink mask artifact.
            recovered_blue_dark_mask_path: Optional path to the recovered-blue-dark mask artifact.
            recovered_red_dark_mask_path: Optional path to the recovered-red-dark mask artifact.
            recovered_green_dark_mask_path: Optional path to the recovered-green-dark mask artifact.
            preparation_artifacts: Dictionary of preparation artifacts.
        Returns:
            Dictionary of saved preview artifact paths.
        """
        components_preview = draw_components_preview(gray_input, components)
        raw_groups_preview = draw_groups_preview(gray_input, raw_groups)
        filtered_groups_preview = draw_groups_preview(gray_input, groups)
        rejected_groups_preview = draw_rejected_groups_preview(gray_input, rejected_groups)

        artifacts = {
            "bw_input": save_image(gray_input, f"{folders['debug_dir']}/00_bw_input.jpeg"),
            "content_ink_mask": save_image(content_ink_mask, f"{folders['debug_dir']}/01_content_ink_mask.jpeg"),
            "components_preview": save_image(components_preview, f"{folders['debug_dir']}/02_components_preview.jpeg"),
            "raw_groups_preview": save_image(raw_groups_preview, f"{folders['debug_dir']}/03_raw_groups_preview.jpeg"),
            "filtered_groups_preview": save_image(filtered_groups_preview, f"{folders['debug_dir']}/04_filtered_groups_preview.jpeg"),
            "rejected_groups_preview": save_image(rejected_groups_preview, f"{folders['debug_dir']}/05_rejected_groups_preview.jpeg"),
        }

        return artifacts

    def _add_line_masks_preview(self, artifacts, gray_input, mask_paths, folders):
        # Optional: only available when Node 00 artifacts include line masks.
        """Add the optional line-mask overlay preview.
        
        Args:
            artifacts: Value used by this function.
            gray_input: Value used by this function.
            mask_paths: Dictionary of optional mask artifact paths.
            folders: Dictionary of output folder paths.
        
        Returns:
            Updated artifact dictionary.
        """
        if not all(mask_paths.values()):
            return artifacts

        horizontal_line_mask = normalize_to_grayscale(load_image(mask_paths["horizontal_line_mask"]))
        short_horizontal_line_mask = normalize_to_grayscale(load_image(mask_paths["short_horizontal_line_mask"]))
        combined_horizontal_line_mask = normalize_to_grayscale(load_image(mask_paths["combined_horizontal_line_mask"]))
        vertical_line_mask = normalize_to_grayscale(load_image(mask_paths["vertical_line_mask"]))
        grouped_vertical_line_mask = normalize_to_grayscale(load_image(mask_paths["grouped_vertical_line_mask"]))

        line_masks_preview = draw_line_masks_preview(
            gray_input,
            horizontal_line_mask,
            short_horizontal_line_mask,
            combined_horizontal_line_mask,
            vertical_line_mask,
            grouped_vertical_line_mask,
        )

        artifacts["line_masks_preview"] = save_image(
            line_masks_preview,
            f"{folders['debug_dir']}/06_line_masks_preview.jpeg"
        )

        return artifacts

    def run_from_preparation_state(self, preparation_state, output_dir):
        """Run ScribeMap from the full N00 preparation state.

        Args:
            preparation_state: Dictionary returned by N00 / Prism.
            output_dir: Output directory for ScribeMap.

        Returns:
            Full ScribeMap result dictionary.
        """
        artifacts = preparation_state.get("artifacts", {})
        metadata = preparation_state.get("metadata", {})

        return self.run_from_prepared_masks(
            prepared_bw_image_path=artifacts["cropped"],
            content_ink_mask_path=artifacts["content_ink_mask"],
            output_dir=output_dir,

            black_pixel_mask_path=artifacts.get("black_pixel_mask"),

            blue_ink_mask_path=artifacts.get("blue_ink_mask"),
            red_ink_mask_path=artifacts.get("red_ink_mask"),
            green_ink_mask_path=artifacts.get("green_ink_mask"),
            unknown_color_ink_mask_path=artifacts.get("unknown_color_ink_mask"),
            colored_ink_mask_path=artifacts.get("colored_ink_mask"),
            black_ink_mask_path=artifacts.get("black_ink_mask"),

            recovered_blue_dark_mask_path=artifacts.get("recovered_blue_dark_mask"),
            recovered_red_dark_mask_path=artifacts.get("recovered_red_dark_mask"),
            recovered_green_dark_mask_path=artifacts.get("recovered_green_dark_mask"),

            horizontal_line_mask_path=artifacts.get("horizontal_line_mask"),
            short_horizontal_line_mask_path=artifacts.get("short_horizontal_line_mask"),
            combined_horizontal_line_mask_path=artifacts.get("combined_horizontal_line_mask"),
            vertical_line_mask_path=artifacts.get("vertical_line_mask"),
            grouped_vertical_line_mask_path=artifacts.get("grouped_vertical_line_mask"),

            vertical_debug=metadata.get("scribemap_masks", {}),
            preparation_artifacts=artifacts,
        )

    def run_from_prepared_masks(
        self,
        prepared_bw_image_path,
        content_ink_mask_path,
        output_dir,
        black_pixel_mask_path=None,
        blue_ink_mask_path=None,
        red_ink_mask_path=None,
        green_ink_mask_path=None,
        unknown_color_ink_mask_path=None,
        colored_ink_mask_path=None,
        black_ink_mask_path=None,
        recovered_blue_dark_mask_path=None,
        recovered_red_dark_mask_path=None,
        recovered_green_dark_mask_path=None,
        horizontal_line_mask_path=None,
        short_horizontal_line_mask_path=None,
        combined_horizontal_line_mask_path=None,
        vertical_line_mask_path=None,
        grouped_vertical_line_mask_path=None,
        vertical_debug=None,
        preparation_artifacts=None,
    ):
            # 1) setup output folders
        """Run ScribeMap from explicit prepared artifact paths.
        
        Args:
            prepared_bw_image_path: Path to the prepared crop image from node 00.
            content_ink_mask_path: Path to the content-ink mask from node 00.
            output_dir: Folder where generated files should be written.
            black_pixel_mask_path: Optional path to the black-pixel mask artifact.
            blue_ink_mask_path: Optional path to the blue-ink mask artifact.
            red_ink_mask_path: Optional path to the red-ink mask artifact.
            green_ink_mask_path: Optional path to the green-ink mask artifact.
            unknown_color_ink_mask_path: Optional path to the unknown-color-ink mask artifact.
            colored_ink_mask_path: Optional path to the colored-ink mask artifact.
            horizontal_line_mask_path: Optional path to the horizontal-line mask artifact.
            short_horizontal_line_mask_path: Optional path to the short-horizontal-line mask artifact.
            combined_horizontal_line_mask_path: Optional path to the combined-horizontal-line mask artifact.
            vertical_line_mask_path: Optional path to the vertical-line mask artifact.
            grouped_vertical_line_mask_path: Optional path to the grouped-vertical-line mask artifact.
            vertical_debug: Optional vertical-line debug metadata from file preparation.
        
        Returns:
            ScribeMap result dictionary with metadata path.
        """
        folders = ensure_output_folders(output_dir)

        # 2) load prepared grayscale image + content mask
        gray_input, content_ink_mask = self._load_prepared_inputs(
            prepared_bw_image_path,
            content_ink_mask_path
        )
        
        # 3) load optional layer masks if available 
        layer_mask_paths = {
            "blue": blue_ink_mask_path,
            "red": red_ink_mask_path,
            "green": green_ink_mask_path,
            "unknown_color": unknown_color_ink_mask_path,
            "colored": colored_ink_mask_path,
            "black": black_ink_mask_path or black_pixel_mask_path,
        }

        layer_masks = {}

        for layer_name, mask_path in layer_mask_paths.items():
            if mask_path is None:
                continue

            layer_mask = load_image(mask_path)
            layer_mask = normalize_to_grayscale(layer_mask)
            layer_masks[layer_name] = layer_mask

        # 4) detect components and build filtered groups
        components, raw_groups, groups, rejected_groups = self._build_groups_from_content_mask(
            content_ink_mask
        )

        layer_results = {}

        for layer_name, layer_mask in layer_masks.items():
            layer_results[layer_name] = self._build_groups_for_layer(
                layer_mask,
                layer_name
            )

        # 5) optional crop materialization for components/groups
        components_output, groups_output = self._build_crop_outputs(
            gray_input,
            components,
            groups,
            folders
        )

        # 6) previews for debugging/QA
        artifacts = self._build_preview_artifacts(
            gray_input,
            content_ink_mask,
            components,
            raw_groups,
            groups,
            rejected_groups,
            folders
        )

        for layer_name, layer_result in layer_results.items():
            layer_preview = draw_groups_preview(
                gray_input,
                layer_result["groups"]
            )

            artifacts[f"{layer_name}_groups_preview"] = save_image(
                layer_preview,
                f"{folders['debug_dir']}/layer_{layer_name}_groups_preview.jpeg"
            )

        # 7) optional line-mask overlay preview
        mask_paths = {
            "black_pixel_mask": black_pixel_mask_path,
            "horizontal_line_mask": horizontal_line_mask_path,
            "short_horizontal_line_mask": short_horizontal_line_mask_path,
            "combined_horizontal_line_mask": combined_horizontal_line_mask_path,
            "vertical_line_mask": vertical_line_mask_path,
            "grouped_vertical_line_mask": grouped_vertical_line_mask_path,
        }

        artifacts = self._add_line_masks_preview(
            artifacts,
            gray_input,
            mask_paths,
            folders
        )

        # 8) structured result payload
        result = {
            "algorithm": "ScribeMap_from_prepared_content_ink_mask",
            "prepared_bw_image_path": prepared_bw_image_path,
            "content_ink_mask_path": content_ink_mask_path,
            "output_dir": output_dir,
            "settings": self.settings,

            "preparation_artifacts": preparation_artifacts or {},
            "layer_mask_paths": layer_mask_paths,

            "component_count": len(components),
            "raw_group_count": len(raw_groups),
            "group_count": len(groups),
            "rejected_group_count": len(rejected_groups),

            "layer_results": layer_results,
            "vertical_debug": vertical_debug or {},

            "components": components_output,
            "raw_groups": raw_groups,
            "groups": groups_output,
            "rejected_groups": rejected_groups,
            "artifacts": artifacts,
        }

        # 9) persist metadata for reproducibility
        metadata_path = f"{folders['metadata_dir']}/scribemap_from_prepared_masks.json"
        save_json(result, metadata_path)
        result["metadata_path"] = metadata_path

        return result

    def run(self, input_bw_image_path, output_dir):
        """Run the detector/orchestrator entry point.
        
        Args:
            input_bw_image_path: Path to the prepared black/white input image.
            output_dir: Folder where generated files should be written.
        
        Returns:
            Detector result dictionary or raises when unsupported.
        """
        raise RuntimeError(
            "ScribeMap mask creation has been moved to file_preparation. "
            "Use run_from_prepared_masks(...) with file_preparation artifacts."
        )


if __name__ == "__main__":
    print("Run ScribeMap through prepared mask artifacts from file_preparation.")
