#IMPORTS-------------------------------------------------------
import cv2
import json
import os

from image_preprocessor import ImagePreprocessor
from artifact_saver import ArtifactSaver
from image_quality import ImageQualityAnalyzer
from file_preparation_scribemap_masks import (
    create_black_pixel_mask,
    detect_scribemap_line_mask,
    build_grouped_vertical_line_mask,
    create_basic_color_ink_masks,
    isolate_layer_as_image,
)

#DEFAULT CONSTANTS---------------------------------------------------

DENOISE_STRENGTH = 10

CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

THRESHOLD_BLOCK_SIZE = 35
THRESHOLD_C = 15

CROP_PADDING = 20

MANUAL_MAJOR_ROTATION = 0

BLACK_PIXEL_THRESHOLD = 80
SCRIBEMAP_HORIZONTAL_LINE_KERNEL = (80, 1)
SCRIBEMAP_SHORT_HORIZONTAL_LINE_KERNEL = (25, 1)
SCRIBEMAP_VERTICAL_LINE_KERNEL = (1, 40)
REMOVE_GROUPED_VERTICAL_LINES = True
REMOVE_VERTICAL_LINES = False
VERTICAL_FRAGMENT_MIN_HEIGHT = 10
VERTICAL_FRAGMENT_MAX_WIDTH = 20
VERTICAL_FRAGMENT_MIN_ASPECT_RATIO = 2.5
VERTICAL_CLUSTER_X_TOLERANCE = 10
VERTICAL_CLUSTER_MIN_FRAGMENTS = 2
VERTICAL_CLUSTER_MIN_TOTAL_HEIGHT = 60
VERTICAL_CLUSTER_MIN_Y_SPAN = 80
GROUPED_VERTICAL_REMOVAL_HALF_WIDTH = 3


#MODULAR STATE SYSTEM---------------------------------------------------

def create_initial_state(input_path, output_dir, settings=None):
    """Create the mutable state object used by file preparation.

    Args:
        input_path: Path to the input document image.
        output_dir: Folder where generated files should be written.
        settings: Optional configuration dictionary used to override defaults.

    Returns:
        Initialized pipeline state dictionary.
    """
    if settings is None:
        settings = {}

    state = {
        "input_path": input_path,
        "output_dir": output_dir,
        "settings": {
            "manual_major_rotation": settings.get("manual_major_rotation", MANUAL_MAJOR_ROTATION),
            "denoise_strength": settings.get("denoise_strength", DENOISE_STRENGTH),
            "clahe_clip_limit": settings.get("clahe_clip_limit", CLAHE_CLIP_LIMIT),
            "clahe_tile_grid_size": settings.get("clahe_tile_grid_size", CLAHE_TILE_GRID_SIZE),
            "threshold_block_size": settings.get("threshold_block_size", THRESHOLD_BLOCK_SIZE),
            "threshold_c": settings.get("threshold_c", THRESHOLD_C),
            "crop_padding": settings.get("crop_padding", CROP_PADDING),
            "black_blue_max": settings.get("black_blue_max", 140),
            "black_green_max": settings.get("black_green_max", 140),
            "black_red_max": settings.get("black_red_max", 140),

            "min_horizontal_aspect_ratio": settings.get("min_horizontal_aspect_ratio", 8),
            "min_vertical_aspect_ratio": settings.get("min_vertical_aspect_ratio", 10),
            "max_horizontal_line_thickness": settings.get("max_horizontal_line_thickness", 18),
            "max_vertical_line_thickness": settings.get("max_vertical_line_thickness", 12),

            "deskew_horizontal_kernel_width": settings.get("deskew_horizontal_kernel_width", 120),
            "deskew_hough_threshold": settings.get("deskew_hough_threshold", 80),
            "deskew_min_line_length": settings.get("deskew_min_line_length", 250),
            "deskew_max_line_gap": settings.get("deskew_max_line_gap", 25),
            "deskew_max_abs_angle": settings.get("deskew_max_abs_angle", 10),
            "deskew_min_abs_angle": settings.get("deskew_min_abs_angle", 0.2),
            "black_pixel_threshold": settings.get("black_pixel_threshold", BLACK_PIXEL_THRESHOLD),

            "color_ink_min_saturation": settings.get("color_ink_min_saturation", 55),
            "color_ink_min_value": settings.get("color_ink_min_value", 70),
            "color_background_max_value": settings.get("color_background_max_value", 245),
            "color_chroma_min": settings.get("color_chroma_min", 25),
            "color_channel_margin": settings.get("color_channel_margin", 12),

            "recovered_dark_chroma_min": settings.get("recovered_dark_chroma_min", 12),
            "recovered_dark_min_saturation": settings.get("recovered_dark_min_saturation", 20),

            "weak_color_channel_margin": settings.get("weak_color_channel_margin", 5),
            "weak_color_chroma_min": settings.get("weak_color_chroma_min", 8),
            "weak_color_value_max": settings.get("weak_color_value_max", 210),

            "seeded_edge_recovery_enabled": settings.get("seeded_edge_recovery_enabled", True),
            "seeded_edge_recovery_iterations": settings.get("seeded_edge_recovery_iterations", 1),
            "seeded_edge_min_saturation": settings.get("seeded_edge_min_saturation", 25),
            "seeded_edge_min_chroma": settings.get("seeded_edge_min_chroma", 10),
            "seeded_edge_channel_margin": settings.get("seeded_edge_channel_margin", 5),
            "seeded_edge_max_value": settings.get("seeded_edge_max_value", 238),
            "seeded_edge_dark_max_value": settings.get("seeded_edge_dark_max_value", 185),

            "black_ink_max_saturation": settings.get("black_ink_max_saturation", 85),
            "black_ink_max_value": settings.get("black_ink_max_value", 180),
            "black_ink_gray_max": settings.get("black_ink_gray_max", 190),

            "horizontal_line_kernel": settings.get("horizontal_line_kernel", SCRIBEMAP_HORIZONTAL_LINE_KERNEL),
            "short_horizontal_line_kernel": settings.get("short_horizontal_line_kernel", SCRIBEMAP_SHORT_HORIZONTAL_LINE_KERNEL),
            "vertical_line_kernel": settings.get("vertical_line_kernel", SCRIBEMAP_VERTICAL_LINE_KERNEL),
            "remove_grouped_vertical_lines": settings.get("remove_grouped_vertical_lines", REMOVE_GROUPED_VERTICAL_LINES),
            "remove_vertical_lines": settings.get("remove_vertical_lines", REMOVE_VERTICAL_LINES),
            "vertical_fragment_min_height": settings.get("vertical_fragment_min_height", VERTICAL_FRAGMENT_MIN_HEIGHT),
            "vertical_fragment_max_width": settings.get("vertical_fragment_max_width", VERTICAL_FRAGMENT_MAX_WIDTH),
            "vertical_fragment_min_aspect_ratio": settings.get("vertical_fragment_min_aspect_ratio", VERTICAL_FRAGMENT_MIN_ASPECT_RATIO),
            "vertical_cluster_x_tolerance": settings.get("vertical_cluster_x_tolerance", VERTICAL_CLUSTER_X_TOLERANCE),
            "vertical_cluster_min_fragments": settings.get("vertical_cluster_min_fragments", VERTICAL_CLUSTER_MIN_FRAGMENTS),
            "vertical_cluster_min_total_height": settings.get("vertical_cluster_min_total_height", VERTICAL_CLUSTER_MIN_TOTAL_HEIGHT),
            "vertical_cluster_min_y_span": settings.get("vertical_cluster_min_y_span", VERTICAL_CLUSTER_MIN_Y_SPAN),
            "grouped_vertical_removal_half_width": settings.get("grouped_vertical_removal_half_width", GROUPED_VERTICAL_REMOVAL_HALF_WIDTH),
        },
        "images": {},
        "data": {},
        "artifacts": {},
        "metadata": {
            "input_path": input_path,
            "output_dir": output_dir,
            "steps_requested": [],
            "steps_completed": [],
            "shapes": {},
            "quality": {},
            "settings": {}
        }
    }

    state["metadata"]["settings"] = state["settings"]               # mirror settings into metadata

    return state                                                       # initial pipeline state object


#STEP WRAPPERS---------------------------------------------------

def step_load_image(state):
    """Load the input image into the preparation state.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = preprocessor.load_image(state["input_path"])

    state["images"]["original"] = image
    state["images"]["current"] = image
    state["images"]["color_current"] = image

    state["metadata"]["steps_completed"].append("load_image")
    state["metadata"]["shapes"]["original"] = list(image.shape)

    return state


def step_rotate_major(state):
    """Apply the configured major page rotation.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    rotated = preprocessor.rotate_major(image)

    state["images"]["rotated"] = rotated
    state["images"]["current"] = rotated

    if "color_current" in state["images"]:
        state["images"]["color_current"] = preprocessor.rotate_major(
            state["images"]["color_current"]
        )

    state["metadata"]["steps_completed"].append("rotate_major")
    state["metadata"]["shapes"]["rotated"] = list(rotated.shape)
    state["metadata"]["rotation"] = {
        "manual_major_rotation": state["settings"]["manual_major_rotation"]
    }

    return state


def step_convert_to_grayscale(state):
    """Convert the current image to grayscale.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    gray = preprocessor.convert_to_grayscale(image)

    state["images"]["gray"] = gray
    state["images"]["current"] = gray

    state["metadata"]["steps_completed"].append("convert_to_grayscale")
    state["metadata"]["shapes"]["gray"] = list(gray.shape)

    return state


def step_denoise_image(state):
    """Denoise the grayscale image.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    gray = state["images"]["current"]
    denoised = preprocessor.denoise_image(gray)

    state["images"]["denoised"] = denoised
    state["images"]["current"] = denoised

    state["metadata"]["steps_completed"].append("denoise_image")
    state["metadata"]["shapes"]["denoised"] = list(denoised.shape)

    return state


def step_improve_contrast(state):
    """Improve local contrast before thresholding.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    enhanced = preprocessor.improve_contrast(image)

    state["images"]["enhanced"] = enhanced
    state["images"]["current"] = enhanced

    state["metadata"]["steps_completed"].append("improve_contrast")
    state["metadata"]["shapes"]["enhanced"] = list(enhanced.shape)

    return state


def step_threshold_image(state):
    """Convert the enhanced image into a binary image.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    thresholded = preprocessor.threshold_image(image)

    state["images"]["thresholded"] = thresholded
    state["images"]["current"] = thresholded

    state["metadata"]["steps_completed"].append("threshold_image")
    state["metadata"]["shapes"]["thresholded"] = list(thresholded.shape)

    return state


def step_deskew_image(state):
    """Deskew the binary image and record the detected angle.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    deskewed, angle = preprocessor.deskew_image(image)

    state["images"]["deskewed"] = deskewed
    state["images"]["current"] = deskewed

    if "color_current" in state["images"]:
        color_deskewed = preprocessor.rotate_by_angle(
            state["images"]["color_current"],
            angle
        )

        state["images"]["color_deskewed"] = color_deskewed
        state["images"]["color_current"] = color_deskewed

    state["data"]["detected_skew_angle"] = angle

    state["metadata"]["steps_completed"].append("deskew_image")
    state["metadata"]["shapes"]["deskewed"] = list(deskewed.shape)
    state["metadata"]["quality"]["detected_skew_angle"] = round(angle, 2)

    return state


def step_crop_white_margins(state):
    """Crop outer white margins from the current image.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    preprocessor = ImagePreprocessor(state["settings"])

    image = state["images"]["current"]
    crop_bounds = preprocessor.get_white_margin_bounds(image)
    cropped = preprocessor.crop_to_bounds(image, crop_bounds)

    state["images"]["cropped"] = cropped
    state["images"]["current"] = cropped
    state["data"]["crop_bounds"] = crop_bounds

    if "color_current" in state["images"]:
        color_current = state["images"]["color_current"]

        if color_current.shape[:2] == image.shape[:2]:
            color_cropped = preprocessor.crop_to_bounds(color_current, crop_bounds)
            state["images"]["color_cropped"] = color_cropped
            state["images"]["color_current"] = color_cropped

    state["metadata"]["steps_completed"].append("crop_white_margins")
    state["metadata"]["shapes"]["cropped"] = list(cropped.shape)
    state["metadata"]["crop_bounds"] = {
        "x1": int(crop_bounds[0]),
        "y1": int(crop_bounds[1]),
        "x2": int(crop_bounds[2]),
        "y2": int(crop_bounds[3]),
    }

    return state


def step_calculate_quality(state):
    """Calculate basic image quality metrics.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    if "gray" in state["images"]:
        gray = state["images"]["gray"]
    else:
        preprocessor = ImagePreprocessor(state["settings"])
        gray = preprocessor.convert_to_grayscale(state["images"]["current"])

    quality_analyzer = ImageQualityAnalyzer(state["settings"])
    quality = quality_analyzer.analyze(gray)
    rounded_quality = quality_analyzer.analyze_rounded(gray)

    state["data"]["blur_score"] = quality["blur_score"]
    state["data"]["brightness"] = quality["brightness"]
    state["data"]["contrast"] = quality["contrast"]

    state["metadata"]["steps_completed"].append("calculate_quality")
    state["metadata"]["quality"]["blur_score"] = rounded_quality["blur_score"]
    state["metadata"]["quality"]["brightness"] = rounded_quality["brightness"]
    state["metadata"]["quality"]["contrast"] = rounded_quality["contrast"]

    return state


def step_create_color_layer_masks(state):
    """Create basic color ink masks from the preserved color image.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Updated pipeline state.
    """
    image = state["images"].get("color_current")

    if image is None:
        raise ValueError("No color image available for color layer masks")

    settings = state["settings"]

    color_masks = create_basic_color_ink_masks(image, settings)

    state["images"]["red_ink_mask"] = color_masks["red_ink_mask"]
    state["images"]["blue_ink_mask"] = color_masks["blue_ink_mask"]
    state["images"]["green_ink_mask"] = color_masks["green_ink_mask"]
    state["images"]["unknown_color_ink_mask"] = color_masks["unknown_color_ink_mask"]
    state["images"]["colored_ink_mask"] = color_masks["colored_ink_mask"]
    state["images"]["black_ink_mask"] = color_masks["black_ink_mask"]
    state["images"]["recovered_blue_dark_mask"] = color_masks["recovered_blue_dark_mask"]
    state["images"]["recovered_red_dark_mask"] = color_masks["recovered_red_dark_mask"]
    state["images"]["recovered_green_dark_mask"] = color_masks["recovered_green_dark_mask"]

    color_image = image

    state["images"]["red_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["red_ink_mask"]
    )

    state["images"]["blue_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["blue_ink_mask"]
    )

    state["images"]["green_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["green_ink_mask"]
    )

    state["images"]["unknown_color_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["unknown_color_ink_mask"]
    )

    state["images"]["colored_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["colored_ink_mask"]
    )

    state["images"]["black_ink_layer"] = isolate_layer_as_image(
        color_image,
        color_masks["black_ink_mask"]
    )

    state["metadata"]["steps_completed"].append("create_color_layer_masks")
    state["metadata"]["color_layer_masks"] = {
        "red_ink_pixels": int((color_masks["red_ink_mask"] > 0).sum()),
        "blue_ink_pixels": int((color_masks["blue_ink_mask"] > 0).sum()),
        "green_ink_pixels": int((color_masks["green_ink_mask"] > 0).sum()),
        "unknown_color_ink_pixels": int((color_masks["unknown_color_ink_mask"] > 0).sum()),
        "colored_ink_pixels": int((color_masks["colored_ink_mask"] > 0).sum()),
        "black_ink_pixels": int((color_masks["black_ink_mask"] > 0).sum()),
        "seeded_red_edge_pixels": int(
            (color_masks["seeded_red_edge_mask"] > 0).sum()
        ),
        "seeded_blue_edge_pixels": int(
            (color_masks["seeded_blue_edge_mask"] > 0).sum()
        ),
        "seeded_green_edge_pixels": int(
            (color_masks["seeded_green_edge_mask"] > 0).sum()
        ),
        "exclusive_overlap_pixels": color_masks.get("exclusive_overlap_pixels", None),
    }

    return state


def step_create_scribemap_masks(state):
    """Build the binary masks consumed by ScribeMap.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    image = state["images"]["current"]
    settings = state["settings"]

    black_pixel_mask = create_black_pixel_mask(image, settings)
    horizontal_line_mask = detect_scribemap_line_mask(
        black_pixel_mask,
        settings.get("horizontal_line_kernel", (80, 1))
    )
    short_horizontal_line_mask = detect_scribemap_line_mask(
        black_pixel_mask,
        settings.get("short_horizontal_line_kernel", (25, 1))
    )
    vertical_line_mask = detect_scribemap_line_mask(
        black_pixel_mask,
        settings.get("vertical_line_kernel", (1, 40))
    )

    (
        grouped_vertical_line_mask,
        vertical_fragments,
        vertical_clusters,
        selected_vertical_clusters
    ) = build_grouped_vertical_line_mask(vertical_line_mask, settings)

    combined_horizontal_line_mask = cv2.bitwise_or(
        horizontal_line_mask,
        short_horizontal_line_mask
    )

    content_ink_mask = cv2.subtract(
        black_pixel_mask,
        combined_horizontal_line_mask
    )

    if settings.get("remove_grouped_vertical_lines", True):
        content_ink_mask = cv2.subtract(content_ink_mask, grouped_vertical_line_mask)
    elif settings.get("remove_vertical_lines", False):
        content_ink_mask = cv2.subtract(content_ink_mask, vertical_line_mask)

    state["images"]["black_pixel_mask"] = black_pixel_mask
    state["images"]["horizontal_line_mask"] = horizontal_line_mask
    state["images"]["short_horizontal_line_mask"] = short_horizontal_line_mask
    state["images"]["combined_horizontal_line_mask"] = combined_horizontal_line_mask
    state["images"]["vertical_line_mask"] = vertical_line_mask
    state["images"]["grouped_vertical_line_mask"] = grouped_vertical_line_mask
    state["images"]["content_ink_mask"] = content_ink_mask

    state["data"]["vertical_fragments"] = vertical_fragments
    state["data"]["vertical_clusters"] = vertical_clusters
    state["data"]["selected_vertical_clusters"] = selected_vertical_clusters

    state["metadata"]["steps_completed"].append("create_scribemap_masks")
    state["metadata"]["scribemap_masks"] = {
        "vertical_fragment_count": len(vertical_fragments),
        "vertical_cluster_count": len(vertical_clusters),
        "selected_vertical_cluster_count": len(selected_vertical_clusters),
    }
    state["metadata"]["shapes"]["black_pixel_mask"] = list(black_pixel_mask.shape)
    state["metadata"]["shapes"]["content_ink_mask"] = list(content_ink_mask.shape)

    return state


def step_save_outputs(state):
    """Save all available artifacts for the preparation state.

    Args:
        state: Mutable pipeline state dictionary.

    Returns:
        Computed result for the caller.
    """
    saver = ArtifactSaver(state["output_dir"])

    state = saver.save_outputs(state)

    state["metadata"]["steps_completed"].append("save_outputs")

    return state


#STEP REGISTRY---------------------------------------------------

STEP_REGISTRY = {
    "load_image": step_load_image,
    "rotate_major": step_rotate_major,
    "convert_to_grayscale": step_convert_to_grayscale,
    "denoise_image": step_denoise_image,
    "improve_contrast": step_improve_contrast,
    "threshold_image": step_threshold_image,
    "deskew_image": step_deskew_image,
    "crop_white_margins": step_crop_white_margins,
    "calculate_quality": step_calculate_quality,
    "create_scribemap_masks": step_create_scribemap_masks,
    "create_color_layer_masks": step_create_color_layer_masks,
    "save_outputs": step_save_outputs,
}


#MAIN MODULAR PREPARATION FUNCTION---------------------------------------------------

def prepare_file(input_path, output_dir, steps, settings=None):
    """Run the selected file-preparation steps for one document.

    Args:
        input_path: Path to the input document image.
        output_dir: Folder where generated files should be written.
        steps: Ordered list of preparation step names to run.
        settings: Optional configuration dictionary used to override defaults.

    Returns:
        Final pipeline state after all requested steps.
    """
    state = create_initial_state(
        input_path=input_path,
        output_dir=output_dir,
        settings=settings
    )                                                               # initialize pipeline state and settings

    state["metadata"]["steps_requested"] = steps                 # record requested processing stages

    for step_name in steps:
        if step_name not in STEP_REGISTRY:
            raise ValueError(f"Unknown preparation step: {step_name}")

        step_function = STEP_REGISTRY[step_name]
        state = step_function(state)                                # execute each pipeline step in order

    return state                                                  # return final pipeline state


#TEST RUN---------------------------------------------------

if __name__ == "__main__":
    TEST_IMAGE_PATH = "/home/vahram/Desktop/image_Processor/type-1_page-2.jpg"
    TEST_OUTPUT_DIR = "/home/vahram/Desktop/image_Processor/test_modular_output"

    OCR_STEPS = [
        "load_image",
        "rotate_major",
        "convert_to_grayscale",
        "denoise_image",
        "improve_contrast",
        "threshold_image",
        "deskew_image",
        "crop_white_margins",
        "calculate_quality",
        "create_scribemap_masks",
        "create_color_layer_masks",
        "save_outputs"
    ]

    result_state = prepare_file(
        input_path=TEST_IMAGE_PATH,
        output_dir=TEST_OUTPUT_DIR,
        steps=OCR_STEPS,
        settings={
            "manual_major_rotation": 0
        }
    )

    print(json.dumps(result_state["metadata"], indent=4, ensure_ascii=False))
    print("Artifacts saved:")
    print(json.dumps(result_state["artifacts"], indent=4, ensure_ascii=False))
