import os
import cv2
import json


def load_image(image_path):
    # Unified image loader with explicit failure signal.
    """Load an image from disk.
    
    Args:
        image_path: Path to the image file.
    
    Returns:
        Loaded image array.
    """
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    return image


def ensure_output_folders(output_dir):
    # Fixed output contract used by detector orchestrator.
    """Create and return the output folder structure.
    
    Args:
        output_dir: Folder where generated files should be written.
    
    Returns:
        Dictionary of created folder paths.
    """
    debug_dir = f"{output_dir}/debug"
    components_dir = f"{output_dir}/components"
    groups_dir = f"{output_dir}/groups"
    metadata_dir = f"{output_dir}/metadata"

    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(components_dir, exist_ok=True)
    os.makedirs(groups_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return {
        "debug_dir": debug_dir,
        "components_dir": components_dir,
        "groups_dir": groups_dir,
        "metadata_dir": metadata_dir
    }


def save_image(image, output_path):
    # Keep write checks explicit, since failed writes are easy to miss.
    """Save image.
    
    Args:
        image: Input image array.
        output_path: Path where the result should be saved.
    
    Returns:
        Computed result for the caller.
    """
    success = cv2.imwrite(output_path, image)

    if not success:
        raise ValueError(f"Could not save image: {output_path}")

    return output_path


def save_json(data, output_path):
    # UTF-8 + pretty JSON for readable metadata.
    """Write JSON data to disk.
    
    Args:
        data: Serializable data object.
        output_path: Path where the result should be saved.
    
    Returns:
        Path to the saved JSON file.
    """
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)

    return output_path
