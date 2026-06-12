def crop_components(
    bw_image,
    components,
    components_dir,
    margin,
    save_image,
    file_prefix="component",
):
    # Extract per-component crops with a configurable safety margin.
    # Margin helps preserve local context around thin strokes.
    """Crop each detected component from the source image.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        components: List of detected micro-component dictionaries.
        components_dir: Folder for component crop images.
        margin: Pixel padding to include around the crop.
        save_image: Callback used to save image arrays.
        file_prefix: Filename prefix for each saved crop.
    
    Returns:
        List of component dictionaries with crop paths.
    """
    cropped_components = []
    image_height, image_width = bw_image.shape[:2]

    for component in components:
        x1 = max(component["x1"] - margin, 0)
        y1 = max(component["y1"] - margin, 0)
        x2 = min(component["x2"] + margin, image_width)
        y2 = min(component["y2"] + margin, image_height)
        crop = bw_image[y1:y2, x1:x2]
        file_name = (
            f"{file_prefix}_{component['component_id']:04d}_"
            f"x{x1:04d}_x{x2:04d}_"
            f"y{y1:04d}_y{y2:04d}.jpeg"
        )
        crop_path = f"{components_dir}/{file_name}"
        save_image(crop, crop_path)

        component_with_crop = dict(component)
        component_with_crop["crop_path"] = crop_path
        component_with_crop["crop_bbox"] = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        }
        cropped_components.append(component_with_crop)

    return cropped_components


def crop_groups(bw_image, groups, groups_dir, margin, save_image):
    # Extract per-group crops for classifier input and QA review.
    # Group bbox already encloses multiple components; margin expands context.
    """Crop each accepted group from the source image.
    
    Args:
        bw_image: Black/white or grayscale source image array.
        groups: List of accepted group dictionaries.
        groups_dir: Folder for group crop images.
        margin: Pixel padding to include around the crop.
        save_image: Callback used to save image arrays.
    
    Returns:
        List of group dictionaries with crop paths.
    """
    cropped_groups = []
    image_height, image_width = bw_image.shape[:2]

    for group in groups:
        x1 = max(group["x1"] - margin, 0)
        y1 = max(group["y1"] - margin, 0)
        x2 = min(group["x2"] + margin, image_width)
        y2 = min(group["y2"] + margin, image_height)
        crop = bw_image[y1:y2, x1:x2]
        file_name = (
            f"group_{group['group_id']:04d}_"
            f"x{x1:04d}_x{x2:04d}_"
            f"y{y1:04d}_y{y2:04d}.jpeg"
        )
        crop_path = f"{groups_dir}/{file_name}"
        save_image(crop, crop_path)

        group_with_crop = dict(group)
        group_with_crop["crop_path"] = crop_path
        group_with_crop["crop_bbox"] = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2)
        }
        cropped_groups.append(group_with_crop)

    return cropped_groups
