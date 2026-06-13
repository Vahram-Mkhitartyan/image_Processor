import cv2
import numpy as np


def create_black_pixel_mask(image, settings):
    """Create a binary mask of dark ink pixels.
    
    Args:
        image: Input image array.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        Binary mask with ink pixels set to 255.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    black_threshold = settings.get("black_pixel_threshold", 80)

    black_pixels = gray <= black_threshold
    black_mask = black_pixels.astype("uint8") * 255

    return black_mask

def recover_dark_pixels_near_color(black_mask, color_mask, kernel_size=5):
    """Recover dark pixels that probably belong to a nearby color layer.

    This helps with dark blue/red ink that gets detected as black because
    it is visually very dark.

    Args:
        black_mask: Binary dark-pixel mask.
        color_mask: Binary color mask, such as blue_ink_mask.
        kernel_size: Neighborhood size for nearby-color recovery.

    Returns:
        Binary mask of recovered dark pixels.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    color_neighborhood = cv2.dilate(
        color_mask,
        kernel,
        iterations=1
    )

    recovered = cv2.bitwise_and(
        black_mask,
        color_neighborhood
    )

    return recovered


def grow_seed_mask_through_candidates(seed_pixels, candidate_pixels, iterations=1):
    """Recover bounded stroke edges without performing blind dilation.

    Args:
        seed_pixels: Boolean pixels already trusted as one color layer.
        candidate_pixels: Weaker same-color evidence from the source image.
        iterations: Maximum pixel distance that recovery may travel from a seed.

    Returns:
        Tuple of the grown boolean mask and newly recovered boolean pixels.
    """
    grown = np.asarray(seed_pixels, dtype=bool).copy()
    candidates = np.asarray(candidate_pixels, dtype=bool)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max(int(iterations), 0)):
        neighborhood = cv2.dilate(
            grown.astype(np.uint8),
            kernel,
            iterations=1,
        ) > 0
        grown |= candidates & neighborhood

    return grown, grown & ~np.asarray(seed_pixels, dtype=bool)


def shift_boolean_mask(mask, dx, dy):
    """Shift a boolean mask without wrapping pixels around image borders.

    Args:
        mask: Two-dimensional boolean-compatible mask.
        dx: Horizontal shift in pixels; positive values move pixels right.
        dy: Vertical shift in pixels; positive values move pixels down.

    Returns:
        Shifted boolean mask with newly exposed pixels set to False.
    """
    source = np.asarray(mask, dtype=bool)
    height, width = source.shape
    shifted = np.zeros_like(source)

    source_x1 = max(0, -dx)
    source_x2 = min(width, width - dx)
    source_y1 = max(0, -dy)
    source_y2 = min(height, height - dy)

    if source_x1 >= source_x2 or source_y1 >= source_y2:
        return shifted

    destination_x1 = source_x1 + dx
    destination_x2 = source_x2 + dx
    destination_y1 = source_y1 + dy
    destination_y2 = source_y2 + dy

    shifted[
        destination_y1:destination_y2,
        destination_x1:destination_x2,
    ] = source[source_y1:source_y2, source_x1:source_x2]

    return shifted


def find_cross_color_bridge_pixels(target_mask, donor_mask, radius=4):
    """Find donor-color pixels that bridge two opposing target-stroke sides.

    The detector checks four undirected axes: horizontal, vertical, and both
    diagonals. A donor pixel is accepted only when target ink exists on both
    sides of at least one axis within ``radius`` pixels.

    Args:
        target_mask: Exclusive semantic mask whose continuity is being repaired.
        donor_mask: Other-color pixels that may occupy a true crossing.
        radius: Maximum distance searched on either side of a donor pixel.

    Returns:
        Binary uint8 mask containing only accepted borrowed crossing pixels.
    """
    target = np.asarray(target_mask) > 0
    donor = np.asarray(donor_mask) > 0
    search_radius = max(1, min(int(radius), 12))
    bridge_support = np.zeros_like(target)

    for dx, dy in ((1, 0), (0, 1), (1, 1), (1, -1)):
        side_a = np.zeros_like(target)
        side_b = np.zeros_like(target)

        for distance in range(1, search_radius + 1):
            side_a |= shift_boolean_mask(
                target,
                dx * distance,
                dy * distance,
            )
            side_b |= shift_boolean_mask(
                target,
                -dx * distance,
                -dy * distance,
            )

        bridge_support |= side_a & side_b

    return (donor & bridge_support).astype(np.uint8) * 255


def create_cross_color_continuity_masks(red_mask, blue_mask, settings):
    """Create trace-safe red/blue masks while preserving semantic ownership.

    Exclusive masks remain the source of truth for color classification. These
    continuity masks may overlap only at geometrically supported crossings so
    skeleton-based consumers do not see artificial breaks.

    Args:
        red_mask: Exclusive red semantic mask.
        blue_mask: Exclusive blue semantic mask.
        settings: Configuration dictionary containing continuity options.

    Returns:
        Dictionary with repaired masks, borrowed-pixel masks, and pixel counts.
    """
    red = np.asarray(red_mask, dtype=np.uint8)
    blue = np.asarray(blue_mask, dtype=np.uint8)

    if not settings.get("cross_color_continuity_enabled", True):
        empty = np.zeros_like(red)
        return {
            "red_continuity_mask": red.copy(),
            "blue_continuity_mask": blue.copy(),
            "red_borrowed_bridge_mask": empty.copy(),
            "blue_borrowed_bridge_mask": empty,
            "red_borrowed_bridge_pixels": 0,
            "blue_borrowed_bridge_pixels": 0,
        }

    radius = settings.get("cross_color_bridge_radius_px", 4)
    blue_borrowed = find_cross_color_bridge_pixels(
        target_mask=blue,
        donor_mask=red,
        radius=radius,
    )
    red_borrowed = find_cross_color_bridge_pixels(
        target_mask=red,
        donor_mask=blue,
        radius=radius,
    )

    return {
        "red_continuity_mask": cv2.bitwise_or(red, red_borrowed),
        "blue_continuity_mask": cv2.bitwise_or(blue, blue_borrowed),
        "red_borrowed_bridge_mask": red_borrowed,
        "blue_borrowed_bridge_mask": blue_borrowed,
        "red_borrowed_bridge_pixels": int(np.count_nonzero(red_borrowed)),
        "blue_borrowed_bridge_pixels": int(np.count_nonzero(blue_borrowed)),
    }


def create_basic_color_ink_masks(image, settings):
    """Create exclusive color ink masks from a BGR color image.

    Color Update invariant:
        One ink pixel may belong to only one semantic layer.

    Outputs:
        red_ink_mask
        blue_ink_mask
        green_ink_mask
        unknown_color_ink_mask
        colored_ink_mask
        black_ink_mask

    Notes:
        - Colored pixels are assigned by hue family.
        - Dark colored pixels can be recovered near already-detected color ink.
        - Recovered dark pixels are assigned with winner-take-all logic.
        - Black is computed after colored recovery and explicitly excludes
          red/blue/green/unknown colored pixels.
    """
    if len(image.shape) != 3:
        raise ValueError("create_basic_color_ink_masks expects a color BGR image")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    hue = hsv[:, :, 0]          # OpenCV hue range: 0-179
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    b = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    r = image[:, :, 2].astype(np.int16)

    max_channel = np.maximum.reduce([b, g, r])
    min_channel = np.minimum.reduce([b, g, r])
    chroma = max_channel - min_channel

    color_chroma_min = settings.get("color_chroma_min", 25)
    color_channel_margin = settings.get("color_channel_margin", 12)
    red_hue_max = settings.get("red_hue_max", 8)
    red_hue_min_high = settings.get("red_hue_min_high", 172)
    blue_hue_min = settings.get("blue_hue_min", 92)
    blue_hue_max = settings.get("blue_hue_max", 148)
    green_hue_min = settings.get("green_hue_min", 40)
    green_hue_max = settings.get("green_hue_max", 82)

    colored_candidate = (
        (saturation >= settings.get("color_ink_min_saturation", 55)) &
        (value >= settings.get("color_ink_min_value", 70)) &
        (value <= settings.get("color_background_max_value", 245)) &
        (chroma >= color_chroma_min)
    )

    red_pixels = (
        colored_candidate &
        (
            (hue <= red_hue_max) |
            (hue >= red_hue_min_high)
        ) &
        (r >= b + color_channel_margin) &
        (r >= g + color_channel_margin)
    )

    blue_pixels = (
        colored_candidate &
        (hue >= blue_hue_min) &
        (hue <= blue_hue_max) &
        (b >= r + color_channel_margin) &
        (b >= g + color_channel_margin)
    )

    green_pixels = (
        colored_candidate &
        (hue >= green_hue_min) &
        (hue <= green_hue_max) &
        (g >= r + color_channel_margin) &
        (g >= b + color_channel_margin)
    )

    known_colored_pixels = red_pixels | blue_pixels | green_pixels

    unknown_color_pixels = (
        colored_candidate &
        ~known_colored_pixels
    )

    # Weak color evidence is useful only when it continues an already reliable
    # stroke. Global weak-channel tests mistake warm paper and scan shadows for
    # red ink, so every weak pixel must touch a same-color seed neighborhood.
    weak_color_margin = settings.get("weak_color_channel_margin", 7)
    weak_color_chroma_min = settings.get("weak_color_chroma_min", 14)
    weak_color_min_saturation = settings.get("weak_color_min_saturation", 28)
    weak_color_min_value = settings.get("weak_color_min_value", 30)
    weak_color_value_max = settings.get("weak_color_value_max", 235)
    weak_color_kernel_size = settings.get("weak_color_recovery_kernel", 5)

    weak_candidate = (
        (saturation >= weak_color_min_saturation) &
        (value >= weak_color_min_value) &
        (value <= weak_color_value_max) &
        (chroma >= weak_color_chroma_min)
    )

    weak_red_pixels = (
        weak_candidate &
        ((hue <= red_hue_max) | (hue >= red_hue_min_high)) &
        (r >= b + weak_color_margin) &
        (r >= g + weak_color_margin)
    )
    weak_blue_pixels = (
        weak_candidate &
        (hue >= blue_hue_min) &
        (hue <= blue_hue_max) &
        (b >= r + weak_color_margin) &
        (b >= g + weak_color_margin)
    )
    weak_green_pixels = (
        weak_candidate &
        (hue >= green_hue_min) &
        (hue <= green_hue_max) &
        (g >= r + weak_color_margin) &
        (g >= b + weak_color_margin)
    )

    weak_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (weak_color_kernel_size, weak_color_kernel_size),
    )

    red_neighborhood = cv2.dilate(
        red_pixels.astype("uint8") * 255,
        weak_kernel,
        iterations=1,
    ) > 0
    blue_neighborhood = cv2.dilate(
        blue_pixels.astype("uint8") * 255,
        weak_kernel,
        iterations=1,
    ) > 0
    green_neighborhood = cv2.dilate(
        green_pixels.astype("uint8") * 255,
        weak_kernel,
        iterations=1,
    ) > 0

    red_pixels = red_pixels | (weak_red_pixels & red_neighborhood)
    blue_pixels = blue_pixels | (weak_blue_pixels & blue_neighborhood)
    green_pixels = green_pixels | (weak_green_pixels & green_neighborhood)

    red_ink_mask = red_pixels.astype("uint8") * 255
    blue_ink_mask = blue_pixels.astype("uint8") * 255
    green_ink_mask = green_pixels.astype("uint8") * 255
    unknown_color_ink_mask = unknown_color_pixels.astype("uint8") * 255

    black_pixel_mask = create_black_pixel_mask(image, settings)

    recovered_blue_dark = recover_dark_pixels_near_color(
        black_pixel_mask,
        blue_ink_mask,
        kernel_size=settings.get("recover_blue_dark_kernel", 5),
    )

    recovered_red_dark = recover_dark_pixels_near_color(
        black_pixel_mask,
        red_ink_mask,
        kernel_size=settings.get("recover_red_dark_kernel", 5),
    )

    recovered_green_dark = recover_dark_pixels_near_color(
        black_pixel_mask,
        green_ink_mask,
        kernel_size=settings.get("recover_green_dark_kernel", 5),
    )

    # Winner-take-all recovery:
    # A dark pixel may be near multiple color neighborhoods.
    # Assign it to exactly one recovered color by direct hue distance.
    recovered_any = (
        (recovered_blue_dark > 0) |
        (recovered_red_dark > 0) |
        (recovered_green_dark > 0)
    )

    recoverable_colored_dark = (
        recovered_any &
        (chroma >= settings.get("recovered_dark_chroma_min", 12)) &
        (saturation >= settings.get("recovered_dark_min_saturation", 20))
    )

    recovered_any = recoverable_colored_dark

    recovered_blue_pixels = np.zeros_like(recovered_any, dtype=bool)
    recovered_red_pixels = np.zeros_like(recovered_any, dtype=bool)
    recovered_green_pixels = np.zeros_like(recovered_any, dtype=bool)

    if np.any(recovered_any):
        hue_int = hue.astype(np.int16)

        # Circular hue distance in OpenCV hue space.
        red_distance_low = np.abs(hue_int - 0)
        red_distance_high = np.abs(hue_int - 179)
        red_distance = np.minimum(red_distance_low, red_distance_high)

        blue_center = settings.get("blue_hue_center", 120)
        red_center = settings.get("red_hue_center", 0)
        green_center = settings.get("green_hue_center", 60)

        blue_distance = np.abs(hue_int - int(blue_center))
        green_distance = np.abs(hue_int - int(green_center))

        # Red wraps around hue boundary.
        red_center_distance_low = np.abs(hue_int - int(red_center))
        red_center_distance_high = np.abs(hue_int - (179 - int(red_center)))
        red_center_distance = np.minimum(
            red_center_distance_low,
            red_center_distance_high,
        )

        # Keep the simple red_distance useful when red_center is 0.
        red_distance = np.minimum(red_distance, red_center_distance)

        color_distance_stack = np.stack(
            [red_distance, blue_distance, green_distance],
            axis=2,
        )

        winner = np.argmin(color_distance_stack, axis=2)

        recovered_red_pixels = recovered_any & (winner == 0)
        recovered_blue_pixels = recovered_any & (winner == 1)
        recovered_green_pixels = recovered_any & (winner == 2)

    red_pixels = red_pixels | recovered_red_pixels
    blue_pixels = blue_pixels | recovered_blue_pixels
    green_pixels = green_pixels | recovered_green_pixels

    seeded_red_edge_pixels = np.zeros_like(red_pixels, dtype=bool)
    seeded_blue_edge_pixels = np.zeros_like(blue_pixels, dtype=bool)
    seeded_green_edge_pixels = np.zeros_like(green_pixels, dtype=bool)

    if settings.get("seeded_edge_recovery_enabled", True):
        edge_min_saturation = settings.get(
            "seeded_edge_min_saturation", 23
        )
        edge_min_chroma = settings.get("seeded_edge_min_chroma", 9)
        edge_channel_margin = settings.get(
            "seeded_edge_channel_margin", 4
        )
        edge_max_value = settings.get("seeded_edge_max_value", 240)
        edge_dark_max_value = settings.get(
            "seeded_edge_dark_max_value", 185
        )
        edge_iterations = settings.get(
            "seeded_edge_recovery_iterations", 1
        )

        # The dark branch catches faded interior stroke pixels. It remains
        # color-sensitive and cannot grow without touching a trusted seed.
        dark_min_saturation = max(edge_min_saturation - 7, 1)
        dark_min_chroma = max(edge_min_chroma - 3, 1)
        dark_channel_margin = max(edge_channel_margin - 2, 1)

        edge_candidate = (
            (saturation >= edge_min_saturation) &
            (value >= 25) &
            (value <= edge_max_value) &
            (chroma >= edge_min_chroma)
        )
        dark_edge_candidate = (
            (saturation >= dark_min_saturation) &
            (value <= edge_dark_max_value) &
            (chroma >= dark_min_chroma)
        )

        red_hue_candidate = (
            (hue <= red_hue_max) |
            (hue >= red_hue_min_high)
        )
        blue_hue_candidate = (
            (hue >= blue_hue_min) &
            (hue <= blue_hue_max)
        )
        green_hue_candidate = (
            (hue >= green_hue_min) &
            (hue <= green_hue_max)
        )

        red_edge_candidate = (
            (
                edge_candidate &
                (r >= b + edge_channel_margin) &
                (r >= g + edge_channel_margin)
            ) |
            (
                dark_edge_candidate &
                (r >= b + dark_channel_margin) &
                (r >= g + dark_channel_margin)
            )
        ) & red_hue_candidate
        blue_edge_candidate = (
            (
                edge_candidate &
                (b >= r + edge_channel_margin) &
                (b >= g + edge_channel_margin)
            ) |
            (
                dark_edge_candidate &
                (b >= r + dark_channel_margin) &
                (b >= g + dark_channel_margin)
            )
        ) & blue_hue_candidate
        green_edge_candidate = (
            (
                edge_candidate &
                (g >= r + edge_channel_margin) &
                (g >= b + edge_channel_margin)
            ) |
            (
                dark_edge_candidate &
                (g >= r + dark_channel_margin) &
                (g >= b + dark_channel_margin)
            )
        ) & green_hue_candidate

        red_pixels, seeded_red_edge_pixels = grow_seed_mask_through_candidates(
            red_pixels,
            red_edge_candidate,
            edge_iterations,
        )
        blue_pixels, seeded_blue_edge_pixels = grow_seed_mask_through_candidates(
            blue_pixels,
            blue_edge_candidate,
            edge_iterations,
        )
        green_pixels, seeded_green_edge_pixels = grow_seed_mask_through_candidates(
            green_pixels,
            green_edge_candidate,
            edge_iterations,
        )

    known_or_recovered_colored_pixels = (
        red_pixels |
        blue_pixels |
        green_pixels |
        unknown_color_pixels
    )

    black_pixels = (
        (saturation <= settings.get("black_ink_max_saturation", 80)) &
        (value <= settings.get("black_ink_max_value", 170)) &
        (gray <= settings.get("black_ink_gray_max", 180)) &
        ~known_or_recovered_colored_pixels
    )

    # Force exclusive masks.
    # Priority: red/blue/green/unknown first, black last.
    red_only = red_pixels
    blue_only = blue_pixels & ~red_only
    green_only = green_pixels & ~red_only & ~blue_only
    unknown_only = unknown_color_pixels & ~red_only & ~blue_only & ~green_only
    black_only = black_pixels & ~red_only & ~blue_only & ~green_only & ~unknown_only

    red_ink_mask = red_only.astype("uint8") * 255
    blue_ink_mask = blue_only.astype("uint8") * 255
    green_ink_mask = green_only.astype("uint8") * 255
    unknown_color_ink_mask = unknown_only.astype("uint8") * 255
    black_ink_mask = black_only.astype("uint8") * 255

    colored_ink_mask = (
        red_only |
        blue_only |
        green_only |
        unknown_only
    ).astype("uint8") * 255

    recovered_blue_dark = recovered_blue_pixels.astype("uint8") * 255
    recovered_red_dark = recovered_red_pixels.astype("uint8") * 255
    recovered_green_dark = recovered_green_pixels.astype("uint8") * 255

    # Debug invariant: no pixel should belong to more than one final layer.
    overlap_count = (
        (red_ink_mask > 0).astype(np.uint8) +
        (blue_ink_mask > 0).astype(np.uint8) +
        (green_ink_mask > 0).astype(np.uint8) +
        (unknown_color_ink_mask > 0).astype(np.uint8) +
        (black_ink_mask > 0).astype(np.uint8)
    )

    exclusive_overlap_pixels = int(np.count_nonzero(overlap_count > 1))
    continuity_masks = create_cross_color_continuity_masks(
        red_mask=red_ink_mask,
        blue_mask=blue_ink_mask,
        settings=settings,
    )

    return {
        "red_ink_mask": red_ink_mask,
        "blue_ink_mask": blue_ink_mask,
        "green_ink_mask": green_ink_mask,
        "unknown_color_ink_mask": unknown_color_ink_mask,
        "colored_ink_mask": colored_ink_mask,
        "black_ink_mask": black_ink_mask,
        "recovered_blue_dark_mask": recovered_blue_dark,
        "recovered_red_dark_mask": recovered_red_dark,
        "recovered_green_dark_mask": recovered_green_dark,
        "seeded_blue_edge_mask":
            seeded_blue_edge_pixels.astype("uint8") * 255,
        "seeded_red_edge_mask":
            seeded_red_edge_pixels.astype("uint8") * 255,
        "seeded_green_edge_mask":
            seeded_green_edge_pixels.astype("uint8") * 255,
        "exclusive_overlap_pixels": exclusive_overlap_pixels,
        **continuity_masks,
    }


def isolate_layer_as_image(source_image, mask, background_value=255):
    """Create an isolated visual layer from a source image and mask.

    Pixels inside the mask keep their original color.
    Everything outside the mask becomes white.

    Args:
        source_image: Original BGR image.
        mask: Binary mask where target pixels are 255.
        background_value: Background value, usually 255.

    Returns:
        Isolated BGR image.
    """
    if len(source_image.shape) == 2:
        output = np.full_like(source_image, background_value)
        output[mask > 0] = source_image[mask > 0]
        return output

    output = np.full_like(source_image, background_value)
    output[mask > 0] = source_image[mask > 0]

    return output


def detect_scribemap_line_mask(ink_mask, kernel_size):
    """Detect line-like structures using morphology.
    
    Args:
        ink_mask: Binary mask where ink pixels are white.
        kernel_size: Morphology kernel size as (width, height).
    
    Returns:
        Binary line mask produced by morphology.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    return cv2.morphologyEx(ink_mask, cv2.MORPH_OPEN, kernel)


def extract_vertical_line_fragments(vertical_line_mask, settings):
    """Extract candidate vertical line fragments from a mask.
    
    Args:
        vertical_line_mask: Binary mask containing vertical line candidates.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        List of vertical fragment dictionaries.
    """
    contours, _ = cv2.findContours(
        vertical_line_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    fragments = []

    min_height = settings.get("vertical_fragment_min_height", 10)
    max_width = settings.get("vertical_fragment_max_width", 20)
    min_aspect_ratio = settings.get("vertical_fragment_min_aspect_ratio", 2.5)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h < min_height:
            continue

        if w > max_width:
            continue

        aspect_ratio = h / max(w, 1)

        if aspect_ratio < min_aspect_ratio:
            continue

        fragments.append({
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + w),
            "y2": int(y + h),
            "center_x": int(x + w // 2),
            "center_y": int(y + h // 2),
            "aspect_ratio": round(float(aspect_ratio), 3)
        })

    fragments = sorted(
        fragments,
        key=lambda fragment: (fragment["center_x"], fragment["y1"])
    )

    return fragments


def group_vertical_fragments_by_x(fragments, settings):
    """Group vertical fragments that share similar x positions.
    
    Args:
        fragments: List of vertical fragment dictionaries.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        List of x-position cluster dictionaries.
    """
    x_tolerance = settings.get("vertical_cluster_x_tolerance", 10)
    clusters = []

    for fragment in fragments:
        matched_cluster = None

        for cluster in clusters:
            if abs(fragment["center_x"] - cluster["center_x"]) <= x_tolerance:
                matched_cluster = cluster
                break

        if matched_cluster is None:
            clusters.append({
                "center_x": fragment["center_x"],
                "fragments": [fragment]
            })
        else:
            matched_cluster["fragments"].append(fragment)
            centers = [item["center_x"] for item in matched_cluster["fragments"]]
            matched_cluster["center_x"] = int(sum(centers) / len(centers))

    return sorted(clusters, key=lambda cluster: cluster["center_x"])


def select_structural_vertical_clusters(clusters, settings):
    """Select vertical fragment clusters likely to be structure.
    
    Args:
        clusters: List of grouped vertical-fragment clusters.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        List of selected structural cluster dictionaries.
    """
    selected_clusters = []

    min_fragments = settings.get("vertical_cluster_min_fragments", 2)
    min_total_height = settings.get("vertical_cluster_min_total_height", 60)
    min_y_span = settings.get("vertical_cluster_min_y_span", 80)

    for cluster in clusters:
        fragments = cluster["fragments"]
        total_height = sum(fragment["h"] for fragment in fragments)
        y1 = min(fragment["y1"] for fragment in fragments)
        y2 = max(fragment["y2"] for fragment in fragments)
        y_span = y2 - y1
        fragment_count = len(fragments)

        is_structural = (
            fragment_count >= min_fragments
            or total_height >= min_total_height
            or y_span >= min_y_span
        )

        if not is_structural:
            continue

        selected_clusters.append({
            "center_x": cluster["center_x"],
            "fragment_count": fragment_count,
            "total_height": int(total_height),
            "y_span": int(y_span),
            "y1": int(y1),
            "y2": int(y2),
            "fragments": fragments
        })

    return sorted(selected_clusters, key=lambda cluster: cluster["center_x"])


def build_grouped_vertical_line_mask(vertical_line_mask, settings):
    """Build a mask for selected structural vertical clusters.
    
    Args:
        vertical_line_mask: Binary mask containing vertical line candidates.
        settings: Optional configuration dictionary used to override defaults.
    
    Returns:
        Tuple of grouped mask, fragments, clusters, and selected clusters.
    """
    fragments = extract_vertical_line_fragments(vertical_line_mask, settings)
    clusters = group_vertical_fragments_by_x(fragments, settings)
    selected_clusters = select_structural_vertical_clusters(clusters, settings)

    grouped_vertical_mask = np.zeros_like(vertical_line_mask)
    removal_half_width = settings.get("grouped_vertical_removal_half_width", 3)
    image_height, image_width = vertical_line_mask.shape[:2]

    for cluster in selected_clusters:
        center_x = cluster["center_x"]

        for fragment in cluster["fragments"]:
            x1 = max(center_x - removal_half_width, 0)
            x2 = min(center_x + removal_half_width + 1, image_width)
            y1 = max(fragment["y1"], 0)
            y2 = min(fragment["y2"], image_height)
            grouped_vertical_mask[y1:y2, x1:x2] = 255

    return grouped_vertical_mask, fragments, clusters, selected_clusters
