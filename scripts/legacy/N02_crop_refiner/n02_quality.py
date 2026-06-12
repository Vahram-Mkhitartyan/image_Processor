"""4x4 ink-density quality scoring for N02 crops."""

import cv2

from n02_io import crop_image

def build_ink_mask(gray_crop, settings):
    """Convert a grayscale crop into a binary ink mask.

    Args:
        gray_crop: Grayscale crop image array.
        settings: RefinerSettings instance.

    Returns:
        Binary mask where ink pixels are 1 and background pixels are 0.
    """
    return (gray_crop < settings.ink_threshold).astype("uint8")


def density_for_region(mask_region):
    """Calculate ink density for a mask region.

    Args:
        mask_region: Binary mask region.

    Returns:
        Ink density from 0.0 to 1.0.
    """
    if mask_region.size == 0:
        return 0.0

    return float(mask_region.mean())


def grid_density_4x4(ink_mask):
    """Split a binary ink mask into a 4x4 grid and measure each zone.

    Args:
        ink_mask: Binary crop mask where ink is 1.

    Returns:
        4x4 nested list of density values.
    """
    height, width = ink_mask.shape[:2]
    grid = []

    for row in range(4):
        row_values = []
        y1 = int(round(row * height / 4))
        y2 = int(round((row + 1) * height / 4))

        for col in range(4):
            x1 = int(round(col * width / 4))
            x2 = int(round((col + 1) * width / 4))
            row_values.append(density_for_region(ink_mask[y1:y2, x1:x2]))

        grid.append(row_values)

    return grid


def summarize_grid_densities(grid):
    """Build named edge/core density summaries from a 4x4 grid.

    Args:
        grid: 4x4 nested list of density values.

    Returns:
        Dictionary with total edge and center density summaries.
    """
    left_edge = sum(grid[row][0] for row in range(4)) / 4.0
    right_edge = sum(grid[row][3] for row in range(4)) / 4.0
    top_edge = sum(grid[0][col] for col in range(4)) / 4.0
    bottom_edge = sum(grid[3][col] for col in range(4)) / 4.0
    center = (
        grid[1][1]
        + grid[1][2]
        + grid[2][1]
        + grid[2][2]
    ) / 4.0

    return {
        "left": left_edge,
        "right": right_edge,
        "top": top_edge,
        "bottom": bottom_edge,
        "center": center,
        "max_edge": max(left_edge, right_edge, top_edge, bottom_edge),
    }


def collect_quality_infractions(total_density, density_summary, settings):
    """Convert crop density evidence into human-readable infractions.

    Args:
        total_density: Overall ink density for the crop.
        density_summary: Named density summary from summarize_grid_densities().
        settings: RefinerSettings instance.

    Returns:
        List of infraction strings.
    """
    infractions = []

    if total_density < settings.min_ink_density:
        infractions.append("too_little_ink")

    if total_density > settings.max_ink_density:
        infractions.append("too_much_ink")

    if density_summary["center"] < settings.center_density_low:
        infractions.append("empty_center")

    if density_summary["left"] > settings.edge_density_high:
        infractions.append("left_edge_dense")

    if density_summary["right"] > settings.edge_density_high:
        infractions.append("right_edge_dense")

    if density_summary["top"] > settings.edge_density_high:
        infractions.append("top_edge_dense")

    if density_summary["bottom"] > settings.edge_density_high:
        infractions.append("bottom_edge_dense")

    return infractions


def score_crop_quality(total_density, density_summary, infractions):
    """Calculate a deterministic quality score for one crop.

    Args:
        total_density: Overall ink density for the crop.
        density_summary: Named density summary from summarize_grid_densities().
        infractions: List of detected quality issues.

    Returns:
        Score from 0.0 to 100.0.
    """
    score = 100.0

    if "too_little_ink" in infractions:
        score -= 45.0

    if "too_much_ink" in infractions:
        score -= 45.0

    if "empty_center" in infractions:
        score -= 20.0

    edge_penalty = density_summary["max_edge"] * 30.0
    score -= min(edge_penalty, 25.0)

    # A small amount of ink is good; this lightly rewards useful-looking crops
    # without letting density overpower the hard infractions above.
    if 0.02 <= total_density <= 0.35:
        score += 5.0

    return max(0.0, min(score, 100.0))


def resolve_quality_status(score, infractions, settings):
    """Assign accepted/review from score and infractions.

    Args:
        score: Numeric quality score.
        infractions: List of quality issue strings.
        settings: RefinerSettings instance.

    Returns:
        Status string. N02 does not reject final crops; questionable crops go to
        review so the classifier can decide.
    """
    if infractions:
        return "review"

    if score >= settings.accepted_score_threshold:
        return "accepted"

    return "review"


def evaluate_crop_quality(gray_image, bbox, settings):
    """Evaluate one crop using deterministic ink-density evidence.

    Args:
        gray_image: Grayscale source image array.
        bbox: Crop bbox to evaluate.
        settings: RefinerSettings instance.

    Returns:
        Quality result dictionary containing score, status, grid, and evidence.
    """
    crop = crop_image(gray_image, bbox)
    ink_mask = build_ink_mask(crop, settings)
    grid = grid_density_4x4(ink_mask)
    density_summary = summarize_grid_densities(grid)
    total_density = density_for_region(ink_mask)
    infractions = collect_quality_infractions(
        total_density=total_density,
        density_summary=density_summary,
        settings=settings,
    )
    score = score_crop_quality(
        total_density=total_density,
        density_summary=density_summary,
        infractions=infractions,
    )
    status = resolve_quality_status(
        score=score,
        infractions=infractions,
        settings=settings,
    )

    return {
        "status": status,
        "score": round(score, 4),
        "infractions": infractions,
        "total_density": round(total_density, 6),
        "density_summary": {
            key: round(value, 6)
            for key, value in density_summary.items()
        },
        "grid_density_4x4": [
            [round(value, 6) for value in row]
            for row in grid
        ],
    }
