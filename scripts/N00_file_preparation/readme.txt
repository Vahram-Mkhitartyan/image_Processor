Node N00: File Preparation
==========================

Purpose
-------
N00 normalizes each document and creates every binary/visual mask consumed by
ScribeMap and later crop-generation stages. It prepares pixels; it does not
group text or run OCR.

Main Entry
----------

    file_preparation.py
    prepare_file(input_path, output_dir, steps, settings=None)

Active Pipeline Steps
---------------------

    load_image
    rotate_major
    convert_to_grayscale
    denoise_image
    improve_contrast
    threshold_image
    deskew_image
    crop_white_margins
    create_scribemap_masks
    create_color_layer_masks
    save_outputs

The color image is transformed with the same rotation, deskew, and crop bounds
as the grayscale image so all later bounding boxes share one coordinate space.

Main Files
----------
file_preparation.py:
    Pipeline state, step registry, settings, and orchestration.

image_preprocessor.py:
    Loading, rotation, grayscale conversion, denoising, contrast, thresholding,
    deskewing, and white-margin cropping.

file_preparation_scribemap_masks.py:
    Form-line masks, content mask, exclusive color masks, dark-color recovery,
    and isolated visual layer images.

artifact_saver.py:
    Writes images, masks, visual layers, and metadata.

image_quality.py:
    Basic image-quality measurements.

Retired region, line, row, and field splitting utilities are preserved under:

    scripts/legacy/N00_file_preparation/

They are not imported or registered by active N00.

Output Folder
-------------

    temp_processing/<document_id>/n00_file_preparation/
        full_images/
        lines/
        masks/
        metadata/

Important Full Images
---------------------

    02_gray.jpeg
    03_denoised.jpeg
    04_enhanced.jpeg
    05_thresholded.jpeg
    06_deskewed.jpeg
    07_cropped.jpeg
    08_red_ink_layer.jpeg
    09_blue_ink_layer.jpeg
    10_green_ink_layer.jpeg
    11_unknown_color_ink_layer.jpeg
    12_colored_ink_layer.jpeg
    13_black_ink_layer.jpeg

Important Masks
---------------

    01_black_pixel_mask.png
    02_horizontal_line_mask.png
    03_short_horizontal_line_mask.png
    04_combined_horizontal_line_mask.png
    05_vertical_line_mask.png
    06_grouped_vertical_line_mask.png
    07_content_ink_mask.png
    08_red_ink_mask.png
    09_blue_ink_mask.png
    10_green_ink_mask.png
    11_unknown_color_ink_mask.png
    12_colored_ink_mask.png
    13_black_ink_mask.png

Color Segmentation
------------------
The final red, blue, green, unknown-color, and black masks are exclusive. Strong
color seeds use hue, saturation, chroma, value, and channel-dominance checks.
Weak color pixels may join a layer only near a strong seed of the same color.
This preserves faded strokes without classifying warm paper as red ink.

Dark colored pixels near established color strokes are recovered using local
neighborhoods and winner-take-all hue assignment.

After classification, a one-pixel seeded edge-recovery pass admits weaker
same-color source pixels only when they directly continue trusted red, blue, or
green ink. Unlike ordinary dilation, this restores faded stroke width without
growing into unrelated paper texture. Recovery counts are stored in metadata as
seeded_red_edge_pixels, seeded_blue_edge_pixels, and seeded_green_edge_pixels.

Key returned artifacts include:

    state["artifacts"]["cropped"]
    state["artifacts"]["content_ink_mask"]
    state["artifacts"]["blue_ink_mask"]
    state["artifacts"]["red_ink_mask"]
    state["artifacts"]["green_ink_mask"]
    state["artifacts"]["unknown_color_ink_mask"]
    state["artifacts"]["black_ink_mask"]

When To Edit N00
----------------
Edit N00 when deskew, crop alignment, form-line removal, or color/mask separation
is wrong. Do not tune N00 to change ScribeMap grouping behavior; that belongs in
N01.
