Legacy N00 Utilities
====================

These modules belong to the retired preparation branch that detected broad
regions and form lines, then split documents into rows and fields:

    region_detector.py
    line_detector.py
    field_splitter.py

The active N00 pipeline prepares aligned images and masks only. Its current
ScribeMap form-line masks live in file_preparation_scribemap_masks.py and are
not replaced by these legacy modules.

Nothing in this folder is imported by the active pipeline.
