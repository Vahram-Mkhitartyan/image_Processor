Character Detector Expert
=========================

Purpose
-------
This folder is the new home of the character-level Armenian recognizer that
will be remade from the current glyph OCR work.

Current Assets
--------------
    numeric_label_map.json
    scan_matenadata.py

The training and evaluation scripts remain in Cyber_Lin_Kuei_Assembly, but now
read the label map from this expert folder.

Interface
---------
    expert.py
    get_expert_manifest(settings=None)
    recognize(crop_path, context=None, settings=None)
