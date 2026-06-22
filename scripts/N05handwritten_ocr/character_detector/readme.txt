Character Detector CNN
======================

Purpose
-------
The character detector is N05's pixel-based Armenian glyph expert. It consumes
one character crop and returns an independent top-k candidate list compatible
with ANTAR and the future N05 evidence assembler.

Model
-----
The active baseline is:

    models/glyph_classifier_v0_1/glyph_classifier_v0_1_best.pt

It is a four-block PyTorch CNN trained on 78 Matenadata classes at 64x64. Model
weights are cached after the first prediction.

Input Contract
--------------
The preferred input is an exact single-character analysis mask. This model was
trained on the original Matenadata convention: white glyph ink on a black
background, followed by unconditional `1 - grayscale` tensor conversion.

Because v0.1 did not persist an explicit polarity contract, inference labels it
`legacy_raw_invert` and reproduces training exactly. Future retrained checkpoints
may declare `input_polarity_mode=normalize_black_ink_on_white` without changing
the expert JSON schema.

Output Contract
---------------
Each candidate contains:

    rank
    class_id
    label
    text
    confidence
    source
    evidence_kind=pixel_cnn
    provenance=character_detector_cnn

Evidence also records preprocessing, model metadata, top-1 margin, normalized
entropy, and schema_version=n05_candidate_evidence_v1.

Standalone Test
---------------

    python scripts/N05handwritten_ocr/character_detector/expert.py \
      Matenadata/8/3.png \
      --out temp_processing/character_detector_smoke.json

N05 Integration
---------------
The implementation is ready, but the expert remains disabled in settings.json.
N05 currently owns word/text-unit crops; the CNN must be enabled only after a
character segmentation hypothesis has been selected.
