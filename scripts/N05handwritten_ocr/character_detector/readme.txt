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

The current experimental Aristotel-trained checkpoint is:

    models/glyph_classifier_v0_2_aristotel/
        glyph_classifier_v0_2_aristotel_best.pt

Training report:

    reports/glyph_classifier_v0_2_aristotel/training_report.json

Full Matenadata + lightweight Aristotel training produced:

    best epoch: 6
    test top-1: 0.6764
    test top-5: 0.8701

This model is useful evidence, but it is not trusted as a final decision maker.
Manual N05 segment probes still show domain mismatch on some real masks, so the
decision matrix should downweight it until a pipeline-mask-trained CNN exists.

Aristotel Training
------------------
The CNN trainer can now add lightweight Aristotel damage variants during
training without saving generated images to disk. This is meant to teach the
pixel expert damaged/cut/thinned glyphs while keeping validation and test clean.

Example:

    python scripts/Cyber_Lin_Kuei_Assembly/glyph_classifier.py \
      --model-name glyph_classifier_v0_2_aristotel \
      --use-aristotel \
      --aristotel-variants-per-sample 2 \
      --aristotel-recipes light_cut light_erosion threshold_failure light_blur

New checkpoints declare `input_polarity_mode=normalize_black_ink_on_white`, so
white-on-black N05 masks are inverted before the usual `1 - grayscale` tensor
conversion.

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
N05 runs the CNN only on selected assembly segment masks. It should not be run
on whole word/text-unit crops, because that produces loud but misleading
single-letter guesses.

Segment outputs are attached under each selected segment as:

    segment.expert_outputs.character_detector
    segment.character_detector

The current output is evidence for the future decision matrix, not a final OCR
answer.
